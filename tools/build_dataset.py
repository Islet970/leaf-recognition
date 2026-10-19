#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the class registry, stratified split manifests and the real-scene manifest.

Flavia mode (default):
    python tools/build_dataset.py --classes 4,12,15,17,20 --subset 5class \
        --out data/splits --registry data/class_registry.json --seed 42 --materialize

Real-scene mode:
    python tools/build_dataset.py --real-dir data/leaf --subset real --out data/splits

The manifest column order is fixed:
    filename, image_id, flavia_label, py_label, key, zh_name, scientific, common, split, sha256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = [
    "filename", "image_id", "flavia_label", "py_label", "key",
    "zh_name", "scientific", "common", "split", "sha256",
]
SPLITS = ("train", "val", "test")
FLAVIA_RE = re.compile(r"^(\d{4})\.jpg$", re.IGNORECASE)

DEFAULT_CLASSES = "4,12,15,17,20"
DEFAULT_CLASS_MAP = "data/class_map.json"
DEFAULT_ZH_NAMES = "data/zh_names.json"

# Expected counts for the fixed 5-class training subset (plans.md §1.3).
EXPECTED_5CLASS = {
    4:  {"total": 67, "train": 41, "val": 13, "test": 13},
    12: {"total": 55, "train": 33, "val": 11, "test": 11},
    15: {"total": 61, "train": 37, "val": 12, "test": 12},
    17: {"total": 54, "train": 32, "val": 11, "test": 11},
    20: {"total": 57, "train": 35, "val": 11, "test": 11},
}

# data/leaf/<dir> -> Flavia label (plans.md §1.5). Real frames never enter train/val.
REAL_DIR_MAP = [
    ("camphor_test", 15),
    ("cercis_test", 4),
    ("lagerstroemia_test", 20),
    ("osmanthus_test", 17),
    ("pittosporum_test", 12),
]
EXPECTED_REAL_COUNTS = {
    "camphor_test": 160,
    "cercis_test": 163,
    "lagerstroemia_test": 155,
    "osmanthus_test": 145,
    "pittosporum_test": 168,
}
# Fallback py_label mapping: identical to the default 5class --classes order.
DEFAULT_REAL_PY_LABELS = {4: 0, 12: 1, 15: 2, 17: 3, 20: 4}


def as_posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# Mapping / selection
# --------------------------------------------------------------------------- #
def load_class_map(path: str) -> dict:
    """Read data/class_map.json and validate labels, keys and ranges."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read class map {path}: {exc}")

    rows = data.get("classes")
    if not isinstance(rows, list) or not rows:
        fail(f"{path}: 'classes' must be a non-empty list")

    required = ("flavia_label", "key", "zh_name", "scientific", "common", "range")
    labels, keys = set(), set()
    for row in rows:
        missing = [field for field in required if field not in row]
        if missing:
            fail(f"{path}: class row missing {missing}: {row}")
        label = row["flavia_label"]
        if not isinstance(label, int):
            fail(f"{path}: flavia_label must be an int, got {label!r}")
        if label in labels:
            fail(f"{path}: duplicate flavia_label {label}")
        labels.add(label)
        key = row["key"]
        if not isinstance(key, str) or not key:
            fail(f"{path}: invalid key for label {label}")
        if key in keys:
            fail(f"{path}: duplicate key {key}")
        keys.add(key)
        rng = row["range"]
        if (not isinstance(rng, (list, tuple)) or len(rng) != 2
                or not all(isinstance(value, int) for value in rng)
                or rng[0] > rng[1]):
            fail(f"{path}: invalid range for label {label}: {rng!r}")

    ordered = sorted(rows, key=lambda row: row["range"][0])
    for previous, current in zip(ordered, ordered[1:]):
        if current["range"][0] <= previous["range"][1]:
            fail(f"{path}: overlapping ranges for labels "
                 f"{previous['flavia_label']} and {current['flavia_label']}")
    return data


def select_classes(rows: list[dict], spec: str) -> list[dict]:
    """Select classes by 'all', comma-separated Flavia labels, or comma-separated keys.

    Returned order (and therefore py_label order) follows the written order.
    """
    spec = spec.strip()
    if not spec:
        fail("--classes must not be empty")
    if spec.lower() == "all":
        return list(rows)

    by_label = {row["flavia_label"]: row for row in rows}
    by_key = {row["key"].lower(): row for row in rows}
    selected, seen = [], set()
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            fail(f"--classes contains an empty entry: {spec!r}")
        if re.fullmatch(r"\d+", token):
            row = by_label.get(int(token))
            label = int(token)
        else:
            row = by_key.get(token.lower())
            label = row["flavia_label"] if row else token
        if row is None:
            fail(f"--classes entry {token!r} is not in the class map")
        if row["flavia_label"] in seen:
            fail(f"--classes entry {token!r} is selected twice")
        seen.add(row["flavia_label"])
        selected.append(row)
    return selected


def load_zh_names(path: str) -> dict:
    try:
        table = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read zh names {path}: {exc}")
    if not isinstance(table, dict):
        fail(f"{path}: expected a JSON object {{flavia_label: zh_name}}")
    return table


# --------------------------------------------------------------------------- #
# Assignment / split / leakage
# --------------------------------------------------------------------------- #
def scan_leaf_files(leaves_dir: str) -> list[tuple[int, str, Path]]:
    """Return (image_id, filename, path) for 4-digit .jpg files, sorted by image_id."""
    root = Path(leaves_dir)
    if not root.is_dir():
        fail(f"leaves directory not found: {leaves_dir}")
    files, bad = [], []
    for entry in os.scandir(root):
        if not entry.is_file() or not entry.name.lower().endswith(".jpg"):
            continue
        match = FLAVIA_RE.fullmatch(entry.name)
        if not match:
            bad.append(entry.name)
            continue
        files.append((int(match.group(1)), entry.name, Path(entry.path)))
    if bad:
        fail(f"{leaves_dir}: filenames must be 4 digits + .jpg; bad: {bad[:10]}")
    files.sort(key=lambda item: item[0])
    return files


def assign_by_ranges(leaf_files: list[tuple[int, str, Path]],
                     selected: list[dict]) -> tuple[list[dict], int]:
    """Assign files inside a selected range to that class; ignore files outside."""
    records, ignored = [], 0
    for image_id, filename, path in leaf_files:
        hits = [row for row in selected
                if row["range"][0] <= image_id <= row["range"][1]]
        if len(hits) > 1:
            labels = [row["flavia_label"] for row in hits]
            fail(f"{filename} falls into multiple selected ranges: {labels}")
        if not hits:
            ignored += 1
            continue
        row = hits[0]
        records.append({
            "filename": filename,
            "image_id": image_id,
            "flavia_label": row["flavia_label"],
            "py_label": selected.index(row),
            "key": row["key"],
            "zh_name": row["zh_name"],
            "scientific": row["scientific"],
            "common": row["common"],
            "path": str(path),
        })

    if not records:
        fail("no images matched the selected classes")
    per_class = defaultdict(int)
    for record in records:
        per_class[record["flavia_label"]] += 1
    empty = [row["flavia_label"] for row in selected if per_class[row["flavia_label"]] == 0]
    if empty:
        fail(f"selected classes have no images: {empty}")
    names = [record["filename"] for record in records]
    if len(names) != len(set(names)):
        fail("duplicate filenames among selected images")
    return records, ignored


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_hashes(records: list[dict]) -> None:
    for record in records:
        record["sha256"] = sha256_file(record["path"])


def stratified_split(records: list[dict], seed: int) -> None:
    """Per class: sort by image id, shuffle with Random(seed + flavia_label).

    First n_val records -> val, next n_test -> test, remainder -> train.
    """
    grouped = defaultdict(list)
    for record in records:
        grouped[record["flavia_label"]].append(record)
    for label, items in grouped.items():
        items.sort(key=lambda record: record["image_id"])
        random.Random(seed + label).shuffle(items)
        n_val = int(round(0.2 * len(items)))
        n_test = int(round(0.2 * len(items)))
        for index, record in enumerate(items):
            if index < n_val:
                record["split"] = "val"
            elif index < n_val + n_test:
                record["split"] = "test"
            else:
                record["split"] = "train"


def leakage_check(records: list[dict]) -> dict:
    filename_sets = {split: set() for split in SPLITS}
    for record in records:
        filename_sets[record["split"]].add(record["filename"])

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = filename_sets[first] & filename_sets[second]
        if overlap:
            fail(f"leakage: {len(overlap)} filenames in both {first} and {second} "
                 f"(e.g. {sorted(overlap)[:5]})")

    union = set().union(*filename_sets.values())
    all_names = {record["filename"] for record in records}
    if union != all_names:
        fail(f"split union mismatch: union={len(union)} total={len(all_names)}")

    sha_to_classes, sha_to_rows = defaultdict(set), defaultdict(list)
    for record in records:
        sha_to_classes[record["sha256"]].add(record["flavia_label"])
        sha_to_rows[record["sha256"]].append(record)
    cross = {sha: labels for sha, labels in sha_to_classes.items() if len(labels) > 1}
    if cross:
        detail = ", ".join(f"{sha[:12]}={sorted(labels)}" for sha, labels in list(cross.items())[:5])
        fail(f"cross-class duplicate images (same SHA-256): {len(cross)} groups ({detail})")
    duplicates_within = sum(
        1 for rows in sha_to_rows.values()
        if len(rows) > 1 and len({row["flavia_label"] for row in rows}) == 1
    )
    return {
        "pairwise_disjoint": True,
        "union_matches_total": True,
        "cross_class_duplicate_sha256": 0,
        "duplicate_sha256_within_class": duplicates_within,
        "unique_sha256": len(sha_to_rows),
        "total_images": len(records),
    }


# --------------------------------------------------------------------------- #
# Registry / manifest writers
# --------------------------------------------------------------------------- #
def build_registry(selected: list[dict], zh_table: dict, counts: dict,
                   seed: int, subset: str, source: str) -> dict:
    classes = []
    for py_label, row in enumerate(selected):
        label = row["flavia_label"]
        zh_name = zh_table.get(str(label), row["zh_name"])
        if zh_name != row["zh_name"]:
            fail(f"zh_names.json disagrees with class_map for label {label}: "
                 f"{zh_name!r} != {row['zh_name']!r}")
        classes.append({
            "flavia_label": label,
            "py_label": py_label,
            "key": row["key"],
            "zh_name": zh_name,
            "scientific": row["scientific"],
            "common": row["common"],
            "range": list(row["range"]),
            "image_count": counts[label],
            "subset": subset,
        })
    return {"source": source, "seed": seed, "subset": subset, "classes": classes}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_sort_key(record: dict):
    return (record["py_label"], record["image_id"], record["filename"])


def write_manifests(records: list[dict], out_root: str, subset: str) -> Path:
    out_dir = Path(out_root) / subset
    rows = [{column: record[column] for column in COLUMNS} for record in records]
    for split in SPLITS:
        split_rows = sorted((r for r in rows if r["split"] == split), key=row_sort_key)
        write_csv(out_dir / f"{split}.csv", split_rows)
    write_csv(out_dir / "all.csv", sorted(rows, key=row_sort_key))
    return out_dir


def write_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize(records: list[dict], dataset_dir: str, subset: str) -> dict:
    counts = {"hardlink": 0, "copy": 0}
    for record in records:
        dest_dir = Path(dataset_dir) / subset / record["split"] / record["key"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / record["filename"]
        if dest.exists():
            dest.unlink()
        try:
            os.link(record["path"], dest)
            counts["hardlink"] += 1
        except OSError:
            shutil.copy2(record["path"], dest)
            counts["copy"] += 1
    return counts


# --------------------------------------------------------------------------- #
# Validation (fresh reads of generated artifacts)
# --------------------------------------------------------------------------- #
def read_csv_checked(path: Path) -> list[dict]:
    if not path.is_file():
        fail(f"manifest not found after writing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            fail(f"{path}: header is {reader.fieldnames}, expected {COLUMNS}")
        return list(reader)


def validate_flavia_outputs(registry_path: Path, class_map: dict, selected: list[dict],
                            out_dir: Path, records: list[dict], subset: str,
                            seed: int) -> dict:
    """Re-read artifacts, verify counts/columns/registry against class_map."""
    expected_counts = {split: sum(1 for r in records if r["split"] == split)
                       for split in SPLITS}
    read_counts = {}
    for split in SPLITS:
        rows = read_csv_checked(out_dir / f"{split}.csv")
        read_counts[split] = len(rows)
        if len(rows) != expected_counts[split]:
            fail(f"{out_dir / (split + '.csv')}: {len(rows)} rows, "
                 f"expected {expected_counts[split]}")
        if any(row["split"] != split for row in rows):
            fail(f"{out_dir / (split + '.csv')}: unexpected split value")
    all_rows = read_csv_checked(out_dir / "all.csv")
    if len(all_rows) != len(records):
        fail(f"{out_dir / 'all.csv'}: {len(all_rows)} rows, expected {len(records)}")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if [row["py_label"] for row in registry["classes"]] != list(range(len(selected))):
        fail("registry py_label is not a contiguous 0..N-1 sequence")
    source_by_label = {row["flavia_label"]: row for row in class_map["classes"]}
    for index, (entry, chosen) in enumerate(zip(registry["classes"], selected)):
        label = entry["flavia_label"]
        if entry["py_label"] != index:
            fail(f"registry py_label mismatch for label {label}")
        if entry["key"] != chosen["key"] or list(entry["range"]) != list(chosen["range"]):
            fail(f"registry key/range mismatch for label {label}")
        original = source_by_label[label]
        if entry["key"] != original["key"] or list(entry["range"]) != list(original["range"]):
            fail(f"registry differs from class_map for label {label}")
        expected_images = sum(1 for r in records if r["flavia_label"] == label)
        if entry["image_count"] != expected_images:
            fail(f"registry image_count mismatch for label {label}: "
                 f"{entry['image_count']} != {expected_images}")
        if entry["subset"] != subset:
            fail(f"registry subset mismatch for label {label}")
    if registry["seed"] != seed or registry["subset"] != subset:
        fail("registry seed/subset mismatch")

    return {"read_counts": read_counts, "all_rows": len(all_rows),
            "py_labels": [row["py_label"] for row in registry["classes"]]}


def expected_5class_status(selected: list[dict], count_rows: dict) -> str:
    labels = sorted(row["flavia_label"] for row in selected)
    if labels != sorted(EXPECTED_5CLASS):
        return "skipped (class set is not the fixed five)"
    for label, expected in EXPECTED_5CLASS.items():
        actual = count_rows[label]
        for key in ("total", "train", "val", "test"):
            if actual[key] != expected[key]:
                fail(f"5class count mismatch for label {label} {key}: "
                     f"{actual[key]} != {expected[key]}")
    return "PASS"


# --------------------------------------------------------------------------- #
# Real-scene manifest
# --------------------------------------------------------------------------- #
def real_py_labels(registry_path: Path) -> tuple[dict, str]:
    """Use the trained 5class registry's py_label order when available."""
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            mapping = {row["flavia_label"]: row["py_label"]
                       for row in registry.get("classes", [])
                       if "flavia_label" in row and "py_label" in row}
            if all(label in mapping for _, label in REAL_DIR_MAP):
                return mapping, f"{as_posix(registry_path)} (py_label from registry)"
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_REAL_PY_LABELS, "default 5class order (registry not usable)"


def run_real(args) -> None:
    class_map = load_class_map(args.class_map)
    selected = select_classes(class_map["classes"], DEFAULT_CLASSES)
    by_label = {row["flavia_label"]: row for row in selected}
    zh_table = load_zh_names(args.zh_names)
    py_labels, py_source = real_py_labels(Path(args.registry))

    real_root = Path(args.real_dir)
    if not real_root.is_dir():
        fail(f"real directory not found: {args.real_dir}")

    records, seen_dirs = [], set()
    for dirname, label in REAL_DIR_MAP:
        folder = real_root / dirname
        if not folder.is_dir():
            fail(f"missing real-scene directory: {folder}")
        seen_dirs.add(dirname)
        sources = sorted(
            (entry for entry in os.scandir(folder)
             if entry.is_file() and entry.name.lower().endswith(".jpg")),
            key=lambda entry: entry.name,
        )
        if len(sources) != EXPECTED_REAL_COUNTS[dirname]:
            fail(f"{folder}: {len(sources)} jpg files, "
                 f"expected {EXPECTED_REAL_COUNTS[dirname]}")
        row = by_label[label]
        zh_name = zh_table.get(str(label), row["zh_name"])
        if zh_name != row["zh_name"]:
            fail(f"zh_names.json disagrees with class_map for label {label}")
        for entry in sources:
            relative = f"{dirname}/{entry.name}"
            records.append({
                "filename": relative,
                # Frame basenames repeat across directories, so the directory
                # is part of the unique image id as well.
                "image_id": f"{dirname}/{Path(entry.name).stem}",
                "flavia_label": label,
                "py_label": py_labels[label],
                "key": row["key"],
                "zh_name": zh_name,
                "scientific": row["scientific"],
                "common": row["common"],
                "split": "test",
                "sha256": sha256_file(entry.path),
                "path": entry.path,
            })

    extra = sorted(
        entry.name for entry in os.scandir(real_root)
        if entry.is_dir() and entry.name.endswith("_test") and entry.name not in seen_dirs
    )
    if extra:
        print(f"WARNING: ignored unmapped *_test directories: {extra}")

    if len(records) != sum(EXPECTED_REAL_COUNTS.values()):
        fail(f"real total {len(records)} != {sum(EXPECTED_REAL_COUNTS.values())}")
    if len({record["image_id"] for record in records}) != len(records):
        fail("real image_id values are not unique")

    subset = args.subset or "real"
    out_dir = Path(args.out) / subset
    records.sort(key=row_sort_key)
    write_csv(out_dir / "all.csv", [{column: record[column] for column in COLUMNS}
                                    for record in records])

    # Fresh read-back validation.
    rows = read_csv_checked(out_dir / "all.csv")
    if len(rows) != sum(EXPECTED_REAL_COUNTS.values()):
        fail(f"{out_dir / 'all.csv'}: {len(rows)} rows, expected 791")
    if any(row["split"] != "test" for row in rows):
        fail(f"{out_dir / 'all.csv'}: every row must have split=test")
    per_dir = defaultdict(int)
    per_label = defaultdict(int)
    for row in rows:
        per_dir[row["filename"].split("/", 1)[0]] += 1
        per_label[int(row["flavia_label"])] += 1
    for dirname in EXPECTED_REAL_COUNTS:
        if per_dir[dirname] != EXPECTED_REAL_COUNTS[dirname]:
            fail(f"real manifest count mismatch for {dirname}: "
                 f"{per_dir[dirname]} != {EXPECTED_REAL_COUNTS[dirname]}")

    print("REAL SCENE MANIFEST")
    print(f"  real dir : {as_posix(real_root)}")
    print(f"  manifest : {as_posix(out_dir / 'all.csv')}")
    print(f"  py_label : {py_source}")
    print(f"  rows     : {len(rows)} (expected 791)")
    print(f"  split    : all test")
    print("  directories:")
    for dirname, label in REAL_DIR_MAP:
        folder_count = sum(1 for entry in os.scandir(real_root / dirname)
                           if entry.is_file() and entry.name.lower().endswith(".jpg"))
        print(f"    {dirname:<22} label={label:>2} py={py_labels[label]} "
              f"{folder_count:>3} files -> csv {per_dir[dirname]:>3}")
    print("  class totals: " + ", ".join(
        f"{label}:{per_label[label]}" for _, label in REAL_DIR_MAP))
    print("  CSV header : " + ",".join(COLUMNS))
    print("  VALIDATION : PASS (791 rows, split=test, counts and header verified)")


# --------------------------------------------------------------------------- #
# Flavia mode
# --------------------------------------------------------------------------- #
def run_flavia(args) -> None:
    subset = args.subset or "5class"
    class_map = load_class_map(args.class_map)
    selected = select_classes(class_map["classes"], args.classes)
    zh_table = load_zh_names(args.zh_names)

    leaf_files = scan_leaf_files(args.leaves)
    records, ignored = assign_by_ranges(leaf_files, selected)
    attach_hashes(records)
    stratified_split(records, args.seed)
    leakage = leakage_check(records)

    counts = {row["flavia_label"]: defaultdict(int) for row in selected}
    for record in records:
        counts[record["flavia_label"]]["total"] += 1
        counts[record["flavia_label"]][record["split"]] += 1
    split_totals = {split: sum(1 for r in records if r["split"] == split)
                    for split in SPLITS}
    expected_status = expected_5class_status(selected, counts)

    registry = build_registry(selected, zh_table,
                              {label: counter["total"] for label, counter in counts.items()},
                              args.seed, subset, args.class_map)
    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    out_dir = write_manifests(records, args.out, subset)
    stats = {
        "subset": subset,
        "seed": args.seed,
        "source": args.class_map,
        "leaves_dir": as_posix(args.leaves),
        "total_images": len(records),
        "split_totals": split_totals,
        "classes": {
            row["key"]: {
                "flavia_label": row["flavia_label"],
                "py_label": selected.index(row),
                "total": counts[row["flavia_label"]]["total"],
                "train": counts[row["flavia_label"]]["train"],
                "val": counts[row["flavia_label"]]["val"],
                "test": counts[row["flavia_label"]]["test"],
            }
            for row in selected
        },
        "leakage": leakage,
        "expected_5class_check": expected_status,
        "ignored_out_of_range": ignored,
        "manifests": {name: as_posix(out_dir / f"{name}.csv")
                      for name in ("train", "val", "test", "all")},
    }
    write_stats(out_dir / "split_stats.json", stats)

    if args.materialize:
        linked = materialize(records, args.dataset_dir, subset)
    else:
        linked = None

    readback = validate_flavia_outputs(registry_path, class_map, selected,
                                       out_dir, records, subset, args.seed)

    print("FLAVIA DATASET BUILD")
    print(f"  class map : {args.class_map}")
    print(f"  leaves    : {as_posix(args.leaves)} ({len(leaf_files)} jpg files, "
          f"{ignored} out of range ignored)")
    print(f"  registry  : {as_posix(registry_path)}")
    print(f"  manifests : {as_posix(out_dir)}/{{train,val,test,all}}.csv")
    print(f"  seed      : {args.seed} (Random(seed + flavia_label) per class)")
    print("  class counts (flavia_label py_label key total train/val/test):")
    for entry in registry["classes"]:
        key = entry["key"]
        c = stats["classes"][key]
        print(f"    {entry['flavia_label']:>2}  py={entry['py_label']}  "
              f"{key:<24} {c['total']:>3}  "
              f"{c['train']:>3}/{c['val']:>2}/{c['test']:>2}  {entry['zh_name']}")
    print(f"  totals    : {len(records)} images -> "
          f"train/val/test = {split_totals['train']}/{split_totals['val']}/{split_totals['test']}")
    print(f"  checks    : train/val/test pairwise disjoint = PASS")
    print(f"              union == selected ({len(records)}) = PASS")
    print(f"              cross-class duplicate SHA-256 = {leakage['cross_class_duplicate_sha256']}")
    print(f"              duplicate SHA-256 within a class = "
          f"{leakage['duplicate_sha256_within_class']}")
    print(f"              registry py_label = {readback['py_labels']}; "
          f"fields match class_map = PASS")
    print(f"              5class expected counts = {expected_status}")
    print(f"  CSV re-read: header OK, rows train/val/test/all = "
          f"{readback['read_counts']['train']}/{readback['read_counts']['val']}/"
          f"{readback['read_counts']['test']}/{readback['all_rows']}")
    if linked is not None:
        print(f"  materialize: {as_posix(Path(args.dataset_dir) / subset)} "
              f"(hardlink={linked['hardlink']}, copy={linked['copy']})")
    print("  VALIDATION : PASS" if expected_status == "PASS" else
          f"  VALIDATION : PASS (fixed-5class check {expected_status})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build class registry and train/val/test manifests for Flavia leaves.")
    parser.add_argument("--leaves", default="data/Leaves", help="Flavia image directory")
    parser.add_argument("--classes", default=DEFAULT_CLASSES,
                        help="comma-separated Flavia labels, keys, or 'all' "
                             f"(default: {DEFAULT_CLASSES})")
    parser.add_argument("--subset", default=None,
                        help="subset name (default: 5class; real mode default: real)")
    parser.add_argument("--out", default="data/splits", help="output root for manifests")
    parser.add_argument("--registry", default="data/class_registry.json",
                        help="class registry path")
    parser.add_argument("--seed", type=int, default=42, help="split seed (default: 42)")
    parser.add_argument("--materialize", action="store_true",
                        help="hard-link (or copy) images into data/dataset/<subset>/...")
    parser.add_argument("--dataset-dir", default="data/dataset",
                        help="materialize destination root")
    parser.add_argument("--class-map", default=DEFAULT_CLASS_MAP,
                        help="authoritative class map JSON")
    parser.add_argument("--zh-names", default=DEFAULT_ZH_NAMES,
                        help="flavia_label -> zh_name JSON")
    parser.add_argument("--real-dir", default=None,
                        help="real-scene frame root (e.g. data/leaf); writes <out>/<subset>/all.csv")
    args = parser.parse_args(argv)
    if args.real_dir and args.materialize:
        parser.error("--materialize is only valid for the Flavia split mode")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.real_dir:
        run_real(args)
    else:
        run_flavia(args)


if __name__ == "__main__":
    main()
