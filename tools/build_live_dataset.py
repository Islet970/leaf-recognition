#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build live-scene manifests for ``data/leaf`` and run leakage checks.

The live frames live in ``data/leaf/<prefix>_test`` (normal) and
``data/leaf/<prefix>_test_hard`` (hard).  File names look like
``leaf04_000010.jpg``; ``leaf01``-``leaf05`` identify the source video.

Hard frames: **all** 227 ``*_test_hard`` frames are written to
``hard_test.csv`` and are used **only as a robustness / stress test set**
(per-class counts 45/39/50/40/53, which are more balanced than the old
rule that kept only leaf04).  They never enter train/val.  Because the
leaf01-leaf03 hard frames come from the same videos as the live training
split, hard_test results must be interpreted as a robustness check, **not**
as an independent validation set.

Modes
-----
``--mode all``
    Writes ``test.csv`` (all normal frames, 432 rows) and
    ``hard_test.csv`` (all hard frames, 227 rows, not filtered by video).
    Used by E1 (W5) and E3b (W32).

``--mode video-split``
    Writes a video-level split that keeps every source video in one split:
    train = leaf01/02/03 (normal, 301), val = leaf05 (normal, 79),
    test = leaf04 (normal, 52), plus ``hard_test.csv`` with **all** hard
    frames (227, not filtered by video).

``py_label`` and ``flavia_label`` always come from the registry, matched by
``key``; they are never inferred from directory order.

Usage
-----
    python tools/build_live_dataset.py --leaf-dir data/leaf --mode all \
        --registry data/class_registry.json --out data/splits/live_all
    python tools/build_live_dataset.py --leaf-dir data/leaf --mode video-split \
        --registry data/class_registry.json --out data/splits/live5
    python tools/build_live_dataset.py --leaf-dir data/leaf --mode all \
        --registry data/class_registry_all32.json --out data/splits/live_all32

Manifest columns are fixed:
    filename, image_id, flavia_label, py_label, key, zh_name, scientific,
    common, split, sha256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

COLUMNS = [
    "filename", "image_id", "flavia_label", "py_label", "key",
    "zh_name", "scientific", "common", "split", "sha256",
]

# data/leaf/<dir prefix> -> registry key (plans.md §1.5).
PREFIX_TO_KEY = {
    "camphor": "cinnamomum_camphora",
    "cercis": "cercis_chinensis",
    "lagerstroemia": "lagerstroemia_indica",
    "osmanthus": "osmanthus_fragrans",
    "pittosporum": "pittosporum_tobira",
}
PREFIX_ORDER = ["camphor", "cercis", "lagerstroemia", "osmanthus", "pittosporum"]

# plans.md §1.5: 432 normal frames, all used for the normal manifests.
EXPECTED_ALL_NORMAL = {
    "camphor": 83, "cercis": 96, "lagerstroemia": 100,
    "osmanthus": 86, "pittosporum": 67,
}
# plans.md §1.5: all 227 hard frames form the robustness/stress test set.
EXPECTED_ALL_HARD = {
    "camphor": 45, "cercis": 39, "lagerstroemia": 50,
    "osmanthus": 40, "pittosporum": 53,
}

# plans.md §1.5: train = leaf01/02/03, val = leaf05, test = leaf04.
# Hard frames always form their own hard_test split (all videos, 227 rows)
# and are never assigned to train/val.
VIDEO_SPLIT_TOTALS = {"train": 301, "val": 79, "test": 52, "hard_test": 227}
EXPECTED_VIDEO_COUNTS = {
    "train": {
        "camphor": 61, "cercis": 63, "lagerstroemia": 76,
        "osmanthus": 58, "pittosporum": 43,
    },
    "val": {
        "camphor": 16, "cercis": 21, "lagerstroemia": 11,
        "osmanthus": 21, "pittosporum": 10,
    },
    "test": {
        "camphor": 6, "cercis": 12, "lagerstroemia": 13,
        "osmanthus": 7, "pittosporum": 14,
    },
    "hard_test": EXPECTED_ALL_HARD,
}
EXPECTED_VIDEO_PREFIXES = {
    "train": {"leaf01", "leaf02", "leaf03"},
    "val": {"leaf05"},
    "test": {"leaf04"},
    # All hard frames are used, so every source video must be present.
    "hard_test": {"leaf01", "leaf02", "leaf03", "leaf04", "leaf05"},
}
PREFIX_TO_SPLIT = {
    "leaf01": "train", "leaf02": "train", "leaf03": "train",
    "leaf04": "test", "leaf05": "val",
}

FRAME_RE = re.compile(r"^(leaf\d{2})_(\d{6})\.jpg$", re.IGNORECASE)
OUTPUT_SPLITS = {"all": ("test", "hard_test"),
                 "video-split": ("train", "val", "test", "hard_test")}
NORMAL_SPLITS = ("train", "val", "test")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def as_posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def load_registry(path: Path) -> dict[str, dict]:
    if not path.is_file():
        fail(f"registry not found: {as_posix(path)}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        fail(f"{as_posix(path)}: missing non-empty 'classes' list")

    by_key: dict[str, dict] = {}
    seen_py_labels: dict[int, str] = {}
    for entry in classes:
        missing = [field for field in
                   ("flavia_label", "py_label", "key", "zh_name",
                    "scientific", "common") if field not in entry]
        if missing:
            fail(f"{as_posix(path)}: class entry missing {missing}: {entry}")
        key = entry["key"]
        py_label = int(entry["py_label"])
        if key in by_key:
            fail(f"{as_posix(path)}: duplicated key {key!r}")
        if py_label in seen_py_labels:
            fail(f"{as_posix(path)}: duplicated py_label {py_label} "
                 f"for {key!r} and {seen_py_labels[py_label]!r}")
        seen_py_labels[py_label] = key
        by_key[key] = entry

    missing_keys = [key for key in PREFIX_TO_KEY.values() if key not in by_key]
    if missing_keys:
        fail(f"{as_posix(path)} does not contain the live-scene classes: "
             f"{missing_keys}")
    return by_key


# --------------------------------------------------------------------------- #
# Frame discovery
# --------------------------------------------------------------------------- #
def discover_dirs(leaf_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (prefix, kind, dir_path) for normal and hard frame directories.

    ``kind`` is ``"normal"`` for ``<prefix>_test`` and ``"hard"`` for
    ``<prefix>_test_hard``.  Hard frames are always collected and written to
    ``hard_test.csv``; they never enter train/val.
    """
    if not leaf_dir.is_dir():
        fail(f"leaf dir not found: {as_posix(leaf_dir)}")
    found: list[tuple[str, str, Path]] = []
    for entry in sorted(leaf_dir.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.endswith("_test_hard"):
            prefix, kind = name[: -len("_test_hard")], "hard"
        elif name.endswith("_test"):
            prefix, kind = name[: -len("_test")], "normal"
        else:
            continue
        if prefix not in PREFIX_TO_KEY:
            fail(f"unknown live-scene directory: {as_posix(entry)}")
        found.append((prefix, kind, entry))

    observed = {(prefix, kind) for prefix, kind, _ in found}
    for prefix in PREFIX_ORDER:
        for kind in ("normal", "hard"):
            if (prefix, kind) not in observed:
                fail(f"missing directory for {prefix} ({kind}): expected "
                     f"{prefix}_test" + ("_hard" if kind == "hard" else ""))
    return found


def collect_frames(leaf_dir: Path, registry: dict[str, dict]) -> list[dict]:
    records: list[dict] = []
    for prefix, kind, directory in discover_dirs(leaf_dir):
        entry = registry[PREFIX_TO_KEY[prefix]]
        files = sorted(
            (item for item in directory.iterdir()
             if item.is_file() and item.suffix.lower() == ".jpg"),
            key=lambda item: item.name,
        )
        if not files:
            fail(f"no .jpg frames in {as_posix(directory)}")
        for path in files:
            match = FRAME_RE.match(path.name)
            if not match:
                fail(f"unexpected frame name: {as_posix(path)}")
            video = match.group(1).lower()
            filename = f"{directory.name}/{path.name}"
            record = {
                "filename": filename,
                "image_id": f"{directory.name}/{path.stem}",
                "flavia_label": int(entry["flavia_label"]),
                "py_label": int(entry["py_label"]),
                "key": entry["key"],
                "zh_name": entry["zh_name"],
                "scientific": entry["scientific"],
                "common": entry["common"],
                "path": str(path),
                "prefix": prefix,
                "kind": kind,
                "video": video,
            }
            records.append(record)
    return records


# --------------------------------------------------------------------------- #
# Assignment, hashing, writing
# --------------------------------------------------------------------------- #
def assign_splits(records: list[dict], mode: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["kind"] == "hard":
            # All hard frames go to hard_test regardless of source video.
            split = "hard_test"
        elif mode == "all":
            split = "test"
        else:
            video = record["video"]
            if video not in PREFIX_TO_SPLIT:
                fail(f"unexpected video prefix {video!r} in {record['filename']}")
            split = PREFIX_TO_SPLIT[video]
        record["split"] = split
        grouped[split].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["py_label"], row["image_id"], row["filename"]))
    return grouped


def attach_hashes(records: list[dict]) -> None:
    cache: dict[str, str] = {}
    for record in records:
        digest = cache.get(record["path"])
        if digest is None:
            digest = sha256_file(Path(record["path"]))
            cache[record["path"]] = digest
        record["sha256"] = digest


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def count_by_prefix(rows: list[dict]) -> dict[str, int]:
    return Counter(row["prefix"] for row in rows)


def check_counts(label: str, rows: list[dict], expected: dict[str, int]) -> None:
    actual = count_by_prefix(rows)
    if len(rows) != sum(expected.values()):
        fail(f"{label}: expected {sum(expected.values())} rows, got {len(rows)}")
    for prefix in PREFIX_ORDER:
        if actual.get(prefix, 0) != expected[prefix]:
            fail(f"{label}: {prefix} expected {expected[prefix]} rows, "
                 f"got {actual.get(prefix, 0)}")
    detail = " ".join(f"{prefix}={actual[prefix]}" for prefix in PREFIX_ORDER)
    print(f"  {label}: {len(rows)} rows | {detail}")


def check_video_isolation(grouped: dict[str, list[dict]]) -> None:
    videos = {
        split: {row["video"] for row in grouped.get(split, [])}
        for split in ("train", "val", "test", "hard_test")
    }
    for split, expected in EXPECTED_VIDEO_PREFIXES.items():
        if videos[split] != expected:
            fail(f"{split}: expected video prefixes {sorted(expected)}, "
                 f"got {sorted(videos[split])}")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = videos[first] & videos[second]
        if overlap:
            fail(f"video leakage: {sorted(overlap)} appears in both "
                 f"{first} and {second}")
    print("  video prefixes: " + " ".join(
        f"{split}={sorted(videos[split])}" for split in
        ("train", "val", "test", "hard_test")))
    print("  note: hard_test intentionally shares videos with train/val/test; "
          "it is a robustness/stress set, not an independent validation set.")


def check_hard_isolation(grouped: dict[str, list[dict]]) -> None:
    """Hard frames must live only in hard_test, never in train/val/test."""
    leaked = [
        row["filename"] for split in NORMAL_SPLITS
        for row in grouped.get(split, []) if row["kind"] == "hard"
    ]
    if leaked:
        fail(f"hard frames leaked into train/val/test: {leaked[:5]}")
    non_hard = [row["filename"] for row in grouped.get("hard_test", [])
                if row["kind"] != "hard"]
    if non_hard:
        fail(f"hard_test contains normal frames: {non_hard[:5]}")
    print("  hard isolation: all hard_* rows are in hard_test only")


def check_sha_leakage(grouped: dict[str, list[dict]]) -> dict:
    """Report cross-split duplicate content by SHA-256.

    A hard failure is raised only when train/val/test share identical
    content.  Byte-identical hard/normal frames are reported as a note
    (hard frames are a robustness set and may overlap by construction).
    """
    split_of: dict[str, set[str]] = defaultdict(set)
    rows_of: dict[str, list[dict]] = defaultdict(list)
    for split, rows in grouped.items():
        for row in rows:
            split_of[row["sha256"]].add(split)
            rows_of[row["sha256"]].append(row)

    core_splits = set(NORMAL_SPLITS)
    core_cross = []
    within_duplicates = 0
    hard_overlap = []
    for digest, splits in split_of.items():
        core_hits = splits & core_splits
        if len(core_hits) > 1:
            core_cross.append((digest, sorted(core_hits), rows_of[digest]))
        elif len(rows_of[digest]) > 1 and len(splits) == 1:
            within_duplicates += 1
        if "hard_test" in splits and core_hits:
            hard_overlap.append((digest, sorted(core_hits), rows_of[digest]))

    print(f"  SHA-256 unique={len(rows_of)} / total={sum(len(v) for v in grouped.values())}")
    print(f"  cross-split duplicate content groups (train/val/test): {len(core_cross)}")
    print(f"  duplicate SHA-256 groups inside a single split: {within_duplicates}")
    if core_cross:
        examples = "; ".join(
            f"{digest[:12]}={splits}" for digest, splits, _ in core_cross[:5])
        fail(f"SHA-256 leakage across train/val/test: {len(core_cross)} groups "
             f"({examples})")

    if hard_overlap:
        example = hard_overlap[0]
        names = ", ".join(row["filename"] for row in example[2][:4])
        core_names = sorted({split for _, splits, _ in hard_overlap
                             for split in splits})
        print(f"  note: {len(hard_overlap)} hard_test content group(s) also appear "
              f"byte-identically in {core_names} "
              f"(e.g. {example[0][:12]}: {names}); "
              f"hard_test is a robustness set, not an independent test set")
    else:
        print("  note: no byte-identical hard_test/normal content")

    hard_rows = grouped.get("hard_test", [])
    hard_dup_rows = sum(
        1 for row in hard_rows
        if split_of[row["sha256"]] - {"hard_test"}
    )
    return {
        "unique_sha256": len(rows_of),
        "cross_split_duplicate_groups": len(core_cross),
        "within_split_duplicate_groups": within_duplicates,
        "hard_test_cross_split_groups": len(hard_overlap),
        "hard_test_cross_split_rows": hard_dup_rows,
    }


def verify_csv(path: Path, expected_rows: int, expected_splits: set[str],
               registry_by_key: dict[str, dict], leaf_dir: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            fail(f"{as_posix(path)}: header {reader.fieldnames} != {COLUMNS}")
        rows = list(reader)
    if len(rows) != expected_rows:
        fail(f"{as_posix(path)}: expected {expected_rows} rows, got {len(rows)}")
    seen_ids: set[str] = set()
    for row in rows:
        if row["split"] not in expected_splits:
            fail(f"{as_posix(path)}: unexpected split {row['split']!r}")
        key = row["key"]
        if key not in registry_by_key:
            fail(f"{as_posix(path)}: key {key!r} not in registry")
        entry = registry_by_key[key]
        if int(row["py_label"]) != int(entry["py_label"]):
            fail(f"{as_posix(path)}: py_label mismatch for {key}: "
                 f"{row['py_label']} != {entry['py_label']}")
        if int(row["flavia_label"]) != int(entry["flavia_label"]):
            fail(f"{as_posix(path)}: flavia_label mismatch for {key}")
        if "\\" in row["filename"]:
            fail(f"{as_posix(path)}: non-posix filename {row['filename']!r}")
        expected_image_id = row["filename"].rsplit(".", 1)[0]
        if row["image_id"] != expected_image_id:
            fail(f"{as_posix(path)}: image_id {row['image_id']!r} != "
                 f"{expected_image_id!r}")
        if not (leaf_dir / row["filename"]).is_file():
            fail(f"{as_posix(path)}: missing image {row['filename']!r}")
        if row["image_id"] in seen_ids:
            fail(f"{as_posix(path)}: duplicated image_id {row['image_id']!r}")
        seen_ids.add(row["image_id"])
    print(f"  re-read {path.name}: {len(rows)} rows, header/py_label OK")


def dhash_near_duplicates(rows: list[dict], threshold: int = 2) -> tuple[int, list[str]]:
    """Informational cross-split dHash check; returns (pairs, examples)."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is present in the project env
        print("  dHash: skipped (Pillow not available)")
        return 0, []

    hashes: dict[str, tuple[int, str]] = {}
    for row in rows:
        try:
            with Image.open(row["path"]) as image:
                gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                pixels = gray.tobytes()
        except Exception as exc:  # noqa: BLE001 - report, never fail on dHash
            print(f"  dHash: skipped unreadable {row['filename']}: {exc}")
            continue
        value = 0
        for y in range(8):
            offset = y * 9
            for x in range(8):
                value = (value << 1) | int(pixels[offset + x] > pixels[offset + x + 1])
        hashes[row["filename"]] = (value, row["split"])

    pairs = 0
    examples: list[str] = []
    items = sorted(hashes.items())
    for index, (name_a, (hash_a, split_a)) in enumerate(items):
        for name_b, (hash_b, split_b) in items[index + 1:]:
            if split_a == split_b:
                continue
            if (hash_a ^ hash_b).bit_count() <= threshold:
                pairs += 1
                if len(examples) < 5:
                    examples.append(f"{name_a}~{name_b}")
    print(f"  dHash cross-split pairs with Hamming <= {threshold}: {pairs}")
    if examples:
        print("  dHash examples: " + "; ".join(examples))
    return pairs, examples


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build live-scene manifests and check video-level leakage.")
    parser.add_argument("--leaf-dir", required=True,
                        help="live frame root, e.g. data/leaf")
    parser.add_argument("--mode", required=True, choices=("all", "video-split"),
                        help="all: full normal test + all hard frames; "
                             "video-split: leaf01-03/05/04 normal + all hard "
                             "frames in hard_test")
    parser.add_argument("--registry", required=True,
                        help="class_registry.json (5 classes) or _all32.json")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args(argv)

    leaf_dir = Path(args.leaf_dir)
    registry_path = Path(args.registry)
    out_dir = Path(args.out)

    print(f"mode={args.mode} leaf_dir={as_posix(leaf_dir)} "
          f"registry={as_posix(registry_path)} out={as_posix(out_dir)}")
    registry = load_registry(registry_path)
    print(f"registry: {len(registry)} classes loaded")

    records = collect_frames(leaf_dir, registry)
    normal_count = sum(1 for record in records if record["kind"] == "normal")
    hard_count = sum(1 for record in records if record["kind"] == "hard")
    print(f"discovered frames: normal={normal_count} hard={hard_count}")
    if normal_count != sum(EXPECTED_ALL_NORMAL.values()):
        fail(f"normal frame total {normal_count} != 432")
    if hard_count != sum(EXPECTED_ALL_HARD.values()):
        fail(f"hard frame total {hard_count} != 227")

    grouped = assign_splits(records, args.mode)
    attach_hashes(records)

    # ---- count validation ------------------------------------------------ #
    print("count validation:")
    if args.mode == "all":
        check_counts("test", grouped.get("test", []), EXPECTED_ALL_NORMAL)
        if len(grouped.get("test", [])) != 432:
            fail(f"test expected 432 rows, got {len(grouped.get('test', []))}")
    else:
        for split in NORMAL_SPLITS:
            check_counts(split, grouped.get(split, []), EXPECTED_VIDEO_COUNTS[split])
            if len(grouped.get(split, [])) != VIDEO_SPLIT_TOTALS[split]:
                fail(f"{split}: expected {VIDEO_SPLIT_TOTALS[split]} rows, "
                     f"got {len(grouped.get(split, []))}")
        check_video_isolation(grouped)
    check_counts("hard_test", grouped.get("hard_test", []), EXPECTED_ALL_HARD)
    if len(grouped.get("hard_test", [])) != VIDEO_SPLIT_TOTALS["hard_test"]:
        fail(f"hard_test expected {VIDEO_SPLIT_TOTALS['hard_test']} rows, "
             f"got {len(grouped.get('hard_test', []))}")
    check_hard_isolation(grouped)

    # ---- SHA-256 leakage ------------------------------------------------- #
    print("leakage validation:")
    check_sha_leakage(grouped)
    if args.mode == "video-split":
        # dHash is only meaningful for the ordinary train/val/test frames;
        # hard_test is a robustness set and is intentionally excluded.
        print("  dHash: normal train/val/test only (hard_test excluded)")
        all_rows = [row for split in NORMAL_SPLITS
                    for row in grouped.get(split, [])]
        dhash_near_duplicates(all_rows)

    # ---- write + re-read ------------------------------------------------- #
    print("writing manifests:")
    for split in OUTPUT_SPLITS[args.mode]:
        path = out_dir / f"{split}.csv"
        rows = grouped.get(split, [])
        if not rows:
            fail(f"{split}: no rows to write")
        write_csv(path, rows)
        print(f"  wrote {as_posix(path)} ({len(rows)} rows)")

    print("re-read validation:")
    if args.mode == "all":
        expected_totals = {
            "test": sum(EXPECTED_ALL_NORMAL.values()),
            "hard_test": sum(EXPECTED_ALL_HARD.values()),
        }
    else:
        expected_totals = {split: VIDEO_SPLIT_TOTALS[split]
                           for split in OUTPUT_SPLITS[args.mode]}
    for split in OUTPUT_SPLITS[args.mode]:
        path = out_dir / f"{split}.csv"
        verify_csv(path, expected_totals[split], {split}, registry, leaf_dir)

    print("PASS: all live-scene manifest checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
