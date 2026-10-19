#!/usr/bin/env python3
"""从 plans.md §13 解析 Flavia 32 类映射，生成 data/class_map.json 与 data/zh_names.json。

用法:
    python tools/make_class_map.py

数据来源（唯一权威）:
    plans.md §13 的 32 类表（Flavia 编号 1-12、14-33）。

交叉核对:
    data/Flavia类别对照表.md 的「二、32 类全表」。

校验失败时返回非零退出码，且不覆盖已有输出。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANS_PATH = ROOT / "plans.md"
CHECK_PATH = ROOT / "data" / "Flavia类别对照表.md"
OUT_MAP_PATH = ROOT / "data" / "class_map.json"
OUT_ZH_PATH = ROOT / "data" / "zh_names.json"

SOURCE = "plans.md §13"
EXPECTED_LABELS = set(range(1, 13)) | set(range(14, 34))  # 1-12, 14-33
EXPECTED_COUNT = 32
EXPECTED_IMAGE_TOTAL = 1798

_failures: list[str] = []


def check(ok: bool, message: str) -> None:
    """记录并打印一条校验结果。"""
    if ok:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        _failures.append(message)


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"缺少文件: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def extract_table(lines: list[str], heading_pattern: str, source: str) -> list[list[str]]:
    """在指定小节内提取第一张 Markdown 表格，返回去掉表头与分隔行后的数据行。"""
    start = None
    for i, line in enumerate(lines):
        if re.match(heading_pattern, line):
            start = i
            break
    if start is None:
        raise SystemExit(f"{source}: 未找到小节标题 {heading_pattern!r}")

    table_lines: list[str] = []
    for line in lines[start + 1:]:
        if re.match(r"^##\s", line):  # 下一个二级标题即小节结束
            break
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)

    if len(table_lines) < 3:
        raise SystemExit(f"{source}: 表格行数不足（需表头 + 分隔行 + 数据行）")

    rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
    header, separator, data = rows[0], rows[1], rows[2:]

    if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        raise SystemExit(f"{source}: 表格分隔行格式异常: {table_lines[1]}")

    return [header] + data


def parse_range(text: str, source: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", text)
    if not match:
        raise SystemExit(f"{source}: 无法解析区间 {text!r}")
    return int(match.group(1)), int(match.group(2))


def parse_int(text: str, source: str) -> int:
    if not re.fullmatch(r"\d+", text):
        raise SystemExit(f"{source}: 期望整数，得到 {text!r}")
    return int(text)


def parse_plans_table(lines: list[str]) -> tuple[list[dict], dict[int, int]]:
    """解析 plans.md §13，返回类别列表（按编号升序）与编号->张数。"""
    rows = extract_table(lines, r"^##\s*13\.", str(PLANS_PATH))
    header, data = rows[0], rows[1:]
    expected_header = ["Flavia 编号", "key", "中文名", "学名", "英文名", "区间", "张数", "备注"]
    if header != expected_header:
        raise SystemExit(f"{PLANS_PATH}: §13 表头不符，实际为 {header}")

    classes: list[dict] = []
    counts: dict[int, int] = {}
    for row in data:
        if len(row) != len(expected_header):
            raise SystemExit(f"{PLANS_PATH}: §13 行列数异常: {row}")
        label = parse_int(row[0], f"{PLANS_PATH} §13")
        start, end = parse_range(row[5], f"{PLANS_PATH} §13 第 {label} 类")
        classes.append(
            {
                "flavia_label": label,
                "key": row[1],
                "zh_name": row[2],
                "scientific": row[3],
                "common": row[4],
                "range": [start, end],
            }
        )
        counts[label] = parse_int(row[6], f"{PLANS_PATH} §13 第 {label} 类张数")

    classes.sort(key=lambda item: item["flavia_label"])
    return classes, counts


def parse_check_table(lines: list[str]) -> dict[int, dict]:
    """解析对照表「二、32 类全表」，返回编号->字段。"""
    rows = extract_table(lines, r"^##\s*二、32 类全表", str(CHECK_PATH))
    header, data = rows[0], rows[1:]
    expected_header = [
        "Flavia 编号", "key", "中文名", "学名（原表）", "英文名",
        "编号区间", "实际张数", "备注",
    ]
    if header != expected_header:
        raise SystemExit(f"{CHECK_PATH}: 表头不符，实际为 {header}")

    result: dict[int, dict] = {}
    for row in data:
        if len(row) != len(expected_header):
            raise SystemExit(f"{CHECK_PATH}: 行列数异常: {row}")
        label = parse_int(row[0], str(CHECK_PATH))
        start, end = parse_range(row[5], f"{CHECK_PATH} 第 {label} 类")
        result[label] = {
            "key": row[1],
            "zh_name": row[2],
            "scientific": row[3],
            "common": row[4],
            "range": [start, end],
            "count": parse_int(row[6], f"{CHECK_PATH} 第 {label} 类张数"),
        }
    return result


def validate_classes(classes: list[dict], counts: dict[int, int],
                     reference: dict[int, dict]) -> None:
    labels = [item["flavia_label"] for item in classes]
    keys = [item["key"] for item in classes]

    check(len(classes) == EXPECTED_COUNT, f"类别数 = {EXPECTED_COUNT}（实际 {len(classes)}）")
    check(len(labels) == len(set(labels)), "flavia_label 唯一")
    check(len(keys) == len(set(keys)), "key 唯一")
    check(set(labels) == EXPECTED_LABELS,
          "编号集合恰好为 1-12 与 14-33（共 32 个）")
    check(labels == sorted(labels), "类别按 flavia_label 升序输出")

    ranges_ok = all(item["range"][0] <= item["range"][1] for item in classes)
    check(ranges_ok, "每个 range 起点 <= 终点")

    ordered = sorted(classes, key=lambda item: item["range"][0])
    overlaps = [
        f"{prev['flavia_label']}:{prev['range']} 与 {cur['flavia_label']}:{cur['range']}"
        for prev, cur in zip(ordered, ordered[1:])
        if cur["range"][0] <= prev["range"][1]
    ]
    check(not overlaps,
          "32 个编号区间互不重叠"
          + (f"（重叠: {overlaps}）" if overlaps else ""))

    # 与 data/Flavia类别对照表.md 逐类比对
    diff_key: list[int] = []
    diff_zh: list[int] = []
    diff_range: list[int] = []
    diff_count: list[int] = []
    for item in classes:
        label = item["flavia_label"]
        ref = reference.get(label)
        if ref is None:
            diff_key.append(label)
            continue
        if item["key"] != ref["key"]:
            diff_key.append(label)
        if item["zh_name"] != ref["zh_name"]:
            diff_zh.append(label)
        if item["range"] != ref["range"]:
            diff_range.append(label)
        if counts[label] != ref["count"]:
            diff_count.append(label)

    check(set(reference) == EXPECTED_LABELS,
          f"对照表编号集合与 §13 一致（对照表 {len(reference)} 类）")
    check(not diff_key, f"对照表 key 差异 = 0（差异编号: {diff_key}）")
    check(not diff_zh, f"对照表中文名差异 = 0（差异编号: {diff_zh}）")
    check(not diff_range, f"对照表区间差异 = 0（差异编号: {diff_range}）")
    check(not diff_count, f"对照表张数差异 = 0（差异编号: {diff_count}）")

    total_images = sum(counts.values())
    check(total_images == EXPECTED_IMAGE_TOTAL,
          f"张数合计 = {EXPECTED_IMAGE_TOTAL}（实际 {total_images}）")


def write_outputs(classes: list[dict]) -> None:
    class_map = {"source": SOURCE, "classes": classes}
    zh_names = {str(item["flavia_label"]): item["zh_name"] for item in classes}

    OUT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAP_PATH.write_text(
        json.dumps(class_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUT_ZH_PATH.write_text(
        json.dumps(zh_names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[INFO] 已写入 {OUT_MAP_PATH} 与 {OUT_ZH_PATH}")


def validate_outputs(classes: list[dict]) -> None:
    """重新读入产物并检查结构。"""
    try:
        class_map = json.loads(OUT_MAP_PATH.read_text(encoding="utf-8"))
        zh_names = json.loads(OUT_ZH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        check(False, f"产物可重新读入为 JSON（{exc}）")
        return

    check(isinstance(class_map, dict) and class_map.get("source") == SOURCE,
          f"class_map.json 的 source == {SOURCE!r}")
    loaded = class_map.get("classes") if isinstance(class_map, dict) else None
    check(isinstance(loaded, list) and len(loaded) == EXPECTED_COUNT,
          "class_map.json 含 32 个 class 条目")
    if not isinstance(loaded, list):
        return

    required = {"flavia_label", "key", "zh_name", "scientific", "common", "range"}
    structure_ok = all(
        isinstance(item, dict)
        and set(item) == required
        and isinstance(item["flavia_label"], int)
        and isinstance(item["key"], str) and item["key"]
        and isinstance(item["zh_name"], str) and item["zh_name"]
        and isinstance(item["scientific"], str) and item["scientific"]
        and isinstance(item["common"], str) and item["common"]
        and isinstance(item["range"], list) and len(item["range"]) == 2
        and all(isinstance(value, int) and not isinstance(value, bool)
                for value in item["range"])
        for item in loaded
    )
    check(structure_ok, "class_map.json 每条结构正确（字段/类型/range 两整数）")
    check([item["flavia_label"] for item in loaded]
          == [item["flavia_label"] for item in classes],
          "class_map.json 条目与解析结果一致（编号升序）")

    expected_zh = {str(item["flavia_label"]): item["zh_name"] for item in classes}
    zh_ok = (
        isinstance(zh_names, dict)
        and len(zh_names) == EXPECTED_COUNT
        and all(isinstance(key, str) for key in zh_names)
        and zh_names == expected_zh
    )
    check(zh_ok, "zh_names.json 键为字符串、32 项且与 class_map 一致")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    plans_lines = read_lines(PLANS_PATH)
    check_lines = read_lines(CHECK_PATH)

    classes, counts = parse_plans_table(plans_lines)
    reference = parse_check_table(check_lines)

    print(f"[INFO] 解析 {PLANS_PATH} §13: {len(classes)} 类；"
          f"交叉核对 {CHECK_PATH}: {len(reference)} 类")
    print(f"[INFO] 类别: {[item['flavia_label'] for item in classes]}")
    print(f"[INFO] 张数合计: {sum(counts.values())}")

    validate_classes(classes, counts, reference)

    if _failures:
        print(f"\n[ABORT] 共 {len(_failures)} 项校验失败，未写入输出文件。")
        return 1

    write_outputs(classes)
    validate_outputs(classes)

    if _failures:
        print(f"\n[ABORT] 产物校验失败 {len(_failures)} 项。")
        return 1

    print(f"\n[OK] 全部校验通过：32 类、区间互不重叠、与对照表 0 差异、"
          f"张数合计 {EXPECTED_IMAGE_TOTAL}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
