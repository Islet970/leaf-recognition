#!/usr/bin/env python3
"""把选定图片预处理为 320px 长边缓存，供 CPU 训练加速。

处理流程:
    1. 读取并应用 EXIF 方向（ImageOps.exif_transpose）；
    2. 转为 RGB；
    3. 按长边等比缩放到 --long-edge（默认 320）；
    4. 以 JPEG q=95（默认）写出，不保留 EXIF。

用法:
    python tools/prepare_cache.py --src data/Leaves --dst data/cache/leaf_320 \
        --long-edge 320 --manifest data/splits/5class/all.csv --quality 95

    --manifest 可选：CSV 至少含 filename 列，只处理清单中列出的文件；
    缺省时处理 --src 下所有 .jpg。
    --overwrite 可选：默认跳过已存在的缓存文件。

输出文件名沿用原文件名。失败文件收集后汇总打印，并以非零退出码结束。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

DEFAULT_LONG_EDGE = 320
DEFAULT_QUALITY = 95


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成缩放到指定长边的 JPEG 缓存（EXIF 校正、RGB、q=95）。"
    )
    parser.add_argument("--src", type=Path, required=True, help="原始图片目录")
    parser.add_argument("--dst", type=Path, required=True, help="缓存输出目录")
    parser.add_argument(
        "--long-edge",
        type=int,
        default=DEFAULT_LONG_EDGE,
        help=f"缩放后的长边像素（默认 {DEFAULT_LONG_EDGE}）",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="可选 CSV 清单，至少含 filename 列；只处理清单中的文件",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_QUALITY,
        help=f"JPEG 质量（默认 {DEFAULT_QUALITY}）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的缓存文件（默认跳过）",
    )
    args = parser.parse_args(argv)
    if args.long_edge <= 0:
        parser.error("--long-edge 必须为正整数")
    if not 1 <= args.quality <= 100:
        parser.error("--quality 必须在 1-100 之间")
    return args


def load_manifest(manifest: Path) -> list[str]:
    """读取清单的 filename 列，按出现顺序去重。"""
    if not manifest.is_file():
        raise SystemExit(f"[错误] 清单文件不存在: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise SystemExit(f"[错误] 清单缺少 filename 列: {manifest}")
        names: list[str] = []
        seen: set[str] = set()
        for row in reader:
            name = (row.get("filename") or "").strip()
            if not name:
                raise SystemExit(f"[错误] 清单第 {reader.line_num} 行 filename 为空")
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def collect_source_images(src: Path) -> list[str]:
    """缺省模式下收集 src 下所有 .jpg 的相对路径。"""
    if not src.is_dir():
        raise SystemExit(f"[错误] 源目录不存在: {src}")
    files = sorted(
        p for p in src.rglob("*") if p.is_file() and p.suffix.lower() == ".jpg"
    )
    return [p.relative_to(src).as_posix() for p in files]


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    """按长边等比缩放；长边缩放到恰好 long_edge，短边四舍五入。"""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"无效图像尺寸: {width}x{height}")
    scale = long_edge / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    if (new_width, new_height) != (width, height):
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


def prepare_one(
    src_path: Path, dst_path: Path, long_edge: int, quality: int
) -> None:
    """处理单张图片，先写临时文件再原子替换，避免残留半成品缓存。"""
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    try:
        with Image.open(src_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = resize_long_edge(image, long_edge)
            image.save(tmp_path, format="JPEG", quality=quality)
        os.replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src: Path = args.src
    dst: Path = args.dst

    if args.manifest is not None:
        names = load_manifest(args.manifest)
    else:
        names = collect_source_images(src)

    # 缓存目录为平铺结构，输出文件名必须唯一。
    output_names = [Path(name).name for name in names]
    duplicates = sorted(
        {name for name in output_names if output_names.count(name) > 1}
    )
    if duplicates:
        raise SystemExit(f"[错误] 输出文件名冲突: {', '.join(duplicates)}")

    dst.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    processed = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    for name in names:
        src_path = src / name
        dst_path = dst / Path(name).name
        if not args.overwrite and dst_path.exists():
            skipped += 1
            continue
        try:
            prepare_one(src_path, dst_path, args.long_edge, args.quality)
        except Exception as exc:  # 单张失败不中断，最后汇总报错
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        processed += 1

    elapsed = time.perf_counter() - started
    print(
        f"完成: 处理 {processed} 张，跳过 {skipped} 张，"
        f"失败 {len(failures)} 张，耗时 {elapsed:.2f}s"
    )
    print(f"输出目录: {dst}")

    if failures:
        print(f"[错误] 以下 {len(failures)} 个文件处理失败:", file=sys.stderr)
        for name, reason in failures:
            print(f"  - {name}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
