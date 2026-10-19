# -*- coding: utf-8 -*-
"""
从 data/<类别>/*.mp4 抽帧，输出到 data/<类别>_test/。

与 tools/extract_frames.py 的区别：
- 不做 train/val/test 划分，所有视频的帧都进入 <类别>_test；
- 默认输出 data/<类别>_test/<视频名>_<帧号>.jpg；
- 额外生成 data/test_preview/<类别>_preview.jpg 和 extract_stats.json，方便检查。

用法（项目根目录）：
    C:\\Anaconda\\envs\\torch_env\\python.exe tools\\extract_test_frames.py
    ... --only leaf05            # 只抽第 5 段视频
    ... --sample-fps 3 --save-size 256
"""

import argparse
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
SKIP_DIRS = {"Leaves", "test_preview"}


def dhash(gray: np.ndarray, size: int = 8) -> np.ndarray:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((small[:, 1:] > small[:, :-1]).flatten())


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unpackbits(a ^ b).sum())


def center_square_resize(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    square = frame[y0:y0 + side, x0:x0 + side]
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def open_capture(path: Path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, None
    tmp = Path(tempfile.gettempdir()) / f"dl_test_frames_tmp{path.suffix.lower()}"
    try:
        shutil.copyfile(path, tmp)
        cap = cv2.VideoCapture(str(tmp))
        if cap.isOpened():
            return cap, tmp
    except OSError:
        pass
    return cap, None


def extract_video(
    video: Path,
    out_dir: Path,
    sample_fps: float,
    save_size: int,
    blur_threshold: float,
    dup_distance: int,
    jpeg_quality: int,
    blur_mode: str = "adaptive",
) -> dict:
    cap, tmp = open_capture(video)
    stats = {"video": video.name, "kept": 0, "blur": 0, "dup": 0, "open": cap.isOpened()}
    if not cap.isOpened():
        print(f"  [warn] 无法打开视频: {video}")
        return stats

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    candidates = []  # (frame_idx, laplacian_var, dhash, jpeg_bytes)
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            h = dhash(gray)
            img = center_square_resize(frame, save_size)
            ok2, buf = cv2.imencode(
                ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            candidates.append((idx, var, h, buf.tobytes() if ok2 else None))
        idx += 1

    cap.release()
    if tmp is not None:
        try:
            tmp.unlink()
        except OSError:
            pass

    if blur_mode == "adaptive" and candidates:
        # 白墙/浅色背景会让 Laplacian 方差整体偏低，绝对阈值会误杀清晰帧。
        # 这里保留每个视频中较清晰的约 80%，再用上限避免把明显模糊的帧放进来。
        p20 = float(np.percentile([c[1] for c in candidates], 20))
        threshold = min(max(3.0, p20), blur_threshold)
    else:
        threshold = blur_threshold
    stats["threshold"] = round(threshold, 2)

    last_hash = None
    for frame_idx, var, h, buf in candidates:
        if var < threshold:
            stats["blur"] += 1
            continue
        if last_hash is not None and hamming(h, last_hash) <= dup_distance:
            stats["dup"] += 1
            continue
        if buf is None:
            continue
        (out_dir / f"{video.stem}_{frame_idx:06d}.jpg").write_bytes(buf)
        stats["kept"] += 1
        last_hash = h

    return stats


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def make_preview(images: list, out_path: Path, n: int = 8) -> None:
    if not images:
        return
    picks = random.sample(images, min(n, len(images)))
    cols = min(4, len(picks))
    rows = math.ceil(len(picks) / cols)
    cell = 160
    canvas = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)
    for k, p in enumerate(picks):
        img = imread_unicode(p)
        if img is None:
            continue
        r, c = divmod(k, cols)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.resize(
            img, (cell, cell), interpolation=cv2.INTER_AREA
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        out_path.write_bytes(buf.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="从 data/<类别>/*.mp4 抽帧到 data/<类别>_test/")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--suffix", default="_test", help="输出目录后缀，默认 _test")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--save-size", type=int, default=256)
    parser.add_argument("--blur-threshold", type=float, default=50.0,
                        help="absolute 模式下为固定阈值；adaptive 模式下为上限")
    parser.add_argument("--blur-mode", choices=["adaptive", "absolute"], default="adaptive",
                        help="adaptive: 每个视频保留最清晰的约 80%%；absolute: 固定阈值")
    parser.add_argument("--dup-distance", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--only", default="all", help="all 或逗号分隔的视频名，如 leaf05")
    parser.add_argument("--clean", action="store_true", help="先清空输出目录")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    only = None if args.only.strip().lower() == "all" else {
        s.strip() for s in args.only.split(",") if s.strip()
    }

    class_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.endswith(args.suffix)
    )
    if not class_dirs:
        print(f"{data_dir.resolve()} 下没有类别文件夹")
        return

    summary = {}
    header = f"{'类别':<16}{'视频':>6}{'保留':>8}{'模糊':>8}{'重复':>8}"
    print(header)
    print("-" * 56)

    for class_dir in class_dirs:
        videos = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if only is not None:
            videos = [v for v in videos if v.stem in only]
        if not videos:
            continue

        out_dir = data_dir / f"{class_dir.name}{args.suffix}"
        if args.clean and out_dir.is_dir():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        kept = blur = dup = 0
        per_video = []
        for video in videos:
            st = extract_video(
                video, out_dir, args.sample_fps, args.save_size,
                args.blur_threshold, args.dup_distance, args.jpeg_quality,
                args.blur_mode,
            )
            per_video.append(st)
            kept += st["kept"]
            blur += st["blur"]
            dup += st["dup"]

        images = sorted(out_dir.glob("*.jpg"))
        make_preview(images, data_dir / "test_preview" / f"{class_dir.name}_preview.jpg")
        summary[class_dir.name] = {
            "output": str(out_dir),
            "videos": per_video,
            "kept": kept,
            "blur": blur,
            "dup": dup,
        }
        print(f"{class_dir.name:<16}{len(videos):>6}{kept:>8}{blur:>8}{dup:>8}")

    (data_dir / "test_preview").mkdir(parents=True, exist_ok=True)
    stats_path = data_dir / "test_preview" / "extract_stats.json"
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(v["kept"] for v in summary.values())
    print("-" * 56)
    print(f"总保留帧数: {total}")
    print(f"输出目录: {[v['output'] for v in summary.values()]}")
    print(f"统计与预览: {(data_dir / 'test_preview').resolve()}")


if __name__ == "__main__":
    random.seed(42)
    main()
