# -*- coding: utf-8 -*-
"""
从原始视频抽帧，构建图像分类数据集。

目录约定（类别名建议用拼音/英文，避免 Windows 下中文路径的兼容问题）：
    data_raw/
        ginkgo/          # 银杏
            leaf01.mp4
            leaf02.mp4
            leaf03.mp4
        camphor/         # 香樟
            leaf01.mp4
            ...

输出：
    dataset/
        train/<类别>/*.jpg
        val/<类别>/*.jpg
        test/<类别>/*.jpg
        preview/<类别>_preview.jpg   # 每类 8 张拼图，可直接放进报告

处理规则：
    1. 按 SAMPLE_FPS 抽帧，而不是使用全部帧；
    2. Laplacian 方差低于 BLUR_THRESHOLD 的模糊帧丢弃；
    3. 与上一张保留帧 dHash 汉明距离 <= DUP_DISTANCE 的重复帧丢弃；
    4. 以“视频”为单位划分 train/val/test：同一段视频的相邻帧不会跨集合，
       避免数据泄漏。每类 >=3 个视频时：最后 1 个 test，倒数第 2 个 val，
       其余 train。

用法（在项目根目录执行）：
    C:\\Anaconda\\envs\\torch_env\\python.exe tools/extract_frames.py
"""

import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# ------------------------- 可调参数 -------------------------
RAW_DIR = Path("data_raw")
OUT_DIR = Path("dataset")

SAMPLE_FPS = 2.0        # 每秒保留几帧；叶子旋转快可调到 3~4
SAVE_SIZE = 256         # 输出图片边长（正方形），训练时再缩到 96/128
BLUR_THRESHOLD = 50.0   # Laplacian 方差低于此值判为模糊，建议在 30~80 之间调
DUP_DISTANCE = 4        # 相邻帧 dHash 汉明距离小于等于此值判为重复
JPEG_QUALITY = 95
CLEAN_OUTPUT = True     # 运行前清空 dataset/train|val|test|preview
PREVIEW_PER_CLASS = 8   # 每类预览拼图张数
# -----------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
SPLITS = ("train", "val", "test")

random.seed(42)


def dhash(gray: np.ndarray, size: int = 8) -> np.ndarray:
    """差值哈希：把图像缩成 (size+1) x size，比较相邻像素得到 64 bit。"""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits((small[:, 1:] > small[:, :-1]).flatten())


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unpackbits(a ^ b).sum())


def center_square_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """中心裁成正方形再缩放，避免 16:9 画面被拉伸变形。"""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    square = frame[y0:y0 + side, x0:x0 + side]
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def open_capture(path: Path):
    """打开视频；某些 Windows 中文路径 cv2 打不开时，复制到临时 ASCII 路径重试。"""
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, None

    tmp = Path(tempfile.gettempdir()) / f"dl_frames_tmp{path.suffix.lower()}"
    try:
        shutil.copyfile(path, tmp)
        cap = cv2.VideoCapture(str(tmp))
        if cap.isOpened():
            return cap, tmp
    except OSError:
        pass
    return cap, None


def extract_video(video: Path, out_dir: Path) -> dict:
    """抽取单个视频的帧，返回 kept/blur/dup 统计。"""
    cap, tmp = open_capture(video)
    stats = {"kept": 0, "blur": 0, "dup": 0}
    if not cap.isOpened():
        print(f"  [warn] 无法打开视频: {video}")
        return stats

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    last_hash = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD:
                stats["blur"] += 1
            else:
                h = dhash(gray)
                if last_hash is not None and hamming(h, last_hash) <= DUP_DISTANCE:
                    stats["dup"] += 1
                else:
                    img = center_square_resize(frame, SAVE_SIZE)
                    ok2, buf = cv2.imencode(
                        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )
                    if ok2:
                        name = f"{video.stem}_{idx:06d}.jpg"
                        (out_dir / name).write_bytes(buf.tobytes())
                        stats["kept"] += 1
                        last_hash = h
        idx += 1

    cap.release()
    if tmp is not None:
        try:
            tmp.unlink()
        except OSError:
            pass
    return stats


def imread_unicode(path: Path):
    """cv2.imread 在 Windows 下不认中文路径，用 imdecode 绕开。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def make_preview(images: list, out_path: Path, n: int) -> None:
    """从该类的图片里随机抽 n 张拼成一张图，方便检查数据、填报告。"""
    if not images:
        return
    picks = random.sample(images, min(n, len(images)))
    cols = min(4, len(picks))
    rows = math.ceil(len(picks) / cols)
    cell = 128
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


def split_videos(videos: list) -> dict:
    """按视频划分，保证同一段视频的帧只出现在一个集合里。"""
    videos = sorted(videos)
    n = len(videos)
    if n >= 3:
        return {"train": videos[:-2], "val": [videos[-2]], "test": [videos[-1]]}
    if n == 2:
        return {"train": [videos[0]], "val": [], "test": [videos[1]]}
    if n == 1:
        return {"train": videos, "val": [], "test": []}
    return {"train": [], "val": [], "test": []}


def main() -> None:
    if not RAW_DIR.is_dir():
        print(f"找不到 {RAW_DIR.resolve()}")
        print(r"请按 data_raw/<类别>/<视频文件> 的结构放好视频后重试。")
        return

    classes = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    if not classes:
        print(f"{RAW_DIR.resolve()} 下没有类别文件夹。")
        return

    if CLEAN_OUTPUT:
        for split in SPLITS:
            shutil.rmtree(OUT_DIR / split, ignore_errors=True)
        shutil.rmtree(OUT_DIR / "preview", ignore_errors=True)
    for split in SPLITS:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    total = {s: 0 for s in SPLITS}
    header = f"{'类别':<16}{'视频':>6}{'train':>8}{'val':>6}{'test':>6}{'模糊':>8}{'重复':>8}"
    print(header)
    print("-" * len(header.encode("gbk", errors="ignore")))

    for class_dir in classes:
        cls = class_dir.name
        videos = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS
        )
        if not videos:
            print(f"  [warn] {cls} 下没有视频文件，已跳过")
            continue

        split_map = split_videos(videos)
        counts = {}
        blur = dup = 0

        for split in SPLITS:
            out = OUT_DIR / split / cls
            out.mkdir(parents=True, exist_ok=True)
            kept = 0
            for video in split_map[split]:
                st = extract_video(video, out)
                kept += st["kept"]
                blur += st["blur"]
                dup += st["dup"]
            counts[split] = kept
            total[split] += kept

        if not split_map["val"]:
            print(f"  [warn] {cls} 只有 {len(videos)} 个视频，val 为空，建议每类至少录 3 个视频")

        print(
            f"{cls:<16}{len(videos):>6}{counts['train']:>8}"
            f"{counts['val']:>6}{counts['test']:>6}{blur:>8}{dup:>8}"
        )

        all_imgs = []
        for split in SPLITS:
            all_imgs += sorted((OUT_DIR / split / cls).glob("*.jpg"))
        make_preview(all_imgs, OUT_DIR / "preview" / f"{cls}_preview.jpg", PREVIEW_PER_CLASS)

    print("-" * 60)
    print(f"总计: train={total['train']}  val={total['val']}  test={total['test']}")
    print(f"输出目录: {OUT_DIR.resolve()}")
    print(r"下一步: 用 torchvision.ImageFolder 读取 dataset/train 和 dataset/val 训练。")


if __name__ == "__main__":
    main()
