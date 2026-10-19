"""数据集与 DataLoader。

训练、验证、测试、网页推理共用同一套预处理定义：
- 训练：``build_train_transform()``（plans.md §3.2）
- 验证 / 测试 / 网页：``build_eval_transform()``，三者必须完全一致
"""

from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REQUIRED_COLUMNS = ("filename", "py_label")
INPUT_SIZE = 128
EVAL_RESIZE = 146


def build_train_transform() -> transforms.Compose:
    """训练增强流水线，严格对应 plans.md §3.2。"""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                INPUT_SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.25)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.1),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """验证 / 测试 / 网页推理共用的确定性预处理。"""
    return transforms.Compose(
        [
            transforms.Resize(EVAL_RESIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class LeafDataset(Dataset):
    """按 CSV 清单读取叶片图片。

    CSV 至少包含 ``filename`` 与 ``py_label`` 两列。图片优先从
    ``cache_dir/filename`` 读取（存在时），否则从 ``image_root/filename`` 读取。
    ``__getitem__`` 返回 ``(image_tensor, int(py_label), filename)``。
    """

    def __init__(
        self,
        csv_path: Path | str,
        image_root: Path | str,
        cache_dir: Path | str | None = None,
        transform: Any | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV {self.csv_path} is missing required column(s): {missing}. "
                f"Required columns: {list(REQUIRED_COLUMNS)} "
                f"(e.g. filename,py_label)."
            )
        if df.empty:
            raise ValueError(f"CSV is empty: {self.csv_path}")

        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # 未显式传入 transform 时使用推理预处理，保证返回值始终是 tensor。
        self.transform = transform if transform is not None else build_eval_transform()

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, filename: str) -> Path:
        if self.cache_dir is not None:
            cached = self.cache_dir / filename
            if cached.is_file():
                return cached
        return self.image_root / filename

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[index]
        filename = str(row["filename"])
        path = self._resolve_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            image = self.transform(img)

        label = int(row["py_label"])
        return image, label, filename


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """为每个 DataLoader worker 固定 random / numpy / torch 的种子。"""
    worker_seed = (int(base_seed) + int(worker_id)) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_dataloaders(
    csv_dir: Path | str,
    image_root: Path | str,
    cache_dir: Path | str | None = None,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """构建 train / val / test DataLoader；不存在的 split 直接跳过。

    CPU 训练：``pin_memory=False``；train 打乱，val/test 保持顺序；
    所有 split 均 ``drop_last=False``。
    """
    csv_dir = Path(csv_dir)
    worker_init_fn = partial(_seed_worker, base_seed=int(seed))
    loaders: dict[str, DataLoader] = {}

    split_settings = (
        ("train", batch_size, True, build_train_transform()),
        ("val", eval_batch_size, False, build_eval_transform()),
        ("test", eval_batch_size, False, build_eval_transform()),
    )

    for split, bs, shuffle, transform in split_settings:
        csv_path = csv_dir / f"{split}.csv"
        if not csv_path.is_file():
            continue

        dataset = LeafDataset(
            csv_path=csv_path,
            image_root=image_root,
            cache_dir=cache_dir,
            transform=transform,
        )
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        loaders[split] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=int(num_workers),
            pin_memory=False,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=generator if shuffle else None,
        )

    return loaders
