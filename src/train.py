"""A/B/C/D/E 五模型训练入口（plans.md §3）。

用法（项目根目录执行）：
    python -m src.train --models A B C D E \
        --data data/splits/5class --registry data/class_registry.json \
        --num-classes 5 --cache data/cache/leaf_320 --epochs 40 \
        --batch-size 32 --eval-batch-size 64 --seed 42 \
        --output runs/5class --num-workers 4

复用模块：
    src.models  build_model / MODEL_NAMES
    src.data    build_dataloaders
    src.utils   set_seed / get_logger / AverageMeter /
                save_checkpoint / load_checkpoint / macro_f1 / count_parameters
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.data import INPUT_SIZE, build_dataloaders
from src.models import MODEL_IDS, MODEL_NAMES, build_model
from src.utils import (
    AverageMeter,
    count_parameters,
    get_logger,
    load_checkpoint,
    macro_f1,
    save_checkpoint,
    set_seed,
)

HISTORY_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_f1",
    "epoch_sec",
]
MIN_LR = 1e-5
GRAD_CLIP = 5.0
SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# 参数与注册表
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leaf classification models A-E.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS),
                        choices=MODEL_IDS, help="要训练的模型编号")
    parser.add_argument("--data", required=True, help="split CSV 目录（train/val/test.csv）")
    parser.add_argument("--registry", required=True, help="class_registry.json 路径")
    parser.add_argument("--num-classes", type=int, required=True, help="类别数")
    parser.add_argument("--cache", default=None, help="320px 图片缓存目录（可选）")
    parser.add_argument("--image-root", default="data/Leaves",
                        help="原始图片目录（缓存未命中时回退）")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/5class", help="输出根目录")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0,
                        help=">0 时每 epoch 最多跑这么多 batch（仅冒烟）")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_registry(path: str | Path, num_classes: int) -> tuple[list[str], list[int]]:
    """读取注册表，返回按 py_label 排序的 (class_keys, flavia_labels)。"""
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = registry.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"注册表缺少非空 classes 列表: {path}")
    if len(classes) != int(num_classes):
        raise ValueError(
            f"--num-classes={num_classes} 与注册表类别数 {len(classes)} 不一致: {path}"
        )

    ordered = sorted(classes, key=lambda c: int(c["py_label"]))
    py_labels = [int(c["py_label"]) for c in ordered]
    if py_labels != list(range(int(num_classes))):
        raise ValueError(f"注册表 py_label 必须是从 0 开始的连续整数，实际: {py_labels}")

    class_keys = [str(c["key"]) for c in ordered]
    flavia_labels = [int(c["flavia_label"]) for c in ordered]
    if len(set(class_keys)) != len(class_keys):
        raise ValueError(f"注册表 class key 重复: {class_keys}")
    if len(set(flavia_labels)) != len(flavia_labels):
        raise ValueError(f"注册表 flavia_label 重复: {flavia_labels}")
    return class_keys, flavia_labels


def validate_split_labels(
    data_dir: str | Path,
    class_keys: list[str],
    flavia_labels: list[int],
) -> dict[str, int]:
    """训练前交叉校验 split CSV 的 py_label/key/flavia_label 与注册表。

    校验内容：
    - CSV 中每个 py_label 只对应唯一的 (key, flavia_label)；
    - py_label 必须落在 [0, num_classes)；
    - (key, flavia_label) 必须与按 py_label 排序后的注册表完全一致。

    任何不一致都会抛出带文件路径与行号的 ValueError，绝不静默继续。
    返回 {split: 数据行数}，只包含实际存在的 CSV。
    """
    data_dir = Path(data_dir)
    expected = {
        py_label: (class_keys[py_label], int(flavia_labels[py_label]))
        for py_label in range(len(class_keys))
    }
    required_columns = ("py_label", "key", "flavia_label")
    checked: dict[str, int] = {}

    for split in SPLIT_NAMES:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.is_file():
            continue

        seen: dict[int, tuple[str, int]] = {}
        rows = 0
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            header = list(reader.fieldnames or [])
            missing_columns = [c for c in required_columns if c not in header]
            if missing_columns:
                raise ValueError(
                    f"{csv_path}: 缺少必需列 {missing_columns}；实际列为 {header}"
                )

            for row in reader:
                rows += 1
                line_no = reader.line_num  # 表头为第 1 行，数据行号即物理行号
                raw_label = (row.get("py_label") or "").strip()
                raw_flavia = (row.get("flavia_label") or "").strip()
                key = (row.get("key") or "").strip()
                try:
                    py_label = int(raw_label)
                except ValueError:
                    raise ValueError(
                        f"{csv_path}:{line_no}: py_label 不是整数: {raw_label!r}"
                    ) from None
                try:
                    flavia = int(raw_flavia)
                except ValueError:
                    raise ValueError(
                        f"{csv_path}:{line_no}: flavia_label 不是整数: {raw_flavia!r}"
                    ) from None

                pair = (key, flavia)
                if py_label in seen and seen[py_label] != pair:
                    prev_key, prev_flavia = seen[py_label]
                    raise ValueError(
                        f"{csv_path}:{line_no}: py_label={py_label} 对应多个类别，"
                        f"此前为 (key={prev_key!r}, flavia_label={prev_flavia})，"
                        f"本行为 (key={key!r}, flavia_label={flavia})"
                    )
                seen.setdefault(py_label, pair)

                if py_label not in expected:
                    raise ValueError(
                        f"{csv_path}:{line_no}: py_label={py_label} 超出 "
                        f"[0, {len(class_keys)})"
                    )
                expected_pair = expected[py_label]
                if pair != expected_pair:
                    raise ValueError(
                        f"{csv_path}:{line_no}: py_label={py_label} 与注册表不一致，"
                        f"CSV 为 (key={key!r}, flavia_label={flavia})，"
                        f"注册表为 (key={expected_pair[0]!r}, "
                        f"flavia_label={expected_pair[1]})"
                    )

        if rows == 0:
            raise ValueError(f"{csv_path}: CSV 为空（只有表头或没有数据行）")
        checked[split] = rows

    return checked


# --------------------------------------------------------------------------- #
# 优化器与调度器
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, args: argparse.Namespace) -> AdamW:
    """按参数名是否以 features. 开头分两组：backbone / 新模块+分类头。"""
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("features.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("features.")]
    if not backbone_params:
        raise RuntimeError("未找到 features.* 参数，无法划分 backbone 参数组")
    if not head_params:
        raise RuntimeError("未找到新模块/分类头参数，无法划分 head 参数组")

    return AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone, "name": "backbone"},
            {"params": head_params, "lr": args.lr_head, "name": "head"},
        ],
        weight_decay=args.weight_decay,
    )


def build_scheduler(
    optimizer: AdamW, total_epochs: int, warmup_epochs: int
) -> LambdaLR:
    """3 epoch 线性 warmup + cosine 衰减到 MIN_LR；每组按自己的 base_lr 算最小比例。"""
    warmup = max(0, min(int(warmup_epochs), int(total_epochs)))
    decay_span = max(1, int(total_epochs) - 1 - warmup)

    def make_lambda(base_lr: float):
        min_ratio = min(1.0, MIN_LR / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / float(decay_span)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return lr_lambda

    return LambdaLR(
        optimizer,
        lr_lambda=[make_lambda(float(g["lr"])) for g in optimizer.param_groups],
    )


def optimizer_lrs(optimizer: AdamW) -> tuple[float, float]:
    """返回 (backbone lr, head lr)，供日志与 history 使用。"""
    lrs = {g.get("name", ""): float(g["lr"]) for g in optimizer.param_groups}
    return lrs.get("backbone", 0.0), lrs.get("head", 0.0)


# --------------------------------------------------------------------------- #
# 数据加载（Windows 下 worker 启动失败自动回退 0）
# --------------------------------------------------------------------------- #
def build_loaders(args: argparse.Namespace, num_workers: int) -> dict[str, Any]:
    return build_dataloaders(
        csv_dir=args.data,
        image_root=args.image_root,
        cache_dir=args.cache,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=num_workers,
        seed=args.seed,
    )


def build_loaders_with_fallback(
    args: argparse.Namespace, logger
) -> tuple[dict[str, Any], int]:
    loaders = build_loaders(args, args.num_workers)
    if args.num_workers <= 0:
        return loaders, 0

    probe_split = "val" if "val" in loaders else "train"
    try:
        next(iter(loaders[probe_split]))
    except Exception as exc:  # noqa: BLE001 - worker 启动失败时回退
        logger.warning(
            "DataLoader num_workers=%d 启动失败（%s: %s），回退 num_workers=0。",
            args.num_workers, type(exc).__name__, exc,
        )
        loaders = build_loaders(args, 0)
        next(iter(loaders[probe_split]))
        return loaders, 0
    return loaders, args.num_workers


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def topk_k_for(num_classes: int) -> int:
    """top-k 的 k：类别数 > 5 时取 5，否则取 3，且不超过类别数。"""
    num_classes = int(num_classes)
    if num_classes < 1:
        raise ValueError(f"num_classes 必须为正整数，收到 {num_classes}")
    return min(5 if num_classes > 5 else 3, num_classes)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    topk_correct = 0
    topk_k = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    for batch_idx, (images, labels, _filenames) in enumerate(loader):
        images = images.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)
        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        batch_size = int(labels.numel())
        loss_meter.update(loss.item(), batch_size)
        correct += int((preds == labels).sum().item())
        total += batch_size

        topk_k = topk_k_for(int(logits.shape[1]))
        topk = logits.topk(topk_k, dim=1).indices
        topk_correct += int((topk == labels.unsqueeze(1)).any(dim=1).sum().item())

        all_targets.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        if max_batches > 0 and batch_idx + 1 >= max_batches:
            break

    if total == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "f1": 0.0,
            "topk_acc": 0.0,
            "topk_k": topk_k,
            "n": 0,
        }
    return {
        "loss": float(loss_meter.avg),
        "acc": correct / total,
        "f1": macro_f1(all_targets, all_preds),
        "topk_acc": topk_correct / total,
        "topk_k": topk_k,
        "n": total,
    }


# --------------------------------------------------------------------------- #
# 单个模型训练
# --------------------------------------------------------------------------- #
def train_one_model(
    model_id: str,
    args: argparse.Namespace,
    class_keys: list[str],
    flavia_labels: list[int],
    device: torch.device,
) -> dict[str, Any]:
    run_dir = Path(args.output) / f"{MODEL_NAMES[model_id]}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("src.train", run_dir / "train.log")

    logger.info("=" * 78)
    logger.info("模型 %s (%s) -> %s", model_id, MODEL_NAMES[model_id], run_dir)

    set_seed(args.seed)
    loaders, used_workers = build_loaders_with_fallback(args, logger)
    if "train" not in loaders:
        raise FileNotFoundError(f"未找到 train.csv: {Path(args.data) / 'train.csv'}")
    if "val" not in loaders:
        raise FileNotFoundError(f"未找到 val.csv: {Path(args.data) / 'val.csv'}")

    model = build_model(model_id, args.num_classes, pretrained=True).to(device)
    report = getattr(model, "pretrained_report", {})
    logger.info(
        "参数量: %s (%.3f M) | 预训练 missing=%d unexpected=%d | device=%s | workers=%d",
        format(count_parameters(model), ","),
        count_parameters(model) / 1e6,
        len(report.get("missing_keys", [])),
        len(report.get("unexpected_keys", [])),
        device,
        used_workers,
    )
    logger.info(
        "数据: train=%d val=%d test=%d | batch=%d/%d | classes=%d",
        len(loaders["train"].dataset),
        len(loaders["val"].dataset),
        len(loaders["test"].dataset) if "test" in loaders else 0,
        args.batch_size,
        args.eval_batch_size,
        args.num_classes,
    )

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    # 训练使用 label smoothing；val/test 统一用无平滑 CE，保证 loss 同量纲。
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    config = {
        **vars(args),
        "model_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "class_keys": class_keys,
        "flavia_labels": flavia_labels,
        "input_size": INPUT_SIZE,
        "min_lr": MIN_LR,
        "grad_clip": GRAD_CLIP,
        "device_used": str(device),
        "num_workers_used": used_workers,
        "torch_version": torch.__version__,
        "parameter_count": count_parameters(model),
        "train_size": len(loaders["train"].dataset),
        "val_size": len(loaders["val"].dataset),
        "test_size": len(loaders["test"].dataset) if "test" in loaders else 0,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "类别映射: %s",
        ", ".join(
            f"{i}:{k}(flavia={f})" for i, (k, f) in enumerate(zip(class_keys, flavia_labels))
        ),
    )

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    max_batches = max(0, int(args.max_batches))

    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        history_file.flush()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            lr_backbone, lr_head = optimizer_lrs(optimizer)

            # ---- train ----
            model.train()
            train_loss = AverageMeter()
            train_correct = 0
            train_total = 0
            for batch_idx, (images, labels, _filenames) in enumerate(loaders["train"]):
                images = images.to(device, non_blocking=False)
                labels = labels.to(device, non_blocking=False)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                batch_size = int(labels.numel())
                train_loss.update(loss.item(), batch_size)
                train_correct += int((logits.argmax(dim=1) == labels).sum().item())
                train_total += batch_size

                if max_batches > 0 and batch_idx + 1 >= max_batches:
                    break

            # ---- val ----
            # 与 test 使用同一无平滑 CE，val_loss 才能和 test ce_loss 直接比较。
            val = evaluate(model, loaders["val"], eval_criterion, device, max_batches)
            scheduler.step()

            epoch_sec = time.perf_counter() - epoch_start
            record = {
                "epoch": epoch,
                "lr": round(lr_head, 10),
                "train_loss": round(train_loss.avg, 6),
                "train_acc": round(train_correct / max(train_total, 1), 6),
                "val_loss": round(val["loss"], 6),
                "val_acc": round(val["acc"], 6),
                "val_f1": round(val["f1"], 6),
                "epoch_sec": round(epoch_sec, 3),
            }
            history.append(record)
            writer.writerow(record)
            history_file.flush()
            logger.info(
                "Epoch %02d/%02d | lr backbone=%.2e head=%.2e | "
                "train loss=%.4f acc=%.4f | val loss=%.4f acc=%.4f f1=%.4f | %.1fs",
                epoch, args.epochs, lr_backbone, lr_head,
                record["train_loss"], record["train_acc"],
                record["val_loss"], record["val_acc"], record["val_f1"], epoch_sec,
            )

            # ---- checkpoint ----
            state = {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_id": model_id,
                "class_keys": class_keys,
                "flavia_labels": flavia_labels,
                "num_classes": args.num_classes,
                "input_size": INPUT_SIZE,
                "epoch": epoch,
                "val_acc": record["val_acc"],
                "val_f1": record["val_f1"],
                "config": config,
            }
            save_checkpoint(run_dir / "last.pt", state)

            # val macro-F1 优先，平局看 val loss（越小越好）
            improved = (
                record["val_f1"] > best_f1 + 1e-9
                or (abs(record["val_f1"] - best_f1) <= 1e-9 and record["val_loss"] < best_loss)
            )
            if improved:
                best_f1 = record["val_f1"]
                best_loss = record["val_loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(run_dir / "best.pt", state)
                logger.info("  -> 新的 best.pt（epoch %d, val_f1=%.4f）", epoch, best_f1)
            else:
                epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    logger.info(
                        "Early stopping：连续 %d 个 epoch 无提升（best epoch=%d）。",
                        epochs_without_improvement, best_epoch,
                    )
                    break

    (run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 用 best.pt 在 test 上评估 ----
    test_metrics: dict[str, Any]
    if "test" not in loaders:
        logger.warning("未找到 test.csv，跳过 test 评估。")
        test_metrics = {
            "status": "skipped",
            "reason": "test.csv not found",
            "model_id": model_id,
            "model_name": MODEL_NAMES[model_id],
        }
    else:
        checkpoint = load_checkpoint(run_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        test = evaluate(model, loaders["test"], eval_criterion, device, max_batches=0)
        topk_k = topk_k_for(args.num_classes)
        test_metrics = {
            "model_id": model_id,
            "model_name": MODEL_NAMES[model_id],
            "checkpoint": str(run_dir / "best.pt"),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "num_samples": test["n"],
            "accuracy": round(test["acc"], 6),
            "macro_f1": round(test["f1"], 6),
            "ce_loss": round(test["loss"], 6),
            "topk_k": topk_k,
            "topk_accuracy": round(test["topk_acc"], 6),
            "class_keys": class_keys,
            "flavia_labels": flavia_labels,
            "num_classes": args.num_classes,
            "input_size": INPUT_SIZE,
            "seed": args.seed,
        }
        # 兼容既有报告：k=3 保留 top3_accuracy，k=5 时补充 top5_accuracy。
        if topk_k == 3:
            test_metrics["top3_accuracy"] = test_metrics["topk_accuracy"]
        elif topk_k == 5:
            test_metrics["top5_accuracy"] = test_metrics["topk_accuracy"]
        logger.info(
            "Test | acc=%.4f macro_f1=%.4f ce_loss=%.4f top%d=%.4f (n=%d, best epoch=%d)",
            test["acc"], test["f1"], test["loss"], topk_k, test["topk_acc"],
            test["n"], test_metrics["checkpoint_epoch"],
        )

    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("完成 %s：best epoch=%d, val_f1=%.4f", MODEL_NAMES[model_id], best_epoch, best_f1)
    return test_metrics


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.threads))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    class_keys, flavia_labels = load_registry(args.registry, args.num_classes)
    # 训练任何模型之前，先确保每个 split 的标签与注册表完全一致。
    checked_splits = validate_split_labels(args.data, class_keys, flavia_labels)
    if not checked_splits:
        raise FileNotFoundError(
            f"在 {args.data} 下未找到 train/val/test.csv 中的任何一个"
        )
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    logger = get_logger("src.train")
    logger.info(
        "标签交叉校验通过：%s",
        ", ".join(f"{split}={rows}" for split, rows in checked_splits.items()),
    )
    logger.info(
        "开始训练：models=%s | data=%s | num_classes=%d | epochs=%d | "
        "threads=%d | max_batches=%d | device=%s",
        args.models, args.data, args.num_classes, args.epochs,
        args.threads, args.max_batches, device,
    )
    logger.info("注册表类别: %s", class_keys)

    summaries = []
    for model_id in args.models:
        summaries.append(
            train_one_model(model_id, args, class_keys, flavia_labels, device)
        )

    logger.info("=" * 78)
    logger.info("全部训练完成，共 %d 个模型：", len(summaries))
    for item in summaries:
        if item.get("status") == "skipped":
            logger.info("  %s: test 跳过", item["model_name"])
        else:
            logger.info(
                "  %s: test acc=%.4f f1=%.4f top%d=%.4f",
                item["model_name"], item["accuracy"], item["macro_f1"],
                item["topk_k"], item["topk_accuracy"],
            )


if __name__ == "__main__":
    main()
