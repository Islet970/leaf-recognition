"""生成 A/B/C/D 四模型对比图（plans.md §4.2）。

输出（默认 ``pic/``）：
    fig4_1_train_loss_ABCD.png      训练损失 vs 轮次
    fig4_2_train_acc_ABCD.png       训练准确率 vs 轮次（含 20% 随机基线）
    fig4_4_test_acc_bar.png         测试准确率柱状图
    fig4_3_test_loss_bar.png        测试 loss 柱状图
    fig4_6_sample_predictions_ABCD.png  3 个测试样本 × 4 个模型的识别网格

数据来源：
    训练曲线  ``{runs}/{MODEL_NAMES[mid]}_seed{seed}/history.csv``
    测试指标  ``{runs}/{MODEL_NAMES[mid]}_seed{seed}/test_metrics.json``
    样本推理  ``{runs}/.../best.pt`` + ``{data}/test.csv`` + ``data/Leaves/``
    中文类别  ``data/class_registry.json``（与 checkpoint 的 ``class_keys`` 对齐）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_eval_transform  # noqa: E402
from src.models import MODEL_NAMES, build_model  # noqa: E402

# 统一配色：A 蓝、B 橙、C 绿、D 红（与报告其他图保持一致）
COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
}
CORRECT_COLOR = "#2e7d32"
WRONG_COLOR = "#c62828"
RANDOM_BASELINE = 0.2  # 五类随机猜测准确率


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 A/B/C/D 四模型对比图（训练曲线、测试指标柱状图、样本识别网格）"
    )
    parser.add_argument("--runs", default="runs/5class", help="run 根目录（默认 runs/5class）")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="参与对比的模型编号（默认 A B C D）",
    )
    parser.add_argument(
        "--data", default="data/splits/5class", help="划分目录，读取 test.csv（默认 data/splits/5class）"
    )
    parser.add_argument(
        "--registry", default="data/class_registry.json", help="类别注册表路径"
    )
    parser.add_argument("--seed", type=int, default=42, help="run 目录中的 seed 后缀")
    parser.add_argument("--out", default="pic", help="图片输出目录（默认 pic）")
    parser.add_argument("--dpi", type=int, default=160, help="输出 DPI（默认 160）")
    args = parser.parse_args(argv)

    # 规范化并去重模型编号，保持用户给定顺序
    seen: set[str] = set()
    models: list[str] = []
    for model in args.models:
        mid = str(model).strip().upper()
        if mid not in MODEL_NAMES:
            raise SystemExit(
                f"未知模型编号 {model!r}，可选: {sorted(MODEL_NAMES)}"
            )
        if mid not in seen:
            seen.add(mid)
            models.append(mid)
    args.models = models
    return args


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def run_dir_for(runs_root: Path, model_id: str, seed: int) -> Path:
    run_dir = runs_root / f"{MODEL_NAMES[model_id]}_seed{seed}"
    if not run_dir.is_dir():
        expected = runs_root / f"{MODEL_NAMES[model_id]}_seed{seed}"
        raise SystemExit(
            f"找不到模型 {model_id} 的 run 目录: {expected}\n"
            f"请检查 --runs / --seed，或确认该模型已完成训练。"
        )
    return run_dir


def load_history(run_dir: Path) -> dict[str, list[float]]:
    path = run_dir / "history.csv"
    if not path.is_file():
        raise SystemExit(f"找不到训练历史文件: {path}")
    epochs: list[int] = []
    train_loss: list[float] = []
    train_acc: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("epoch", "train_loss", "train_acc") if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} 缺少列: {missing}")
        for row in reader:
            epochs.append(int(float(row["epoch"])))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
    if not epochs:
        raise SystemExit(f"训练历史为空: {path}")
    return {"epoch": epochs, "train_loss": train_loss, "train_acc": train_acc}


def load_test_metrics(run_dir: Path) -> dict:
    path = run_dir / "test_metrics.json"
    if not path.is_file():
        raise SystemExit(f"找不到测试指标文件: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    for key in ("accuracy", "ce_loss"):
        if key not in metrics:
            raise SystemExit(f"{path} 缺少字段 {key!r}")
    return metrics


def load_zh_names(registry_path: Path) -> dict[str, str]:
    if not registry_path.is_file():
        raise SystemExit(f"找不到类别注册表: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    classes = registry.get("classes")
    if not isinstance(classes, list) or not classes:
        raise SystemExit(f"注册表缺少非空 classes 列表: {registry_path}")
    zh_by_key: dict[str, str] = {}
    for item in classes:
        key = str(item.get("key", ""))
        zh_name = str(item.get("zh_name", ""))
        if not key or not zh_name:
            raise SystemExit(f"注册表条目缺少 key/zh_name: {item}")
        zh_by_key[key] = zh_name
    return zh_by_key


def load_test_manifest(data_dir: Path) -> tuple[list[str], np.ndarray]:
    csv_path = data_dir / "test.csv"
    if not csv_path.is_file():
        raise SystemExit(f"找不到测试清单: {csv_path}")
    filenames: list[str] = []
    labels: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("filename", "py_label") if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{csv_path} 缺少列: {missing}")
        for row in reader:
            filenames.append(str(row["filename"]))
            labels.append(int(row["py_label"]))
    if not filenames:
        raise SystemExit(f"测试清单为空: {csv_path}")
    return filenames, np.asarray(labels, dtype=np.int64)


def load_checkpoint_model(model_id: str, run_dir: Path) -> tuple[torch.nn.Module, list[str]]:
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"找不到 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "class_keys" not in checkpoint:
        raise SystemExit(f"checkpoint 缺少 model_state / class_keys: {ckpt_path}")
    class_keys = [str(k) for k in checkpoint["class_keys"]]
    num_classes = int(checkpoint.get("num_classes", len(class_keys)))
    model = build_model(model_id, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_keys


def resolve_class_keys(
    models: list[str], checkpoints: dict[str, tuple[torch.nn.Module, list[str]]]
) -> list[str]:
    reference = checkpoints[models[0]][1]
    for mid in models[1:]:
        keys = checkpoints[mid][1]
        if keys != reference:
            raise SystemExit(
                f"模型 {mid} 的 class_keys 与 {models[0]} 不一致：\n"
                f"  {models[0]}: {reference}\n  {mid}: {keys}"
            )
    return reference


def infer_test_set(
    models: list[str],
    checkpoints: dict[str, tuple[torch.nn.Module, list[str]]],
    image_root: Path,
    filenames: list[str],
    batch_size: int = 64,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """在 test 上推理，返回 {mid: (pred_indices, confidences)}。"""
    transform = build_eval_transform()
    tensors: list[torch.Tensor] = []
    for filename in filenames:
        path = image_root / filename
        if not path.is_file():
            raise SystemExit(f"找不到测试图片: {path}")
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            tensors.append(transform(img))

    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mid in models:
        model = checkpoints[mid][0]
        preds: list[int] = []
        confs: list[float] = []
        with torch.no_grad():
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start : start + batch_size])
                probs = torch.softmax(model(batch), dim=1)
                conf, pred = probs.max(dim=1)
                preds.extend(pred.tolist())
                confs.extend(conf.tolist())
        outputs[mid] = (np.asarray(preds, dtype=np.int64), np.asarray(confs, dtype=np.float64))
    return outputs


def select_samples(
    models: list[str],
    filenames: list[str],
    labels: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    count: int = 3,
) -> list[tuple[int, str]]:
    """确定性样本选择，返回 [(index, 选择理由)]。"""
    preds = {mid: predictions[mid][0] for mid in models}
    confs = {mid: predictions[mid][1] for mid in models}
    correct = {mid: preds[mid] == labels for mid in models}

    chosen: list[tuple[int, str]] = []
    chosen_indices: set[int] = set()

    def take(indices, reason: str) -> None:
        for idx in indices:
            if len(chosen) >= count:
                return
            if idx in chosen_indices:
                continue
            chosen.append((int(idx), reason))
            chosen_indices.add(int(idx))

    # 1) A 错而其余模型全对
    if "A" in models:
        others = [mid for mid in models if mid != "A"]
        rule1 = [
            i
            for i in range(len(filenames))
            if not correct["A"][i] and all(correct[mid][i] for mid in others)
        ]
        take(sorted(rule1, key=lambda i: filenames[i]), "A 错、其余模型全对")

    # 2) 四个模型预测不一致
    rule2 = [
        i
        for i in range(len(filenames))
        if len({preds[mid][i] for mid in models}) > 1
    ]
    take(sorted(rule2, key=lambda i: filenames[i]), "模型预测不一致")

    # 3) 最大置信度最低（每个样本取所有模型的最大置信度）
    stacked = np.stack([confs[mid] for mid in models], axis=0)  # [M, N]
    max_conf = stacked.max(axis=0)
    rule3 = sorted(range(len(filenames)), key=lambda i: (float(max_conf[i]), filenames[i]))
    take(rule3, "最大置信度最低")

    # 4) 按文件名顺序补齐
    take(sorted(range(len(filenames)), key=lambda i: filenames[i]), "按文件名补齐")
    return chosen


def plot_train_loss(
    models: list[str], histories: dict[str, dict[str, list[float]]], out_dir: Path, dpi: int
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for mid in models:
        history = histories[mid]
        ax.plot(
            history["epoch"],
            history["train_loss"],
            color=COLORS[mid],
            linewidth=1.9,
            marker="o",
            markersize=3.2,
            markevery=4,
            label=f"{mid}（{MODEL_NAMES[mid]}）",
        )
    ax.set_xlabel("Epoch（迭代轮次）")
    ax.set_ylabel("训练损失")
    ax.set_title("A/B/C/D 训练损失对比")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    path = out_dir / "fig4_1_train_loss_ABCD.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_train_acc(
    models: list[str], histories: dict[str, dict[str, list[float]]], out_dir: Path, dpi: int
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for mid in models:
        history = histories[mid]
        ax.plot(
            history["epoch"],
            history["train_acc"],
            color=COLORS[mid],
            linewidth=1.9,
            marker="o",
            markersize=3.2,
            markevery=4,
            label=f"{mid}（{MODEL_NAMES[mid]}）",
        )
    ax.axhline(
        RANDOM_BASELINE,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.3,
        label=f"随机基线（{RANDOM_BASELINE:.0%}）",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Epoch（迭代轮次）")
    ax.set_ylabel("训练准确率")
    ax.set_title("A/B/C/D 训练准确率对比")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    path = out_dir / "fig4_2_train_acc_ABCD.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_test_acc(
    models: list[str], metrics: dict[str, dict], out_dir: Path, dpi: int
) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    positions = np.arange(len(models))
    values = [float(metrics[mid]["accuracy"]) for mid in models]
    bars = ax.bar(
        positions,
        values,
        width=0.6,
        color=[COLORS[mid] for mid in models],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{mid}\n{MODEL_NAMES[mid]}" for mid in models])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("测试准确率")
    ax.set_title("A/B/C/D 测试准确率对比")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig4_4_test_acc_bar.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_test_loss(
    models: list[str], metrics: dict[str, dict], out_dir: Path, dpi: int
) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    positions = np.arange(len(models))
    values = [float(metrics[mid]["ce_loss"]) for mid in models]
    bars = ax.bar(
        positions,
        values,
        width=0.6,
        color=[COLORS[mid] for mid in models],
        edgecolor="black",
        linewidth=0.8,
    )
    y_top = max(values) * 1.18 if values else 1.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_top * 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{mid}\n{MODEL_NAMES[mid]}" for mid in models])
    ax.set_ylim(0.0, y_top)
    ax.set_ylabel("测试交叉熵损失")
    ax.set_title("A/B/C/D 测试损失对比")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig4_3_test_loss_bar.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_sample_predictions(
    models: list[str],
    selected: list[tuple[int, str]],
    filenames: list[str],
    labels: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    class_keys: list[str],
    zh_by_key: dict[str, str],
    image_root: Path,
    out_dir: Path,
    dpi: int,
) -> Path:
    n_rows = len(selected)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.7 * n_rows + 0.6))
    axes = np.atleast_2d(axes)

    def zh_of(index: int) -> str:
        key = class_keys[index]
        if key not in zh_by_key:
            raise SystemExit(f"checkpoint 类别 {key!r} 不在注册表中")
        return zh_by_key[key]

    for row, (sample_idx, _reason) in enumerate(selected):
        true_zh = zh_of(int(labels[sample_idx]))
        image_path = image_root / filenames[sample_idx]
        with Image.open(image_path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        for col, mid in enumerate(models):
            ax = axes[row, col]
            ax.imshow(image)
            pred_idx = int(predictions[mid][0][sample_idx])
            conf = float(predictions[mid][1][sample_idx])
            pred_zh = zh_of(pred_idx)
            is_correct = pred_idx == int(labels[sample_idx])
            color = CORRECT_COLOR if is_correct else WRONG_COLOR
            ax.set_title(
                f"{mid}（{MODEL_NAMES[mid]}）\n预测：{pred_zh}  {conf * 100:.1f}%",
                fontsize=10.5,
                color=color,
                pad=6,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3.2)

    # 行首的真实类别标签（tight_layout 后按行中心定位）
    fig.tight_layout(rect=(0.085, 0.0, 1.0, 0.94))
    for row, (sample_idx, _reason) in enumerate(selected):
        bbox = axes[row, 0].get_position()
        fig.text(
            0.012,
            (bbox.y0 + bbox.y1) / 2,
            f"真实：{zh_of(int(labels[sample_idx]))}\n{filenames[sample_idx]}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    path = out_dir / "fig4_6_sample_predictions_ABCD.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def print_sample_summary(
    selected: list[tuple[int, str]],
    filenames: list[str],
    labels: np.ndarray,
    models: list[str],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    class_keys: list[str],
    zh_by_key: dict[str, str],
) -> None:
    print("\n选中样本预测对比：")
    for sample_idx, reason in selected:
        true_zh = zh_by_key[class_keys[int(labels[sample_idx])]]
        print(
            f"  [{filenames[sample_idx]}] 真实：{true_zh}（py_label={int(labels[sample_idx])}，"
            f"选择理由：{reason}）"
        )
        for mid in models:
            pred_idx = int(predictions[mid][0][sample_idx])
            conf = float(predictions[mid][1][sample_idx])
            pred_zh = zh_by_key[class_keys[pred_idx]]
            mark = "正确" if pred_idx == int(labels[sample_idx]) else "错误"
            print(
                f"    {mid} {MODEL_NAMES[mid]:<22} 预测：{pred_zh:<4} 置信度：{conf * 100:6.2f}%  [{mark}]"
            )


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    configure_fonts()

    runs_root = Path(args.runs)
    data_dir = Path(args.data)
    registry_path = Path(args.registry)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = PROJECT_ROOT / "data" / "Leaves"

    models: list[str] = args.models
    if not models:
        raise SystemExit("--models 至少需要一个模型编号")

    run_dirs = {mid: run_dir_for(runs_root, mid, args.seed) for mid in models}
    histories = {mid: load_history(run_dirs[mid]) for mid in models}
    metrics = {mid: load_test_metrics(run_dirs[mid]) for mid in models}
    zh_by_key = load_zh_names(registry_path)

    # ---- 图 4.1 / 4.2 ----
    generated: list[Path] = []
    generated.append(plot_train_loss(models, histories, out_dir, args.dpi))
    generated.append(plot_train_acc(models, histories, out_dir, args.dpi))

    # ---- 测试指标摘要 + 图 4.4 / 4.3 ----
    print("测试指标摘要（来自 test_metrics.json）：")
    for mid in models:
        print(
            f"  {mid} {MODEL_NAMES[mid]:<22} "
            f"test acc = {metrics[mid]['accuracy'] * 100:6.2f}%   "
            f"test loss = {metrics[mid]['ce_loss']:.4f}"
        )
    generated.append(plot_test_acc(models, metrics, out_dir, args.dpi))
    generated.append(plot_test_loss(models, metrics, out_dir, args.dpi))

    # ---- 图 4.6：样本识别网格 ----
    checkpoints = {mid: load_checkpoint_model(mid, run_dirs[mid]) for mid in models}
    class_keys = resolve_class_keys(models, checkpoints)
    missing_keys = [key for key in class_keys if key not in zh_by_key]
    if missing_keys:
        raise SystemExit(f"checkpoint 类别不在注册表中: {missing_keys}")

    filenames, labels = load_test_manifest(data_dir)
    num_classes = len(class_keys)
    bad_labels = sorted({int(v) for v in labels if int(v) < 0 or int(v) >= num_classes})
    if bad_labels:
        raise SystemExit(f"test.csv 中 py_label 越界（num_classes={num_classes}）: {bad_labels}")

    predictions = infer_test_set(models, checkpoints, image_root, filenames)
    selected = select_samples(models, filenames, labels, predictions)

    print_sample_summary(selected, filenames, labels, models, predictions, class_keys, zh_by_key)
    generated.append(
        plot_sample_predictions(
            models,
            selected,
            filenames,
            labels,
            predictions,
            class_keys,
            zh_by_key,
            image_root,
            out_dir,
            args.dpi,
        )
    )

    print("\n已生成图片：")
    for path in generated:
        print(f"  {path}  ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
