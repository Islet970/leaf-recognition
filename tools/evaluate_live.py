"""在实景清单上评测 A/B/C/D checkpoint，并画组合柱状图（plans.md §15.3–§15.6）。

输入：
    --runs       run 根目录，实际读取 ``{runs}/{MODEL_NAMES[mid]}_{seed}/best.pt``
    --registry   类别注册表（5 类或 32 类），用于校验 checkpoint 的
                 ``class_keys`` / ``flavia_labels``，并决定 ``num_classes``
    --manifest   实景清单 CSV（至少含 ``filename``、``py_label`` 两列）
    --image-root 图片根目录；``filename`` 相对该目录（如 ``data/leaf``）

输出：
    {out}/{tag}_ABCD.png        1×2 组合图：测试准确率柱状 + 测试损失柱状
    {out}/metrics_{tag}.csv     列：model,accuracy,macro_f1,ce_loss,n

示例（E1，W5 五类白底模型 → 实景普通测试集）：
    python tools/evaluate_live.py \
      --runs runs/5class --models A B C D \
      --registry data/class_registry.json \
      --manifest data/splits/live_all/test.csv \
      --image-root data/leaf --out pic/spe-train-live-test --tag test --seed 42

32 类模式（E3b）只把 ``--runs`` 换成 ``runs/all32``、``--registry`` 换成
``data/class_registry_all32.json``、``--manifest`` 换成
``data/splits/live_all32/test.csv``；模型按注册表类别数构建为 32 类，
类别名以 checkpoint 的 ``class_keys`` 为准并与注册表逐项校验。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# 与训练/绘图一致的容错策略：允许尾部数据缺失的图片正常解码。
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import LeafDataset, build_eval_transform  # noqa: E402
from src.models import MODEL_NAMES, build_model  # noqa: E402
from src.utils import load_checkpoint, set_seed  # noqa: E402

# 与 make_comparison_plots.py 保持一致：A 蓝、B 橙、C 绿、D 红
COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
}

# x 轴“A/B/C/D + 模型简称”
SHORT_NAMES = {
    "A": "MobileNetV3-S",
    "B": "MobileNetV2",
    "C": "+MSFF",
    "D": "+MSFF+CBAM",
    "E": "+CBAM",
}

TAG_TITLES = {
    "test": "普通实景测试集",
    "test_hard": "困难实景测试集",
    "hard_test": "困难实景测试集",
}

REQUIRED_COLUMNS = ("filename", "py_label")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载 A/B/C/D best.pt，在实景清单上计算准确率 / macro-F1 / CE loss，"
        "并输出组合柱状图与 metrics CSV。"
    )
    parser.add_argument("--runs", required=True, help="run 根目录，如 runs/5class 或 runs/all32")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="参与评测的模型编号（默认 A B C D）",
    )
    parser.add_argument(
        "--registry", required=True, help="类别注册表 JSON（5 类或 32 类）"
    )
    parser.add_argument(
        "--manifest", required=True, help="实景清单 CSV（filename, py_label）"
    )
    parser.add_argument(
        "--image-root", required=True, help="图片根目录，filename 相对该目录"
    )
    parser.add_argument("--out", required=True, help="图片与 CSV 输出目录")
    parser.add_argument("--tag", default="test", help="输出文件名后缀（默认 test）")
    parser.add_argument("--seed", type=int, default=42, help="run 目录中的 seed 后缀")
    parser.add_argument("--batch-size", type=int, default=64, help="评测 batch size（默认 64）")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader worker 数（Windows 建议 0）"
    )
    parser.add_argument("--dpi", type=int, default=160, help="图片 DPI（默认 160）")
    parser.add_argument("--device", default="cpu", help="推理设备（默认 cpu）")
    args = parser.parse_args(argv)

    # 规范化并去重模型编号，保持用户给定顺序
    seen: set[str] = set()
    models: list[str] = []
    for model in args.models:
        mid = str(model).strip().upper()
        if mid not in MODEL_NAMES:
            raise SystemExit(f"未知模型编号 {model!r}，可选: {sorted(MODEL_NAMES)}")
        if mid not in seen:
            seen.add(mid)
            models.append(mid)
    args.models = models
    return args


def project_path(value: str | Path) -> Path:
    """相对路径按项目根目录解析，便于从任意目录调用。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def load_registry(registry_path: Path) -> tuple[list[str], list[int]]:
    """读取注册表，返回按注册表顺序排列的 class_keys 与 flavia_labels。"""
    if not registry_path.is_file():
        raise SystemExit(f"找不到类别注册表: {registry_path}")
    try:
        obj = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取类别注册表 {registry_path}: {exc}") from exc

    classes = obj.get("classes")
    if not isinstance(classes, list) or not classes:
        raise SystemExit(f"类别注册表 {registry_path} 缺少非空 classes 列表")

    class_keys: list[str] = []
    flavia_labels: list[int] = []
    py_labels: list[int] = []
    for index, item in enumerate(classes):
        if not isinstance(item, dict):
            raise SystemExit(f"注册表 classes[{index}] 不是对象")
        for field in ("key", "py_label", "flavia_label"):
            if field not in item:
                raise SystemExit(f"注册表 classes[{index}] 缺少字段 {field!r}")
        class_keys.append(str(item["key"]))
        py_labels.append(int(item["py_label"]))
        flavia_labels.append(int(item["flavia_label"]))

    if py_labels != list(range(len(classes))):
        raise SystemExit(
            "注册表 classes 的 py_label 必须从 0 开始连续递增，"
            f"实际为 {py_labels[:8]}{'...' if len(py_labels) > 8 else ''}"
        )
    return class_keys, flavia_labels


def validate_manifest(
    manifest_path: Path, image_root: Path, num_classes: int
) -> pd.DataFrame:
    """读取并校验实景清单，同时预检图片是否存在。"""
    if not manifest_path.is_file():
        raise SystemExit(f"找不到实景清单: {manifest_path}")
    if not image_root.is_dir():
        raise SystemExit(f"找不到图片根目录: {image_root}")

    df = pd.read_csv(manifest_path)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise SystemExit(
            f"清单 {manifest_path} 缺少列 {missing_columns}；"
            f"至少需要 {list(REQUIRED_COLUMNS)}"
        )
    if df.empty:
        raise SystemExit(f"清单为空: {manifest_path}")

    labels = pd.to_numeric(df["py_label"], errors="coerce")
    if labels.isna().any():
        bad = df.loc[labels.isna(), "filename"].astype(str).head(5).tolist()
        raise SystemExit(f"清单 py_label 存在非数值项，示例: {bad}")
    labels = labels.astype(int)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise SystemExit(
            f"清单 py_label 超出 [0, {num_classes - 1}] 范围: "
            f"min={labels.min()}, max={labels.max()}"
        )

    filenames = df["filename"].astype(str)
    missing_files = [
        name for name in filenames if not (image_root / name).is_file()
    ]
    if missing_files:
        preview = missing_files[:5]
        raise SystemExit(
            f"有 {len(missing_files)} 张图片不存在（image-root={image_root}），"
            f"示例: {preview}"
        )

    return df


def load_model_for_eval(
    runs_root: Path,
    model_id: str,
    seed: int,
    class_keys: list[str],
    flavia_labels: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """加载并校验单个模型的 best.pt。"""
    run_dir = runs_root / f"{MODEL_NAMES[model_id]}_seed{seed}"
    if not run_dir.is_dir():
        raise SystemExit(
            f"找不到模型 {model_id} 的 run 目录: {run_dir}\n"
            f"请检查 --runs / --seed，或确认该模型已完成训练。"
        )
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"模型 {model_id} 缺少 checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise SystemExit(f"checkpoint 格式不正确（缺少 model_state）: {checkpoint_path}")

    ckpt_model_id = checkpoint.get("model_id")
    if ckpt_model_id is not None and str(ckpt_model_id).upper() != model_id:
        raise SystemExit(
            f"{checkpoint_path} 的 model_id={ckpt_model_id!r} 与请求的 {model_id} 不一致"
        )

    ckpt_keys = checkpoint.get("class_keys")
    if ckpt_keys is None:
        raise SystemExit(f"{checkpoint_path} 缺少 class_keys，无法校验类别顺序")
    ckpt_keys = [str(k) for k in ckpt_keys]
    if ckpt_keys != class_keys:
        raise SystemExit(
            f"{checkpoint_path} 的 class_keys 与注册表 classes 不一致。\n"
            f"checkpoint: {ckpt_keys}\nregistry  : {class_keys}"
        )

    ckpt_flavia = checkpoint.get("flavia_labels")
    if ckpt_flavia is None:
        raise SystemExit(f"{checkpoint_path} 缺少 flavia_labels，无法校验 Flavia 编号")
    ckpt_flavia = [int(v) for v in ckpt_flavia]
    if ckpt_flavia != flavia_labels:
        raise SystemExit(
            f"{checkpoint_path} 的 flavia_labels 与注册表不一致。\n"
            f"checkpoint: {ckpt_flavia}\nregistry  : {flavia_labels}"
        )

    ckpt_num_classes = checkpoint.get("num_classes")
    if ckpt_num_classes is not None and int(ckpt_num_classes) != num_classes:
        raise SystemExit(
            f"{checkpoint_path} 的 num_classes={ckpt_num_classes} 与注册表类别数 "
            f"{num_classes} 不一致"
        )

    model = build_model(model_id, num_classes, pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise SystemExit(
            f"模型 {model_id} 的 state_dict 与 num_classes={num_classes} 的结构不匹配: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float | int]:
    """计算 accuracy、macro-F1 与无平滑交叉熵损失。"""
    total = 0
    correct = 0
    loss_sum = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels, _filenames in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        preds = logits.argmax(dim=1)
        total += int(labels.numel())
        correct += int((preds == labels).sum().item())
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    if total == 0:
        raise SystemExit("清单没有可评测样本")

    accuracy = correct / total
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    ce_loss = loss_sum / total
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ce_loss": ce_loss,
        "n": total,
    }


def plot_metrics(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
    tag: str,
    dpi: int,
) -> None:
    """1 行 2 列：测试准确率柱状 + 测试损失柱状，四模型同图。"""
    configure_fonts()
    models = [str(row["model"]) for row in rows]
    acc_values = [float(row["accuracy"]) for row in rows]
    loss_values = [float(row["ce_loss"]) for row in rows]
    positions = np.arange(len(models))
    bar_colors = [COLORS.get(mid, "#7f7f7f") for mid in models]
    tick_labels = [
        f"{mid}\n{SHORT_NAMES.get(mid, MODEL_NAMES.get(mid, mid))}" for mid in models
    ]
    tag_title = TAG_TITLES.get(tag, tag)
    n = int(rows[0]["n"]) if rows else 0

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13.0, 5.4))

    # 左：测试准确率
    acc_bars = ax_acc.bar(
        positions,
        acc_values,
        width=0.6,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, value in zip(acc_bars, acc_values):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_acc.set_xticks(positions)
    ax_acc.set_xticklabels(tick_labels)
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.set_ylabel("测试准确率")
    ax_acc.set_title(f"A/B/C/D 测试准确率（{tag_title}，n={n}）")
    ax_acc.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 右：测试 loss
    loss_bars = ax_loss.bar(
        positions,
        loss_values,
        width=0.6,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    max_loss = max(loss_values) if loss_values else 0.0
    y_top = max_loss * 1.18 if max_loss > 0 else 1.0
    for bar, value in zip(loss_bars, loss_values):
        ax_loss.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_top * 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_loss.set_xticks(positions)
    ax_loss.set_xticklabels(tick_labels)
    ax_loss.set_ylim(0.0, y_top)
    ax_loss.set_ylabel("测试交叉熵损失")
    ax_loss.set_title(f"A/B/C/D 测试损失（{tag_title}，n={n}）")
    ax_loss.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    runs_root = project_path(args.runs)
    registry_path = project_path(args.registry)
    manifest_path = project_path(args.manifest)
    image_root = project_path(args.image_root)
    out_dir = project_path(args.out)

    class_keys, flavia_labels = load_registry(registry_path)
    num_classes = len(class_keys)
    df = validate_manifest(manifest_path, image_root, num_classes)

    print(
        f"清单: {manifest_path}  样本数: {len(df)}  "
        f"类别数: {num_classes}  seed: {args.seed}"
    )
    print("清单类别分布:", df["py_label"].astype(int).value_counts().sort_index().to_dict())

    device = torch.device(args.device)
    dataset = LeafDataset(
        csv_path=manifest_path,
        image_root=image_root,
        transform=build_eval_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    rows: list[dict[str, float | int | str]] = []
    for model_id in args.models:
        model = load_model_for_eval(
            runs_root=runs_root,
            model_id=model_id,
            seed=args.seed,
            class_keys=class_keys,
            flavia_labels=flavia_labels,
            num_classes=num_classes,
            device=device,
        )
        metrics = evaluate_model(model, loader, device)
        record = {"model": model_id, **metrics}
        rows.append(record)
        print(
            f"[{model_id}] {MODEL_NAMES[model_id]}: "
            f"accuracy={metrics['accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}  "
            f"ce_loss={metrics['ce_loss']:.4f}  n={metrics['n']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"metrics_{args.tag}.csv"
    pd.DataFrame(
        rows, columns=["model", "accuracy", "macro_f1", "ce_loss", "n"]
    ).to_csv(csv_path, index=False, float_format="%.6f")

    png_path = out_dir / f"{args.tag}_ABCD.png"
    plot_metrics(rows, png_path, tag=args.tag, dpi=args.dpi)
    print(f"指标 CSV: {csv_path}")
    print(f"组合图:   {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
