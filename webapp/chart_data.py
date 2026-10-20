"""pic/ 数据看板的 ECharts 图表规格生成（plans.md §5.7、§14.8 T8.6）。

本模块是纯数据层：
- 只读取 ``runs/`` 下的 ``history.csv`` / ``test_metrics.json`` 与 ``pic/`` 下的 CSV；
- 不读取 PNG 内容、不修改任何文件、不生成假数据；
- **按图降级**：某张图缺少文件、CSV 为空、列缺失或数值非法时，只跳过该图，
  目录内其他图仍正常返回；只有整个目录都无可用数据时 ``charts`` 才为 ``[]``。

数值约束：只接受有限浮点数（拒绝 ``NaN`` / ``Infinity``）；accuracy 类字段必须
位于 ``[0, 1]``，CE 损失类字段必须 ``>= 0``，越界值按缺失处理并告警。

公开入口：
    build_charts(directory_name, pic_dir, project_root) -> list[dict]

图表规格契约::

    {
      "id": str, "title": str, "type": "line" | "bar" | "groupedBar",
      "xName": str, "yName": str, "unit": "percent" | "value",
      "x": list, "series": [{"key": str, "name": str, "data": list}],
      "showDataLabels": bool,
    }
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- 常量

MODEL_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
ABLATION_ORDER: tuple[str, ...] = ("A", "E", "C", "D")

MODEL_FULL_NAMES: dict[str, str] = {
    "A": "A MobileNetV3-Small",
    "B": "B MobileNetV2",
    "C": "C LeafNet-MSFF",
    "D": "D LeafNet-MSFF+CBAM",
    "E": "E MobileNetV3-Small+CBAM",
}

# 精确别名白名单；白名单之外的模型名一律视为未知并跳过。
MODEL_ALIASES: dict[str, str] = {
    "A": "A",
    "A_MNV3S": "A",
    "A MOBILENETV3-SMALL": "A",
    "B": "B",
    "B_MNV2": "B",
    "B MOBILENETV2": "B",
    "C": "C",
    "C_LEAFNET_MSFF": "C",
    "C LEAFNET-MSFF": "C",
    "D": "D",
    "D_LEAFNET_MSFF_CBAM": "D",
    "D LEAFNET-MSFF+CBAM": "D",
    "E": "E",
    "E_MNV3S_CBAM": "E",
    "E MOBILENETV3-SMALL+CBAM": "E",
}

# 与 src/models.py 的 MODEL_NAMES 保持一致（runs/<run>/<name>_seed42/）
RUN_MODEL_NAMES: dict[str, str] = {
    "A": "A_mnv3s",
    "B": "B_mnv2",
    "C": "C_leafnet_msff",
    "D": "D_leafnet_msff_cbam",
    "E": "E_mnv3s_cbam",
}

MAIN_DATASETS: tuple[str, ...] = ("spec5", "live5", "spec32")
HARD_DATASETS: tuple[str, ...] = ("robust_w5", "robust_l5", "robust_w32")

DATASET_LABELS: dict[str, str] = {
    "spec5": "五类白底(n=58)",
    "live5": "五类实景(n=52)",
    "spec32": "32 类白底(n=358)",
    "robust_w5": "R1（W5）(n=227)",
    "robust_l5": "R2（L5）(n=227)",
    "robust_w32": "R3（W32）(n=227)",
}

# 困难帧原始指标文件（以 ablation_metrics.csv 为兜底）
ROBUST_RAW_FILES: dict[str, str] = {
    "robust_w5": "metrics_robust_w5.csv",
    "robust_l5": "metrics_robust_l5.csv",
    "robust_w32": "metrics_robust_w32.csv",
}

_GAIN_PAIRS: tuple[tuple[str, str], ...] = (("E", "A"), ("C", "A"), ("D", "A"))
_GAIN_LABELS: tuple[str, ...] = ("CBAM (E−A)", "MSFF (C−A)", "组合 (D−A)")


# --------------------------------------------------------------------------- 工具


def _warn(message: str) -> None:
    print(f"[chart_data] {message}", file=sys.stderr)


def normalize_model_name(raw: Any) -> str | None:
    """按精确别名白名单把模型名归一化为 ``A``–``E``。

    允许的写法（大小写不敏感、连续空白折叠）：
    ``A`` / ``A_mnv3s`` / ``A MobileNetV3-Small``，其余模型同理。
    白名单之外的名称返回 ``None``，并记录 warning。
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    key = " ".join(text.upper().split())
    model = MODEL_ALIASES.get(key)
    if model is None:
        _warn(f"未知模型名 {text!r}，已跳过")
    return model


def _to_float(value: Any) -> float | None:
    """转换为有限浮点数；``NaN`` / ``±Infinity`` / 非数值均返回 ``None``。"""
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _validated_value(raw: Any, field: str, context: str) -> float | None:
    """按字段约束校验数值：accuracy ∈ [0,1]，CE 损失 >= 0。

    非有限数或越界值按缺失处理并告警，避免把 ``NaN`` 写入图表规格。
    """
    value = _to_float(raw)
    if value is None:
        if raw is not None and str(raw).strip() != "":
            _warn(f"{context} 的 {field} 不是有限数值 {raw!r}，按缺失处理")
        return None

    if "accuracy" in field or field.endswith("_acc"):
        low, high = 0.0, 1.0
    elif "loss" in field:
        low, high = 0.0, None
    else:
        return value

    if value < low or (high is not None and value > high):
        _warn(f"{context} 的 {field}={value} 越界，按缺失处理")
        return None
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV，兼容 ``utf-8-sig`` 与 ``gbk``；失败时返回空列表。"""
    if not path.is_file():
        _warn(f"缺少 CSV 文件: {path}")
        return []

    for encoding in ("utf-8-sig", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                rows: list[dict[str, str]] = []
                for row in reader:
                    cleaned: dict[str, str] = {}
                    for key, value in row.items():
                        if key is None:  # 多余的列（无表头）
                            continue
                        cleaned[str(key).strip()] = (
                            value.strip() if isinstance(value, str) else value
                        )
                    rows.append(cleaned)
                return rows
        except UnicodeDecodeError:
            continue
        except (OSError, csv.Error) as exc:
            _warn(f"CSV 读取失败 {path}: {exc}")
            return []

    _warn(f"CSV 编码无法识别（utf-8-sig/gbk）: {path}")
    return []


def _read_json(path: Path) -> Any:
    if not path.is_file():
        _warn(f"缺少 JSON 文件: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _warn(f"JSON 读取失败 {path}: {exc}")
        return None


def _metrics_from_csv(path: Path) -> dict[str, dict[str, float]]:
    """解析 ``model,accuracy,macro_f1,ce_loss,n`` 形式的指标 CSV。"""
    metrics: dict[str, dict[str, float]] = {}
    for row in _read_csv(path):
        model = normalize_model_name(row.get("model"))
        if model is None:
            continue
        entry: dict[str, float] = {}
        for column in ("accuracy", "ce_loss"):
            value = _validated_value(
                row.get(column), column, f"{path.name} ({model})"
            )
            if value is not None:
                entry[column] = value
        if entry:
            metrics[model] = entry
    return metrics


# --------------------------------------------------------------------------- 数据源


def _epoch_key(value: Any) -> int | float | None:
    """把 epoch 转为整数（整数值）或有限浮点数；非法值返回 ``None``。"""
    epoch = _to_float(value)
    if epoch is None:
        return None
    return int(epoch) if epoch.is_integer() else epoch


def _history_points(
    run_root: Path, model: str, field: str
) -> dict[int | float, float | None]:
    """读取单个模型 ``history.csv``，返回 ``{epoch: value | None}``。

    - x 以 CSV 的 ``epoch`` 列为准，不重新编号；
    - 坏行（epoch 非数值）整行跳过；数值坏点保留 ``None`` 占位，
      因此后续 epoch 不会左移。
    """
    path = run_root / f"{RUN_MODEL_NAMES[model]}_seed42" / "history.csv"
    points: dict[int | float, float | None] = {}
    for row in _read_csv(path):
        raw_epoch = row.get("epoch")
        epoch = _epoch_key(raw_epoch)
        if epoch is None:
            if raw_epoch is not None and str(raw_epoch).strip() != "":
                _warn(
                    f"{path.name} ({model}) 的 epoch 非有限数值 "
                    f"{raw_epoch!r}，跳过该行"
                )
            continue

        value = _validated_value(
            row.get(field), field, f"{path.name} ({model}) epoch {epoch}"
        )
        # 同 epoch 重复时：有效值可覆盖；坏值不覆盖已有有效值
        if value is not None or epoch not in points:
            points[epoch] = value
    return points


def _run_test_metrics(run_root: Path) -> dict[str, dict[str, float]]:
    """从 ``runs/<run>/*_seed42/test_metrics.json`` 读取 A/B/C/D 测试指标。"""
    metrics: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        run_dir = run_root / f"{RUN_MODEL_NAMES[model]}_seed42"
        data = _read_json(run_dir / "test_metrics.json")
        if not isinstance(data, dict):
            continue
        entry: dict[str, float] = {}
        for column in ("accuracy", "ce_loss"):
            value = _validated_value(
                data.get(column), column, f"test_metrics.json ({model})"
            )
            if value is not None:
                entry[column] = value
        if entry:
            metrics[model] = entry
    return metrics


def _ablation_table(pic_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """合并 ablation_metrics.csv 与 raw/metrics_robust_*.csv，键为 (dataset, model)。"""
    table: dict[tuple[str, str], dict[str, float]] = {}

    for row in _read_csv(pic_dir / "ablation_metrics.csv"):
        dataset = str(row.get("dataset", "")).strip()
        if not dataset:
            continue
        model = normalize_model_name(row.get("model"))
        if model is None:
            continue
        entry: dict[str, float] = {}
        for column in ("accuracy", "ce_loss"):
            value = _validated_value(
                row.get(column), column, f"ablation_metrics.csv ({dataset}/{model})"
            )
            if value is not None:
                entry[column] = value
        if entry:
            table[(dataset, model)] = entry

    # 困难帧以 raw/ 下的原始 CSV 为准（内容与消融表一致时覆盖等价）
    for dataset, filename in ROBUST_RAW_FILES.items():
        for model, entry in _metrics_from_csv(pic_dir / "raw" / filename).items():
            table[(dataset, model)] = entry

    return table


# --------------------------------------------------------------------------- 图表构造


def _line_chart(
    chart_id: str,
    title: str,
    field: str,
    y_name: str,
    unit: str,
    run_root: Path,
) -> dict[str, Any] | None:
    """训练曲线：x 使用各模型 ``history.csv`` 的 epoch 交集。

    坏点保留 ``None`` 占位，不做重编号；某个模型完全无有效值时不出现在 series 中。
    """
    series_points: dict[str, dict[int | float, float | None]] = {}
    for model in MODEL_ORDER:
        points = _history_points(run_root, model, field)
        if any(value is not None for value in points.values()):
            series_points[model] = points

    if not series_points:
        return None

    common_epochs = set.intersection(
        *(set(points) for points in series_points.values())
    )
    x = sorted(common_epochs)
    if not x:
        return None

    return {
        "id": chart_id,
        "title": title,
        "type": "line",
        "xName": "轮次",
        "yName": y_name,
        "unit": unit,
        "x": x,
        "series": [
            {
                "key": model,
                "name": MODEL_FULL_NAMES[model],
                "data": [series_points[model].get(epoch) for epoch in x],
            }
            for model in MODEL_ORDER
            if model in series_points
        ],
        "showDataLabels": False,
    }


def _single_metric_bar(
    chart_id: str,
    title: str,
    metrics: dict[str, dict[str, float]],
    metric: str,
    y_name: str,
    unit: str,
) -> dict[str, Any] | None:
    """单指标柱状图：每个模型一个系列，仅自己位置有数值，其余为 null。"""
    x = list(MODEL_ORDER)
    series: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        value = (metrics.get(model) or {}).get(metric)
        if value is None:
            continue
        series.append(
            {
                "key": model,
                "name": MODEL_FULL_NAMES[model],
                "data": [value if other == model else None for other in x],
            }
        )

    if not series:
        return None

    return {
        "id": chart_id,
        "title": title,
        "type": "bar",
        "xName": "模型",
        "yName": y_name,
        "unit": unit,
        "x": x,
        "series": series,
        "showDataLabels": True,
    }


def _grouped_bar(
    chart_id: str,
    title: str,
    x: list[str],
    datasets: tuple[str, ...],
    table: dict[tuple[str, str], dict[str, float]],
    metric: str,
    y_name: str,
    unit: str,
) -> dict[str, Any] | None:
    series: list[dict[str, Any]] = []
    for dataset in datasets:
        data = [(table.get((dataset, model)) or {}).get(metric) for model in x]
        if any(value is not None for value in data):
            series.append(
                {
                    "key": dataset,
                    "name": DATASET_LABELS[dataset],
                    "data": data,
                }
            )

    if not series:
        return None

    return {
        "id": chart_id,
        "title": title,
        "type": "groupedBar",
        "xName": "模型",
        "yName": y_name,
        "unit": unit,
        "x": x,
        "series": series,
        "showDataLabels": False,
    }


def _gain_bar(
    chart_id: str,
    title: str,
    datasets: tuple[str, ...],
    table: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any] | None:
    """模块增益（百分点）：对应模型 accuracy − A accuracy，再 ×100。"""
    series: list[dict[str, Any]] = []
    for dataset in datasets:
        base = (table.get((dataset, "A")) or {}).get("accuracy")
        data: list[float | None] = []
        for model, _baseline in _GAIN_PAIRS:
            value = (table.get((dataset, model)) or {}).get("accuracy")
            if base is None or value is None:
                data.append(None)
            else:
                data.append(round((value - base) * 100.0, 6))
        if any(value is not None for value in data):
            series.append(
                {
                    "key": dataset,
                    "name": DATASET_LABELS[dataset],
                    "data": data,
                }
            )

    if not series:
        return None

    return {
        "id": chart_id,
        "title": title,
        "type": "groupedBar",
        "xName": "指标",
        "yName": "百分点",
        "unit": "value",
        "x": list(_GAIN_LABELS),
        "series": series,
        "showDataLabels": False,
    }


# --------------------------------------------------------------------------- 目录映射


def _charts_for_runs(
    run_root: Path,
    prefix: str,
    title_prefix: str,
    include_test_bars: bool = True,
) -> list[dict[str, Any]]:
    """白底训练目录：损失/准确率折线 + 测试准确率/损失柱状。

    ``include_test_bars=False`` 时只输出训练曲线（如 ``live-train-test`` 的
    测试柱状图应由目录内 ``metrics_*.csv`` 提供）。
    """
    charts: list[dict[str, Any]] = []

    loss = _line_chart(
        f"{prefix}_train_loss",
        f"{title_prefix}训练损失–轮次（A/B/C/D）",
        "train_loss",
        "交叉熵损失",
        "value",
        run_root,
    )
    if loss:
        charts.append(loss)

    acc = _line_chart(
        f"{prefix}_train_acc",
        f"{title_prefix}训练准确率–轮次（A/B/C/D）",
        "train_acc",
        "准确率",
        "percent",
        run_root,
    )
    if acc:
        charts.append(acc)

    if not include_test_bars:
        return charts

    metrics = _run_test_metrics(run_root)

    test_acc = _single_metric_bar(
        f"{prefix}_test_acc",
        f"{title_prefix}测试准确率（A/B/C/D）",
        metrics,
        "accuracy",
        "准确率",
        "percent",
    )
    if test_acc:
        charts.append(test_acc)

    test_loss = _single_metric_bar(
        f"{prefix}_test_loss",
        f"{title_prefix}测试 CE 损失（A/B/C/D）",
        metrics,
        "ce_loss",
        "交叉熵损失",
        "value",
    )
    if test_loss:
        charts.append(test_loss)

    return charts


def _charts_from_metric_files(
    pic_dir: Path,
    id_prefix: str,
    title_prefix: str,
) -> list[dict[str, Any]]:
    """实景测试目录：普通/困难帧 × 准确率/损失 四张单指标柱状图。"""
    charts: list[dict[str, Any]] = []

    specs = (
        ("metrics_test.csv", f"{id_prefix}_test_acc", "accuracy", "准确率", "percent",
         f"{title_prefix}普通实景测试准确率"),
        ("metrics_test.csv", f"{id_prefix}_test_loss", "ce_loss", "交叉熵损失", "value",
         f"{title_prefix}普通实景测试 CE 损失"),
        ("metrics_test_hard.csv", f"{id_prefix}_hard_acc", "accuracy", "准确率", "percent",
         f"{title_prefix}困难帧测试准确率"),
        ("metrics_test_hard.csv", f"{id_prefix}_hard_loss", "ce_loss", "交叉熵损失", "value",
         f"{title_prefix}困难帧测试 CE 损失"),
    )

    cache: dict[str, dict[str, dict[str, float]]] = {}
    for filename, chart_id, metric, y_name, unit, title in specs:
        if filename not in cache:
            cache[filename] = _metrics_from_csv(pic_dir / filename)
        chart = _single_metric_bar(
            chart_id, title, cache[filename], metric, y_name, unit
        )
        if chart:
            charts.append(chart)

    return charts


def _charts_for_melt(pic_dir: Path) -> list[dict[str, Any]]:
    """消融目录：主测试/困难帧的准确率、损失与模块增益。"""
    table = _ablation_table(pic_dir)
    charts: list[dict[str, Any]] = []

    main_acc = _grouped_bar(
        "ablation_main_acc",
        "主测试准确率",
        list(ABLATION_ORDER),
        MAIN_DATASETS,
        table,
        "accuracy",
        "准确率",
        "percent",
    )
    if main_acc:
        charts.append(main_acc)

    main_loss = _grouped_bar(
        "ablation_main_loss",
        "主测试 CE 损失",
        list(ABLATION_ORDER),
        MAIN_DATASETS,
        table,
        "ce_loss",
        "交叉熵损失",
        "value",
    )
    if main_loss:
        charts.append(main_loss)

    hard_acc = _grouped_bar(
        "ablation_hard_acc",
        "困难帧准确率",
        list(ABLATION_ORDER),
        HARD_DATASETS,
        table,
        "accuracy",
        "准确率",
        "percent",
    )
    if hard_acc:
        charts.append(hard_acc)

    hard_loss = _grouped_bar(
        "ablation_hard_loss",
        "困难帧 CE 损失",
        list(ABLATION_ORDER),
        HARD_DATASETS,
        table,
        "ce_loss",
        "交叉熵损失",
        "value",
    )
    if hard_loss:
        charts.append(hard_loss)

    main_gain = _gain_bar(
        "ablation_main_gain", "主测试模块增益（百分点）", MAIN_DATASETS, table
    )
    if main_gain:
        charts.append(main_gain)

    hard_gain = _gain_bar(
        "ablation_hard_gain", "困难帧模块增益（百分点）", HARD_DATASETS, table
    )
    if hard_gain:
        charts.append(hard_gain)

    return charts


# --------------------------------------------------------------------------- 公开入口


def build_charts(
    directory_name: str,
    pic_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """按 ``pic/`` 一级目录名生成 ECharts 图表规格列表。

    只对六个已知目录生成图表。降级粒度为**按图**：某张图缺少文件、CSV 为空、
    列缺失或数值非法时，仅跳过该图，目录内其他图照常返回；未知目录或目录内
    所有图都无法生成时返回 ``[]``。任何情况下都不抛异常、不补造数据。
    """
    if project_root is None:
        if pic_dir is None:
            _warn("build_charts 需要 pic_dir 或 project_root")
            return []
        project_root = Path(pic_dir).parent.parent
    root = Path(project_root)
    pic_path = Path(pic_dir) if pic_dir is not None else root / "pic" / directory_name

    try:
        if directory_name == "spec-train-test":
            return _charts_for_runs(root / "runs" / "spec5", "spec5", "")
        if directory_name == "spec-train-test-all":
            return _charts_for_runs(root / "runs" / "spec32", "spec32", "32 类白底")
        if directory_name == "spec-train-live-test":
            return _charts_from_metric_files(pic_path, "live5", "")
        if directory_name == "spec-train-live-test-all":
            return _charts_from_metric_files(pic_path, "live32", "32 类模型")
        if directory_name == "live-train-test":
            charts = _charts_for_runs(
                root / "runs" / "live5", "live5", "", include_test_bars=False
            )
            # 与 spec-train-live-test 的 live5_* 区分，保证图表 id 全局唯一
            charts.extend(_charts_from_metric_files(pic_path, "live5train", ""))
            return charts
        if directory_name == "melt-test":
            return _charts_for_melt(pic_path)
    except Exception as exc:  # 防御：任何图表数据异常都不能让 /api/gallery 500
        _warn(f"生成图表失败 {directory_name}: {exc}")
        return []

    return []
