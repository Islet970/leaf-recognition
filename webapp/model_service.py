"""模型注册与 CPU 推理服务（plans.md §5.2）。

职责：
- 扫描 runs/spec5、runs/spec32 与 runs/live5 下 A/B/C/D 的 best.pt（E 仅用于消融，不加载）；
- 校验 checkpoint 的 class_keys / flavia_labels / num_classes 与注册表一致；
- 读取同目录 test_metrics.json 的 accuracy / macro_f1 / ce_loss；
- 提供 list_models / predict / health 供 Flask 路由调用。

预处理必须与验证/测试一致：复用 src.data.build_eval_transform()。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

# 项目根目录（webapp/ 的上一级），用于定位 data/ 与 src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_eval_transform  # noqa: E402
from src.models import MODEL_NAMES, build_model  # noqa: E402

from .gradcam import compute_gradcam, encode_png_base64  # noqa: E402

# 网页只提供 A/B/C/D；E 仅用于消融（plans.md §5.1–§5.2）
WEB_MODEL_IDS = ("A", "B", "C", "D")

# 三个模型组：白底五类 / 白底 32 类 / 实景五类（plans.md §5.2、§14.8 T8.1）
# 字典顺序即 /api/models 的组顺序；前端直接显示 label，不在前端硬编码。
GROUP_SPECS: dict[str, dict[str, Any]] = {
    "spec5": {
        "label": "标本（5）",
        "run_dir": "spec5",
        "registry": "data/class_registry.json",
        "num_classes": 5,
    },
    "spec32": {
        "label": "标本（32）",
        "run_dir": "spec32",
        "registry": "data/class_registry_all32.json",
        "num_classes": 32,
    },
    "live5": {
        "label": "实景（5）",
        "run_dir": "live5",
        "registry": "data/class_registry.json",
        "num_classes": 5,
    },
}

# 全局推理锁：Flask 多线程下串行化 forward（plans.md §5.4）
_INFERENCE_LOCK = threading.Lock()


class ModelNotFoundError(KeyError):
    """未知模型 / 模型组。"""


class ModelUnavailableError(RuntimeError):
    """checkpoint 缺失或校验失败，模型不可用。"""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _center_crop_square(image: Image.Image) -> Image.Image:
    """中心裁正方形（与网页取景框一致）。"""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


class ModelService:
    """加载并缓存 A/B/C/D × spec5/spec32/live5，共 12 个组内模型。"""

    def __init__(
        self,
        checkpoints_dir: str | Path = "runs",
        project_root: str | Path | None = None,
    ) -> None:
        torch.set_num_threads(8)
        self.project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        checkpoints = Path(checkpoints_dir)
        if not checkpoints.is_absolute():
            checkpoints = self.project_root / checkpoints
        self.checkpoints_dir = checkpoints

        self._transform = build_eval_transform()
        self._lock = _INFERENCE_LOCK

        # group_id -> {"id","label","num_classes","classes": {key: info}, "order": [key]}
        self._groups: dict[str, dict[str, Any]] = {}
        # (group, model_id) -> {"model", "class_keys", "num_classes", "model_name"}
        self._loaded: dict[tuple[str, str], dict[str, Any]] = {}
        # 对外元数据条目（含 unavailable 项）
        self._entries: list[dict[str, Any]] = []

        self._initialize()

    # ------------------------------------------------------------------ 初始化
    def _initialize(self) -> None:
        for group_id, spec in GROUP_SPECS.items():
            registry_path = self.project_root / spec["registry"]
            classes: list[dict[str, Any]] = []
            registry_error: str | None = None
            try:
                registry = _read_json(registry_path)
                classes = sorted(registry["classes"], key=lambda c: int(c["py_label"]))
                if not classes:
                    raise ValueError("注册表 classes 为空")
            except Exception as exc:  # 注册表缺失/损坏时不崩溃
                registry_error = f"注册表读取失败 {registry_path}: {exc}"
                classes = []

            class_info = {
                str(c["key"]): {
                    "key": str(c["key"]),
                    "zh_name": c.get("zh_name", ""),
                    "scientific_name": c.get("scientific", ""),
                }
                for c in classes
            }
            expected_keys = [str(c["key"]) for c in classes]
            expected_flavia = [int(c["flavia_label"]) for c in classes]
            expected_num = len(classes) if classes else int(spec["num_classes"])

            self._groups[group_id] = {
                "id": group_id,
                "label": spec["label"],
                "num_classes": expected_num,
                "classes": class_info,
                "order": expected_keys,
            }

            for model_id in WEB_MODEL_IDS:
                self._entries.append(
                    self._load_entry(
                        group_id=group_id,
                        model_id=model_id,
                        expected_keys=expected_keys,
                        expected_flavia=expected_flavia,
                        expected_num=expected_num,
                        registry_error=registry_error,
                    )
                )

    def _load_entry(
        self,
        group_id: str,
        model_id: str,
        expected_keys: list[str],
        expected_flavia: list[int],
        expected_num: int,
        registry_error: str | None,
    ) -> dict[str, Any]:
        group_spec = GROUP_SPECS[group_id]
        model_name = MODEL_NAMES[model_id]
        run_dir = self.checkpoints_dir / group_spec["run_dir"] / f"{model_name}_seed42"
        checkpoint_path = run_dir / "best.pt"

        entry: dict[str, Any] = {
            "id": model_id,
            "name": model_name,
            "group": group_id,
            "group_label": group_spec["label"],
            "num_classes": expected_num,
            "accuracy": None,
            "macro_f1": None,
            "ce_loss": None,
            "param_count": None,
            "available": False,
            "error": None,
        }

        # 指标与 checkpoint 是否可用无关，先尽量读取
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.is_file():
            try:
                metrics = _read_json(metrics_path)
                for key in ("accuracy", "macro_f1", "ce_loss"):
                    value = metrics.get(key)
                    if value is not None:
                        entry[key] = round(float(value), 6)
            except Exception as exc:
                entry["error"] = f"test_metrics.json 解析失败: {exc}"

        error = registry_error
        if error is None and not checkpoint_path.is_file():
            error = f"checkpoint 缺失: {checkpoint_path}"
        if error is not None:
            entry["error"] = error
            return entry

        try:
            # weights_only=True 与现有 checkpoint 不兼容：config 中含
            # torch.torch_version.TorchVersion（非默认安全全局），且 checkpoint
            # 由本项目训练脚本生成、来源可信，故保留 weights_only=False。
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            class_keys = [str(k) for k in checkpoint.get("class_keys", [])]
            flavia_labels = [int(v) for v in checkpoint.get("flavia_labels", [])]
            num_classes = int(checkpoint.get("num_classes", 0))

            if num_classes != expected_num:
                raise ValueError(
                    f"num_classes 不一致: checkpoint={num_classes}, registry={expected_num}"
                )
            if class_keys != expected_keys:
                raise ValueError(
                    f"class_keys 与注册表不一致: checkpoint={class_keys}, registry={expected_keys}"
                )
            if flavia_labels != expected_flavia:
                raise ValueError(
                    f"flavia_labels 与注册表不一致: checkpoint={flavia_labels}, "
                    f"registry={expected_flavia}"
                )
            if "model_state" not in checkpoint:
                raise ValueError("checkpoint 缺少 model_state")

            model = build_model(model_id, expected_num, pretrained=False)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            entry["param_count"] = int(sum(p.numel() for p in model.parameters()))

            self._loaded[(group_id, model_id)] = {
                "model": model,
                "model_name": model_name,
                "class_keys": class_keys,
                "num_classes": expected_num,
            }
            entry["available"] = True
            entry["error"] = None
        except Exception as exc:  # 单个模型失败不影响其他模型
            entry["available"] = False
            entry["error"] = f"checkpoint 加载失败: {exc}"
            self._loaded.pop((group_id, model_id), None)
            print(f"[ModelService] {group_id}/{model_id} 不可用: {exc}", file=sys.stderr)

        return entry

    # ------------------------------------------------------------------ 查询
    def list_models(self) -> dict[str, Any]:
        groups = [
            {
                "id": g["id"],
                "label": g["label"],
                "num_classes": g["num_classes"],
            }
            for g in self._groups.values()
        ]
        return {
            "groups": groups,
            "models": [dict(entry) for entry in self._entries],
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "device": "cpu",
            "num_models": len(self._loaded),
            "torch": torch.__version__,
        }

    def _get_entry(self, group: str, model_id: str) -> dict[str, Any]:
        if group not in self._groups:
            raise ModelNotFoundError(f"未知模型组 {group!r}，可选: {list(self._groups)}")
        if model_id not in MODEL_NAMES:
            raise ModelNotFoundError(f"未知模型 {model_id!r}，可选: {list(MODEL_NAMES)}")
        if model_id not in WEB_MODEL_IDS:
            raise ModelNotFoundError(f"模型 {model_id} 仅用于消融，网页不提供")
        for entry in self._entries:
            if entry["group"] == group and entry["id"] == model_id:
                return entry
        raise ModelNotFoundError(f"未找到模型 {group}/{model_id}")

    # ------------------------------------------------------------------ 推理
    def predict(
        self,
        group: str,
        model_id: str,
        image: Image.Image,
        topk: int = 3,
        gradcam: bool = False,
    ) -> dict[str, Any]:
        """对单张 PIL 图片推理，返回 top-k 类别、耗时与可选 Grad-CAM。

        ``gradcam=False`` 走 ``torch.no_grad()`` 快路径，行为与之前一致；
        ``gradcam=True`` 时在同一把全局推理锁内调用 :mod:`webapp.gradcam`，
        只做一次带梯度的 forward/backward，并复用该 logits 计算 softmax/top-k。
        成功时返回 ``gradcam_available=True`` 与非空 ``gradcam_png_base64``；
        生成失败时降级为普通推理（不抛 500），返回
        ``gradcam_available=False``、``gradcam_png_base64=None``，并附
        ``gradcam_error``。
        """
        started = time.perf_counter()
        entry = self._get_entry(group, model_id)
        if not entry["available"]:
            raise ModelUnavailableError(
                entry["error"] or f"模型 {group}/{model_id} 不可用"
            )

        loaded = self._loaded[(group, model_id)]
        model = loaded["model"]
        class_keys: list[str] = loaded["class_keys"]
        num_classes: int = loaded["num_classes"]
        class_info = self._groups[group]["classes"]

        # 与训练/测试一致的预处理：EXIF 校正 -> RGB -> 中心裁正方形 -> eval transform
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = _center_crop_square(image)
        tensor = self._transform(image).unsqueeze(0)

        gradcam_available = False
        gradcam_png_base64: str | None = None
        gradcam_error: str | None = None

        with self._lock:
            if gradcam:
                try:
                    logits, overlay = compute_gradcam(model, tensor, image)
                except Exception as exc:  # Grad-CAM 失败时降级为普通推理
                    print(
                        f"[ModelService] {group}/{model_id} Grad-CAM 生成失败: {exc}",
                        file=sys.stderr,
                    )
                    gradcam_error = "Grad-CAM 生成失败"
                    with torch.no_grad():
                        logits = model(tensor)
                else:
                    try:
                        gradcam_png_base64 = encode_png_base64(overlay)
                        gradcam_available = bool(gradcam_png_base64)
                    except Exception as exc:  # 已拿到 logits，不再重复 forward
                        print(
                            f"[ModelService] {group}/{model_id} Grad-CAM 编码失败: {exc}",
                            file=sys.stderr,
                        )
                        gradcam_error = "Grad-CAM 图像编码失败"
                probs = torch.softmax(logits, dim=1)[0]
            else:
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]

        k = min(max(int(topk), 1), num_classes)
        values, indices = torch.topk(probs, k)
        topk_items = []
        for prob, index in zip(values.tolist(), indices.tolist()):
            key = class_keys[index]
            info = class_info.get(key, {})
            topk_items.append(
                {
                    "key": key,
                    "zh_name": info.get("zh_name", ""),
                    "scientific_name": info.get("scientific_name", ""),
                    "prob": round(float(prob), 6),
                }
            )

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        result = {
            "model": model_id,
            "model_name": loaded["model_name"],
            "group": group,
            "group_label": self._groups[group]["label"],
            "num_classes": num_classes,
            "topk": topk_items,
            "latency_ms": latency_ms,
            "gradcam_png_base64": gradcam_png_base64,
            "gradcam_available": gradcam_available,
        }
        if gradcam_error is not None:
            result["gradcam_error"] = gradcam_error
        return result
