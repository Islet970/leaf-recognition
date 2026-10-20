"""Flask 入口与 JSON API（plans.md §5.2、§5.4）。

启动：
    python -m webapp.app --checkpoints runs --host 127.0.0.1 --port 5000
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import sys
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from .gallery_service import GalleryService
from .model_service import (
    GROUP_SPECS,
    MODEL_NAMES,
    PROJECT_ROOT,
    WEB_MODEL_IDS,
    ModelNotFoundError,
    ModelService,
    ModelUnavailableError,
)

MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB（plans.md §5.3）

# 图片解码上限：防止压缩炸弹（边长 10000、总计 4000 万像素）
MAX_IMAGE_SIDE = 10_000
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}

# /pic/ 白名单扩展名（plans.md §5.7）
PIC_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".csv", ".docx", ".pdf"}


# ---------------------------------------------------------------------- 工具
def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _validate_image(image: Image.Image) -> None:
    """校验图片格式与尺寸；不通过时 400。"""
    if image.format not in ALLOWED_IMAGE_FORMATS:
        abort(400, description="图片格式不支持：仅支持 JPG/PNG")
    width, height = image.size
    if (
        width <= 0
        or height <= 0
        or width > MAX_IMAGE_SIDE
        or height > MAX_IMAGE_SIDE
        or width * height > MAX_IMAGE_PIXELS
    ):
        abort(
            400,
            description=(
                f"图片尺寸超出限制：边长需 ≤ {MAX_IMAGE_SIDE}，"
                f"总像素需 ≤ {MAX_IMAGE_PIXELS}"
            ),
        )


def _open_upload(file_storage) -> Image.Image:
    """从 multipart 文件读取 PIL 图片；失败时 400。"""
    try:
        data = file_storage.read()
        if not data:
            raise ValueError("空文件")
        image = Image.open(io.BytesIO(data))
        _validate_image(image)
        image.load()
        return image
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        abort(400, description="图片解析失败：请上传 JPG/PNG 图片")


def _open_bytes(data: bytes) -> Image.Image:
    try:
        if not data:
            raise ValueError("空数据")
        image = Image.open(io.BytesIO(data))
        _validate_image(image)
        image.load()
        return image
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        abort(400, description="图片解析失败：请上传 JPG/PNG 图片")


def _decode_data_url(image_data_url: str) -> Image.Image:
    """解析 ``data:image/...;base64,<payload>``（也接受裸 base64）。"""
    if not isinstance(image_data_url, str) or not image_data_url.strip():
        abort(400, description="缺少 image_data_url")
    payload = image_data_url.strip()
    if payload.startswith("data:"):
        if "," not in payload:
            abort(400, description="image_data_url 格式错误：缺少 base64 数据")
        _, payload = payload.split(",", 1)
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        abort(400, description="image_data_url 格式错误：base64 解码失败")
    return _open_bytes(raw)


def _validate_group(group: str) -> str:
    if group not in GROUP_SPECS:
        abort(400, description=f"未知模型组 {group!r}，可选: {list(GROUP_SPECS)}")
    return group


def _validate_model(model_id: str) -> str:
    if model_id not in MODEL_NAMES:
        abort(404, description=f"未知模型 {model_id!r}")
    if model_id not in WEB_MODEL_IDS:
        abort(404, description=f"模型 {model_id} 仅用于消融，网页不提供")
    return model_id


def _parse_topk(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        abort(400, description="topk 必须是整数")
    if value < 1:
        abort(400, description="topk 必须大于 0")
    return value


# ---------------------------------------------------------------------- 应用
def create_app(
    checkpoints_dir: str | Path = "runs",
    project_root: str | Path | None = None,
) -> Flask:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.json.ensure_ascii = False

    model_service = ModelService(checkpoints_dir=checkpoints_dir, project_root=root)
    gallery_service = GalleryService(project_root=root)
    app.extensions["model_service"] = model_service
    app.extensions["gallery_service"] = gallery_service

    # ------------------------------------------------------------- 错误处理
    @app.errorhandler(400)
    def handle_400(err):
        return jsonify({"error": getattr(err, "description", "请求参数错误")}), 400

    @app.errorhandler(404)
    def handle_404(err):
        return jsonify({"error": getattr(err, "description", "资源不存在")}), 404

    @app.errorhandler(413)
    def handle_413(_err):
        return jsonify({"error": "上传内容超过 10 MB 限制"}), 413

    @app.errorhandler(503)
    def handle_503(err):
        return jsonify({"error": getattr(err, "description", "模型服务暂不可用")}), 503

    @app.errorhandler(500)
    def handle_500(err):
        # 服务端记录完整异常，响应只给通用信息，避免泄露内部细节
        detail = getattr(err, "original_exception", None)
        app.logger.error("未处理异常: %r", detail, exc_info=True)
        return jsonify({"error": "服务器内部错误"}), 500

    # ------------------------------------------------------------- 页面
    @app.get("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------- 元数据
    @app.get("/api/health")
    def api_health():
        return jsonify(model_service.health())

    @app.get("/api/models")
    def api_models():
        return jsonify(model_service.list_models())

    @app.get("/api/gallery")
    def api_gallery():
        return jsonify(gallery_service.list_gallery())

    # ------------------------------------------------------------- 识别
    @app.post("/api/predict")
    def api_predict():
        file_storage = request.files.get("image")
        if file_storage is None:
            abort(400, description="缺少 image 文件字段")

        group = _validate_group(request.form.get("group", "spec5"))
        model_id = _validate_model(request.form.get("model", "A"))
        topk = _parse_topk(request.form.get("topk", "3"), default=3)
        gradcam = _parse_bool(request.form.get("gradcam", "0"))

        image = _open_upload(file_storage)
        try:
            result = model_service.predict(
                group, model_id, image, topk=topk, gradcam=gradcam
            )
        except ModelNotFoundError as exc:
            abort(404, description=str(exc))
        except ModelUnavailableError as exc:
            abort(503, description=str(exc))
        return jsonify(result)

    @app.post("/api/predict_frame")
    def api_predict_frame():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="请求体必须是 JSON")

        group = _validate_group(payload.get("group", "spec5"))
        model_id = _validate_model(payload.get("model", "A"))
        image = _decode_data_url(
            payload.get("image_data_url") or payload.get("image") or ""
        )
        gradcam = _parse_bool(payload.get("gradcam", False))
        topk = _parse_topk(payload.get("topk", 3), default=3)

        try:
            result = model_service.predict(
                group, model_id, image, topk=topk, gradcam=gradcam
            )
        except ModelNotFoundError as exc:
            abort(404, description=str(exc))
        except ModelUnavailableError as exc:
            abort(503, description=str(exc))
        return jsonify(result)

    @app.post("/api/predict_all")
    def api_predict_all():
        file_storage = request.files.get("image")
        if file_storage is None:
            abort(400, description="缺少 image 文件字段")

        group = _validate_group(request.form.get("group", "spec5"))
        default_topk = 5 if group == "spec32" else 3
        topk = _parse_topk(request.form.get("topk", default_topk), default=default_topk)

        image = _open_upload(file_storage)
        results: list[dict[str, Any]] = []
        for model_id in WEB_MODEL_IDS:
            try:
                result = model_service.predict(group, model_id, image, topk=topk)
                result["available"] = True
                result["error"] = None
                results.append(result)
            except (ModelUnavailableError, ModelNotFoundError) as exc:
                results.append(
                    {
                        "model": model_id,
                        "group": group,
                        "available": False,
                        "error": str(exc),
                    }
                )

        return jsonify(
            {
                "group": group,
                "group_label": GROUP_SPECS[group]["label"],
                "num_classes": GROUP_SPECS[group]["num_classes"],
                "results": results,
            }
        )

    # ------------------------------------------------------------- 演示按钮
    @app.post("/api/demo/add-data")
    def api_demo_add_data():
        return jsonify({"demo": True, "message": "演示功能：未接入真实数据接口"})

    @app.post("/api/demo/train")
    def api_demo_train():
        return jsonify({"demo": True, "message": "演示功能：未启动真实训练"})

    # ------------------------------------------------------------- pic 静态文件
    @app.get("/pic/<path:filename>")
    def pic_file(filename: str):
        suffix = Path(filename).suffix.lower()
        if suffix not in PIC_ALLOWED_SUFFIXES:
            abort(404, description="不支持的文件类型")

        root_dir = gallery_service.pic_root.resolve()
        try:
            target = (root_dir / filename).resolve()
            target.relative_to(root_dir)  # 阻止 ../ 目录穿越
        except (ValueError, OSError):
            abort(404, description="非法路径")

        if not target.is_file():
            abort(404, description="文件不存在")
        if target.stat().st_size > MAX_CONTENT_LENGTH:
            abort(413)
        return send_from_directory(root_dir, filename)

    return app


# ---------------------------------------------------------------------- 入口
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="知叶叶片识别 Web 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=5000, help="监听端口（默认 5000）")
    parser.add_argument(
        "--checkpoints", default="runs", help="checkpoint 根目录（默认 runs）"
    )
    args = parser.parse_args(argv)

    app = create_app(checkpoints_dir=args.checkpoints)
    # 不启用 debug，不自动打开浏览器
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
