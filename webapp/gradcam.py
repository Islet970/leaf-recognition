"""Grad-CAM 热力图生成（plans.md §5.4、§14.8 T8.4）。

调用方式（必须由 ``ModelService.predict`` 在同一把全局推理锁内调用）::

    logits, overlay = compute_gradcam(model, tensor, base_image)
    png_base64 = encode_png_base64(overlay)

实现要点：
- 目标层由 ``model.get_gradcam_layer()`` 提供：A 为 ``features[12][0]``
  （576×4×4），B 为 ``features[18][0]``（1280×4×4），C/D 为 ``msff.out_conv``
  （576×8×8）；
- forward hook 捕获目标层激活，并在激活张量上注册 backward hook 捕获梯度；
- 目标类别取 top-1 logit，``backward()`` 后按 channel 平均梯度加权、ReLU、
  归一化到 0–1（全 0 时保持全 0，不除零）；
- 上采样到限定尺寸（长边 ≤ 768 px、短边 ≥ 128 px，保持长宽比），用 Pillow
  伪彩色（黑→黄→红）生成热力图，再与原图按 ``BLEND_ALPHA`` 透明度叠加；
- 所有 hook 与参数梯度都在 ``finally`` 中清理，调用结束后参数上不残留梯度。
"""

from __future__ import annotations

import base64
import io
from typing import Tuple

import torch
from PIL import Image, ImageOps

# 热力图与原图的叠加透明度（plans.md §5.4：约 0.45）
BLEND_ALPHA = 0.45

# 输出叠加图尺寸上限/下限：CAM 源仅 4×4 / 8×8，超过 768 px 不会增加信息，
# 但会显著放大 base64 体积与锁内编码耗时（手机 4000×3000 图约 3.3 MB+）。
MAX_OUTPUT_SIDE = 768
MIN_OUTPUT_SIDE = 128


def _output_size(size: Tuple[int, int]) -> Tuple[int, int]:
    """按长边 ≤ MAX_OUTPUT_SIDE、短边 ≥ MIN_OUTPUT_SIDE 计算输出尺寸。

    先等比缩放使长边不超过上限，再在短边不足 128 时等比放大到下限。
    """
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"无效图片尺寸: {size}")
    scale = min(1.0, MAX_OUTPUT_SIDE / max(width, height))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    short = min(out_w, out_h)
    if short < MIN_OUTPUT_SIDE:
        upscale = MIN_OUTPUT_SIDE / short
        out_w = max(MIN_OUTPUT_SIDE, int(round(out_w * upscale)))
        out_h = max(MIN_OUTPUT_SIDE, int(round(out_h * upscale)))
    return out_w, out_h


def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    """把 CAM 线性归一化到 0–1；全 0 时保持全 0（不除零）。"""
    cam = torch.relu(cam)
    peak = float(cam.max().item())
    if peak > 0.0:
        cam = cam / peak
    else:
        cam = torch.zeros_like(cam)
    return cam.clamp(0.0, 1.0)


def _colorize_cam(cam: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    """把 [H, W] 的 0–1 CAM 上采样并转成黑→黄→红伪彩色图。"""
    cam_uint8 = (cam * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
    gray = Image.fromarray(cam_uint8.cpu().numpy(), mode="L")
    gray = gray.resize(size, Image.Resampling.BILINEAR)
    # 0 -> 黑，0.5 -> 黄，1 -> 红（不引入 matplotlib）
    return ImageOps.colorize(
        gray, black=(0, 0, 0), white=(255, 0, 0), mid=(255, 255, 0)
    )


def compute_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    base_image: Image.Image,
) -> Tuple[torch.Tensor, Image.Image]:
    """对单张输入做一次带梯度的 forward/backward，返回 logits 与叠加图。

    Args:
        model: 已加载权重的模型（A/B/C/D），需提供 ``get_gradcam_layer()``。
        tensor: 预处理后的输入张量，形状 ``[1, 3, H, W]``。
        base_image: 与模型输入对应的原图（EXIF 校正、RGB、中心裁正方形后）。

    Returns:
        ``(logits_detached, overlay)``：logits 形状 ``[1, num_classes]``，
        已 detach（在 CPU 上）；overlay 为 RGB PIL 图，按 ``base_image`` 等比
        缩放到长边 ≤ 768、短边 ≥ 128（CAM 源分辨率低，缩小不损失信息）。

    Raises:
        RuntimeError: 未捕获到目标层激活或梯度。
    """
    layer = model.get_gradcam_layer()
    captured: dict[str, torch.Tensor] = {}
    tensor_hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _forward_hook(_module, _inputs, output):
        captured["activation"] = output
        tensor_hooks.append(
            output.register_hook(
                lambda grad: captured.__setitem__("gradient", grad)
            )
        )

    was_training = model.training
    model.eval()
    forward_handle = layer.register_forward_hook(_forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor)
            target = int(torch.argmax(logits, dim=1).item())
            model.zero_grad(set_to_none=True)
            logits[0, target].backward()

        activation = captured.get("activation")
        gradient = captured.get("gradient")
        if activation is None or gradient is None:
            raise RuntimeError("未捕获到 Grad-CAM 层激活或梯度")

        # Grad-CAM：channel 平均梯度作为权重，加权求和后 ReLU
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1)[0]
        cam = _normalize_cam(cam.detach().float())

        # 限制输出尺寸：CAM 源仅 4×4/8×8，放大到原图全分辨率只增加体积。
        size = _output_size(base_image.size)
        base_rgb = base_image.convert("RGB")
        if base_rgb.size != size:
            base_rgb = base_rgb.resize(size, Image.Resampling.LANCZOS)

        overlay = Image.blend(
            base_rgb,
            _colorize_cam(cam, size),
            BLEND_ALPHA,
        )
        return logits.detach().cpu(), overlay
    finally:
        forward_handle.remove()
        for handle in tensor_hooks:
            handle.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


def encode_png_base64(image: Image.Image) -> str:
    """把 PIL 图编码为 PNG，返回不带 ``data:`` 前缀的纯 base64 字符串。"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    base_image: Image.Image,
) -> Tuple[torch.Tensor, str]:
    """便捷入口：返回 ``(logits_detached, png_base64)``。"""
    logits, overlay = compute_gradcam(model, tensor, base_image)
    return logits, encode_png_base64(overlay)
