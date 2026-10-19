"""五个叶部病害分类模型（plans.md §2.1–§2.4）。

模型一览（128×128 输入，ImageNet 预训练）：
    A  A_mnv3s             MobileNetV3-Small 基线
    B  B_mnv2              MobileNetV2 对比模型
    C  C_leafnet_msff      A + MSFF（多尺度特征融合）
    D  D_leafnet_msff_cbam A + MSFF + CBAM
    E  E_mnv3s_cbam        A + CBAM（消融）

公共接口：
    MODEL_IDS, MODEL_NAMES
    build_model(model_id, num_classes, pretrained=True) -> nn.Module
每个模型都有 get_gradcam_layer() -> nn.Conv2d，forward 返回 [B, num_classes] logits。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v2,
    mobilenet_v3_small,
)

MODEL_IDS = ["A", "B", "C", "D", "E"]

MODEL_NAMES = {
    "A": "A_mnv3s",
    "B": "B_mnv2",
    "C": "C_leafnet_msff",
    "D": "D_leafnet_msff_cbam",
    "E": "E_mnv3s_cbam",
}

# MobileNetV3-Small features 的分段点（plans.md §2.1，已按真实模型核对）：
#   features[:9]  -> 48×8×8
#   features[9:]  -> 576×4×4
_SPLIT_INDEX = 9
_V3S_FEATURE_CHANNELS = 576


class MSFF(nn.Module):
    """M1：轻量多尺度特征融合（plans.md §2.1）。

    输入：
        low  [B, 48, 8, 8]   （features[:9]）
        high [B, 576, 4, 4]  （features[9:]）
    输出：
        [B, 576, 8, 8]
    """

    def __init__(
        self,
        low_channels: int = 48,
        high_channels: int = _V3S_FEATURE_CHANNELS,
        hidden_channels: int = 96,
        out_channels: int = _V3S_FEATURE_CHANNELS,
    ) -> None:
        super().__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardswish(inplace=True),
        )
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardswish(inplace=True),
        )
        # concat 192×8×8 -> 3×3 Conv-BN-Hardswish（192→96）-> 1×1 Conv-BN（96→576）
        self.fuse_conv = nn.Conv2d(
            2 * hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.fuse_bn = nn.BatchNorm2d(hidden_channels)
        self.fuse_act = nn.Hardswish(inplace=True)
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        low_feat = self.low_conv(low)
        high_feat = self.high_conv(high)
        high_feat = F.interpolate(
            high_feat, size=low_feat.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = torch.cat([low_feat, high_feat], dim=1)
        fused = self.fuse_act(self.fuse_bn(self.fuse_conv(fused)))
        fused = self.out_bn(self.out_conv(fused))
        # 残差：保留预训练高层语义，同时注入 8×8 的叶缘/叶脉细节
        residual = F.interpolate(
            high, size=fused.shape[-2:], mode="bilinear", align_corners=False
        )
        return fused + residual


class ChannelAttention(nn.Module):
    """CBAM 通道分支：avg-pool 与 max-pool 共用 576→36→576 的 MLP。"""

    def __init__(self, channels: int = _V3S_FEATURE_CHANNELS, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)  # 576 → 36
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 空间分支：channel-wise avg/max -> concat 2×H×W -> 7×7 Conv -> sigmoid。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """M2：通道-空间注意力（plans.md §2.2）。"""

    def __init__(self, channels: int = _V3S_FEATURE_CHANNELS, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        return self.spatial_attention(x)


def _build_classifier(num_classes: int) -> nn.Sequential:
    """C/D/E 分类头：GAP -> 576 -> FC 1024 -> Hardswish -> Dropout 0.2 -> FC(num_classes)。"""
    return nn.Sequential(
        nn.Linear(_V3S_FEATURE_CHANNELS, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes),
    )


class MobileNetV3SmallBaseline(nn.Module):
    """模型 A：torchvision MobileNetV3-Small，保留默认分类头，仅替换 classifier[3]。"""

    def __init__(self, num_classes: int, backbone: nn.Module) -> None:
        super().__init__()
        # 复用 torchvision 模块，使 state_dict 前缀保持 features.* / classifier.*
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.classifier[3] = nn.Linear(self.classifier[3].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_gradcam_layer(self) -> nn.Conv2d:
        return self.features[12][0]


class MobileNetV2Baseline(nn.Module):
    """模型 B：torchvision MobileNetV2，保留默认分类头，仅替换 classifier[1]。"""

    def __init__(self, num_classes: int, backbone: nn.Module) -> None:
        super().__init__()
        self.features = backbone.features
        self.classifier = backbone.classifier
        self.classifier[1] = nn.Linear(self.classifier[1].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.classifier(x)

    def get_gradcam_layer(self) -> nn.Conv2d:
        return self.features[18][0]


class LeafNetV3(nn.Module):
    """C/D/E 共用母体：MobileNetV3-Small features + 可选 MSFF / CBAM + 统一分类头。"""

    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        use_msff: bool = False,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        if not use_msff and not use_cbam:
            raise ValueError("LeafNetV3 至少需要 MSFF 或 CBAM 之一；基线请用模型 A。")

        # 保留 features 前缀：strict=False 加载时 missing 只出现在新增模块与分类头
        self.features = backbone.features
        self.use_msff = use_msff
        self.use_cbam = use_cbam
        self.msff = MSFF() if use_msff else None
        self.cbam = CBAM(_V3S_FEATURE_CHANNELS) if use_cbam else None

        # 复用预训练的第一个 FC（576→1024），只重新初始化最终分类层
        self.classifier = _build_classifier(num_classes)
        self.classifier[0] = backbone.classifier[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.features[:_SPLIT_INDEX](x)  # [B, 48, 8, 8]
        high = self.features[_SPLIT_INDEX:](low)  # [B, 576, 4, 4]
        if self.use_msff:
            feat = self.msff(low, high)  # [B, 576, 8, 8]
        else:
            feat = high
        if self.use_cbam:
            feat = self.cbam(feat)
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)

    def get_gradcam_layer(self) -> nn.Conv2d:
        if self.use_msff:
            return self.msff.out_conv
        return self.features[12][0]


def _load_pretrained(
    model: nn.Module,
    source_state_dict: dict,
    final_head_keys: set[str],
    allowed_missing_prefixes: tuple[str, ...],
) -> dict:
    """把原始 torchvision 权重加载进模型（strict=False），并校验 missing/unexpected。

    只丢弃最终分类层（形状随 num_classes 变化）的权重。
    """
    filtered = {k: v for k, v in source_state_dict.items() if k not in final_head_keys}
    result = model.load_state_dict(filtered, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"预训练权重出现 unexpected keys: {unexpected}")
    bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
    if bad_missing:
        raise RuntimeError(f"预训练权重出现非预期 missing keys: {bad_missing}")
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _v3s_weights(pretrained: bool):
    return MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None


def _v2_weights(pretrained: bool):
    return MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None


def build_model(model_id: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """按模型编号构建模型。

    Args:
        model_id: "A" / "B" / "C" / "D" / "E"
        num_classes: 分类数（当前 5，后续 32）
        pretrained: 是否加载 ImageNet 预训练权重
    """
    model_id = model_id.upper()
    if model_id not in MODEL_IDS:
        raise ValueError(f"未知模型编号 {model_id!r}，可选: {MODEL_IDS}")
    if num_classes < 1:
        raise ValueError(f"num_classes 必须为正整数，收到 {num_classes}")

    if model_id == "A":
        backbone = mobilenet_v3_small(weights=_v3s_weights(pretrained))
        source_sd = backbone.state_dict()
        model = MobileNetV3SmallBaseline(num_classes, backbone)
        report = _load_pretrained(
            model,
            source_sd,
            final_head_keys={"classifier.3.weight", "classifier.3.bias"},
            allowed_missing_prefixes=("classifier.3.",),
        )

    elif model_id == "B":
        backbone = mobilenet_v2(weights=_v2_weights(pretrained))
        source_sd = backbone.state_dict()
        model = MobileNetV2Baseline(num_classes, backbone)
        report = _load_pretrained(
            model,
            source_sd,
            final_head_keys={"classifier.1.weight", "classifier.1.bias"},
            allowed_missing_prefixes=("classifier.1.",),
        )

    else:
        backbone = mobilenet_v3_small(weights=_v3s_weights(pretrained))
        source_sd = backbone.state_dict()
        use_msff = model_id in ("C", "D")
        use_cbam = model_id in ("D", "E")
        model = LeafNetV3(num_classes, backbone, use_msff=use_msff, use_cbam=use_cbam)
        allowed = ["classifier.3."]
        if use_msff:
            allowed.append("msff.")
        if use_cbam:
            allowed.append("cbam.")
        report = _load_pretrained(
            model,
            source_sd,
            final_head_keys={"classifier.3.weight", "classifier.3.bias"},
            allowed_missing_prefixes=tuple(allowed),
        )

    if not pretrained:
        # 未使用预训练时不做权重加载，missing/unexpected 无意义
        report = {"missing_keys": [], "unexpected_keys": []}

    model.model_id = model_id
    model.model_name = MODEL_NAMES[model_id]
    model.num_classes = num_classes
    model.pretrained = pretrained
    model.pretrained_report = report
    return model


def _self_test(num_classes: int = 5) -> None:
    """验收自测：forward 形状、参数量、预训练权重、Grad-CAM 层形状。"""
    torch.manual_seed(42)
    x = torch.randn(2, 3, 128, 128)
    expected_gradcam = {
        "A": (576, 4, 4),
        "B": (1280, 4, 4),
        "C": (576, 8, 8),
        "D": (576, 8, 8),
        "E": (576, 4, 4),
    }
    # 新建一份 torchvision 预训练模型作为权重等价性对照。
    reference_v3s = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    reference_v2 = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    print(f"输入: {tuple(x.shape)}, num_classes={num_classes}")
    for model_id in MODEL_IDS:
        model = build_model(model_id, num_classes, pretrained=True)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        params = sum(p.numel() for p in model.parameters())

        if model_id == "B":
            layer_name = "features[18][0].weight"
            model_weight = model.features[18][0].weight
            reference_weight = reference_v2.features[18][0].weight
        else:
            layer_name = "features[12][0].weight"
            model_weight = model.features[12][0].weight
            reference_weight = reference_v3s.features[12][0].weight
        weights_equal = torch.equal(
            model_weight.detach().cpu(), reference_weight.detach().cpu()
        )

        layer = model.get_gradcam_layer()
        captured = {}
        hook = layer.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("shape", tuple(out.shape))
        )
        with torch.no_grad():
            model(x)
        hook.remove()
        cam_shape = tuple(captured["shape"][1:])

        report = model.pretrained_report
        missing = report["missing_keys"]
        unexpected = report["unexpected_keys"]

        print()
        print(f"[{model_id}] {model.model_name}")
        print(f"  forward: {tuple(logits.shape)}  参数量: {params:,} ({params / 1e6:.3f} M)")
        print(f"  Grad-CAM 层: {type(layer).__name__}, 输出: {cam_shape}")
        print(f"  pretrained: missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"    missing_keys: {missing}")
        print(f"    unexpected_keys: {unexpected}")
        print(f"  预训练权重抽查 {layer_name}: torch.equal={weights_equal}")

        assert tuple(logits.shape) == (2, num_classes), logits.shape
        assert not unexpected, unexpected
        assert cam_shape == expected_gradcam[model_id], (cam_shape, expected_gradcam[model_id])
        assert weights_equal, (
            f"[{model_id}] {layer_name} 与新建 torchvision 预训练模型不一致"
        )

    print("\n全部断言通过。")


if __name__ == "__main__":
    _self_test()
