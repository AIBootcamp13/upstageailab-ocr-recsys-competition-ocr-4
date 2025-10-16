"""FPNC 디코더 구현.

DBNet++ 논문(Real-Time Scene Text Detection with Differentiable Binarization
and Adaptive Scale Fusion)을 참고해 Adaptive Scale Fusion(ASF)이 포함된
Feature Pyramid Network in Cascaded form을 단순화해 구현하였다. mmocr의
구현을 PyTorch 기본 구성요소로 재작성해 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    """Kaiming Normal 초기화 + BatchNorm 기본값."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class ConvBNAct(nn.Module):
    """Conv -> (BatchNorm) -> (Activation) 시퀀스."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = False,
        use_bn: bool = False,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            activation = activation.lower()
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        self.block = nn.Sequential(*layers)
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class ScaleChannelSpatialAttention(nn.Module):
    """Adaptive Scale Fusion에서 사용되는 주의 모듈."""

    def __init__(self, in_channels: int, channel_wise_channels: int, out_channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_wise = nn.Sequential(
            nn.Conv2d(in_channels, channel_wise_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_wise_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_wise = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        _init_weights(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.avg_pool(inputs)
        channel_att = self.channel_wise(pooled)
        fused = channel_att * inputs + inputs
        spatial_map = torch.mean(fused, dim=1, keepdim=True)
        spatial_att = self.spatial_wise(spatial_map)
        fused = fused + spatial_att
        attn = self.attention_wise(fused)
        return attn


@dataclass
class ASFConfig:
    attention_type: str = "ScaleChannelSpatial"


class FPNC(nn.Module):
    """DBNet++에서 사용하는 FPNC 디코더."""

    def __init__(
        self,
        in_channels: Sequence[int],
        lateral_channels: int = 256,
        out_channels: int = 64,
        bias_on_lateral: bool = False,
        bn_re_on_lateral: bool = False,
        bias_on_smooth: bool = False,
        bn_re_on_smooth: bool = False,
        asf_cfg: Optional[dict] = None,
        conv_after_concat: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) == 0:
            raise ValueError("in_channels must not be empty")
        self.in_channels = list(in_channels)
        self.num_inputs = len(self.in_channels)
        self.lateral_channels = lateral_channels
        self.out_channels = out_channels
        self.bn_re_on_lateral = bn_re_on_lateral
        self.bn_re_on_smooth = bn_re_on_smooth
        self.asf_cfg = ASFConfig(**asf_cfg) if asf_cfg is not None else None
        self.conv_after_concat = conv_after_concat

        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()

        for in_ch in self.in_channels:
            self.lateral_convs.append(
                ConvBNAct(
                    in_ch,
                    lateral_channels,
                    kernel_size=1,
                    bias=bias_on_lateral,
                    use_bn=bn_re_on_lateral,
                    activation="relu" if bn_re_on_lateral else None,
                )
            )
            self.smooth_convs.append(
                ConvBNAct(
                    lateral_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=bias_on_smooth,
                    use_bn=bn_re_on_smooth,
                    activation="relu" if bn_re_on_smooth else None,
                )
            )

        total_out_channels = out_channels * self.num_inputs
        if self.asf_cfg is not None:
            if self.asf_cfg.attention_type != "ScaleChannelSpatial":
                raise NotImplementedError(
                    f"Unsupported ASF attention type: {self.asf_cfg.attention_type}"
                )
            self.asf_conv = ConvBNAct(
                total_out_channels,
                total_out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                use_bn=False,
                activation=None,
            )
            self.asf_attn = ScaleChannelSpatialAttention(
                total_out_channels,
                total_out_channels // 4,
                self.num_inputs,
            )

        if self.conv_after_concat:
            self.out_conv = ConvBNAct(
                total_out_channels,
                total_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                use_bn=True,
                activation="relu",
            )
        else:
            self.out_conv = None

        _init_weights(self)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"Expected {self.num_inputs} feature maps, got {len(inputs)} instead."
            )

        laterals: List[torch.Tensor] = [
            lateral_conv(feat)
            for feat, lateral_conv in zip(inputs, self.lateral_convs)
        ]

        for idx in range(self.num_inputs - 1, 0, -1):
            prev_shape = laterals[idx - 1].shape[2:]
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx], size=prev_shape, mode="nearest"
            )

        outs: List[torch.Tensor] = [smooth(lateral) for smooth, lateral in zip(self.smooth_convs, laterals)]

        target_size = outs[0].shape[2:]
        for idx in range(len(outs)):
            if outs[idx].shape[2:] != target_size:
                outs[idx] = F.interpolate(outs[idx], size=target_size, mode="nearest")

        fused = torch.cat(outs, dim=1)
        if self.asf_cfg is not None:
            asf_feature = self.asf_conv(fused)
            attention = self.asf_attn(asf_feature)
            enhanced = []
            for level_idx, feature in enumerate(outs):
                weight = attention[:, level_idx:level_idx + 1]
                enhanced.append(feature * weight)
            fused = torch.cat(enhanced, dim=1)

        if self.out_conv is not None:
            fused = self.out_conv(fused)

        return fused
