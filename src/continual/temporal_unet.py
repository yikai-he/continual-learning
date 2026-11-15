"""Shared three-scale temporal U-Net body for trajectory generators."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ConditionedResidualBlock(nn.Module):
    """One ordinary-dilation residual temporal block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, values, condition):
        return (values + self.block(values + condition)) / math.sqrt(2)


def make_temporal_unet_layers(width: int):
    """Create the shared body layers in their stable initialization order."""
    widths = (width, 2 * width, 4 * width)
    return (
        nn.ModuleList(
            [nn.Identity(), nn.Linear(width, widths[1]), nn.Linear(width, widths[2])]
        ),
        nn.ModuleList(
            [ConditionedResidualBlock(widths[0]), ConditionedResidualBlock(widths[1])]
        ),
        nn.ModuleList(
            [
                nn.Conv1d(widths[0], widths[1], 4, stride=2, padding=1),
                nn.Conv1d(widths[1], widths[2], 4, stride=2, padding=1),
            ]
        ),
        nn.ModuleList(
            [ConditionedResidualBlock(widths[2]), ConditionedResidualBlock(widths[2])]
        ),
        nn.ModuleList(
            [
                nn.Conv1d(widths[2], widths[1], 3, padding=1),
                nn.Conv1d(widths[1], widths[0], 3, padding=1),
            ]
        ),
        nn.ModuleList(
            [ConditionedResidualBlock(widths[1]), ConditionedResidualBlock(widths[0])]
        ),
    )


def temporal_unet_forward(module, x, condition):
    """Run shared encoder-decoder layers using the caller's named modules."""
    conditions = [
        projection(condition)[:, :, None]
        for projection in module.condition_projections
    ]
    position = torch.linspace(-1, 1, x.shape[1], device=x.device)
    position = position[None, None].expand(x.shape[0], 1, -1)
    hidden = module.input(torch.cat((x.transpose(1, 2), position), dim=1))

    first_skip = module.encoder_blocks[0](hidden, conditions[0])
    hidden = module.downsamples[0](first_skip)
    second_skip = module.encoder_blocks[1](hidden, conditions[1])
    hidden = module.downsamples[1](second_skip)
    for block in module.bottleneck_blocks:
        hidden = block(hidden, conditions[2])

    hidden = F.interpolate(hidden, size=second_skip.shape[-1], mode="nearest")
    hidden = module.upsample_convolutions[0](hidden)
    hidden = (hidden + second_skip) / math.sqrt(2)
    hidden = module.decoder_blocks[0](hidden, conditions[1])
    hidden = F.interpolate(hidden, size=first_skip.shape[-1], mode="nearest")
    hidden = module.upsample_convolutions[1](hidden)
    hidden = (hidden + first_skip) / math.sqrt(2)
    hidden = module.decoder_blocks[1](hidden, conditions[0])
    return module.output(hidden).transpose(1, 2)
