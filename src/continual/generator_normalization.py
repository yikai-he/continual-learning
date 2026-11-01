"""Shared feature normalization for trajectory replay generators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def validate_cpu_feature_batch(x: torch.Tensor) -> None:
    """Validate the established CPU ``(N,T,F)`` normalization contract."""
    if (
        x.ndim != 3
        or min(x.shape) < 1
        or not x.is_floating_point()
        or x.device.type != "cpu"
        or not torch.isfinite(x).all()
    ):
        raise ValueError("Expected a nonempty finite floating CPU tensor (N,T,F).")


@dataclass
class FeatureNormalizer:
    """Per-feature z-score fitted only on supplied training episodes; no clipping."""

    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(
        cls, train: torch.Tensor, *, minimum_scale: float = 1e-3
    ) -> FeatureNormalizer:
        """Fit per-feature moments over batch and time with a scale floor."""
        validate_cpu_feature_batch(train)
        if not math.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive and finite.")
        return cls(
            train.mean(dim=(0, 1)),
            train.std(dim=(0, 1), unbiased=False).clamp_min(minimum_scale),
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        validate_cpu_feature_batch(x)
        if x.shape[-1] != self.mean.numel():
            raise ValueError("Feature count differs from fitted normalization.")
        return (x - self.mean) / self.scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        validate_cpu_feature_batch(x)
        if x.shape[-1] != self.mean.numel():
            raise ValueError("Feature count differs from fitted normalization.")
        return x * self.scale + self.mean
