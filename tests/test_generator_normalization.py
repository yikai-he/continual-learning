"""Shared normalization and legacy-import compatibility contracts."""

import torch

from src.continual.generator_normalization import FeatureNormalizer
from src.continual.trajectory_diffusion import FeatureNormalizer as LegacyNormalizer


def test_legacy_import_reexports_shared_normalizer():
    assert LegacyNormalizer is FeatureNormalizer


def test_normalization_round_trip_preserves_established_statistics():
    values = torch.tensor(
        [
            [[1.0, 5.0], [3.0, 5.0]],
            [[5.0, 5.0], [7.0, 5.0]],
        ]
    )

    normalizer = FeatureNormalizer.fit(values)
    normalized = normalizer.normalize(values)

    torch.testing.assert_close(normalizer.mean, torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(
        normalizer.scale,
        torch.tensor([values[..., 0].std(unbiased=False), 1e-3]),
    )
    torch.testing.assert_close(normalizer.denormalize(normalized), values)
