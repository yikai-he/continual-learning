"""BC GeneralPolicy uses raw observations and internal LayerNorm."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.config import load_config
from src.continual.bc_policy import GeneralPolicy, fit_bc
from src.continual.trajectory_diffusion import FeatureNormalizer


def policy():
    torch.manual_seed(3)
    return GeneralPolicy(-np.ones(4), np.ones(4), hidden_sizes=(8, 6))


def dataset(offset=0.0):
    generator = torch.Generator().manual_seed(7)
    observations = torch.randn(12, 39, generator=generator) + offset
    actions = torch.tanh(torch.randn(12, 4, generator=generator))
    return TensorDataset(observations, actions)


class GeneralPolicyTests(unittest.TestCase):
    def test_diffusion_and_no_replay_select_same_policy_preprocessing(self):
        root = Path(__file__).resolve().parents[1]
        modes = {
            load_config(root / name).bc.normalization_mode
            for name in (
                "configs/diffcrl/diffcrl.yaml",
                "configs/diffcrl/diffcrl_no_replay.yaml",
            )
        }
        self.assertEqual(modes, {"layer-norm"})

    def test_raw_single_and_batched_observations_and_layer_norm_order(self):
        model = policy()
        self.assertIsInstance(model.network[0], nn.Linear)
        self.assertIsInstance(model.network[1], nn.LayerNorm)
        self.assertIsInstance(model.network[2], nn.ReLU)
        self.assertEqual(model.network[1].normalized_shape, (8,))
        self.assertFalse(hasattr(model, "obs_mean"))
        self.assertFalse(hasattr(model, "obs_std"))

        raw = torch.randn(39)
        seen = []
        handle = model.network[0].register_forward_pre_hook(
            lambda _module, inputs: seen.append(inputs[0].detach().clone())
        )
        try:
            self.assertEqual(model(raw).shape, (4,))
            self.assertEqual(model(torch.stack((raw, raw))).shape, (2, 4))
        finally:
            handle.remove()
        torch.testing.assert_close(seen[0], raw)

    def test_new_stage_data_cannot_drift_output_without_optimizer_step(self):
        model = policy()
        fixed = torch.randn(5, 39)
        before = model(fixed).detach().clone()
        state_before = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with patch.object(optimizer, "step", return_value=None):
            fit_bc(
                model,
                dataset(offset=1000.0),
                dataset(offset=-1000.0),
                epochs=1,
                optimizer=optimizer,
            )

        torch.testing.assert_close(model(fixed), before)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state_before[name])

    def test_loss_modes_and_action_bounds_are_preserved(self):
        for loss in ("post-tanh-mse", "pre-tanh-mse"):
            with self.subTest(loss=loss):
                model = policy()
                result = fit_bc(model, dataset(), dataset(), epochs=1, bc_loss=loss)
                self.assertTrue(np.isfinite(result["train_loss"]))
                actions = model(torch.randn(20, 39))
                self.assertTrue(torch.all(actions >= model.action_low))
                self.assertTrue(torch.all(actions <= model.action_high))

    def test_save_reload_and_old_checkpoint_rejection(self):
        model = policy().eval()
        raw = torch.randn(39)
        expected = model(raw).detach()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            model.save(path)
            restored = GeneralPolicy.load(path)
            torch.testing.assert_close(restored(raw), expected)

            old = Path(tmp) / "old.pt"
            torch.save({"hidden_sizes": (8, 6), "state_dict": model.state_dict()}, old)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                GeneralPolicy.load(old)

    def test_diffusion_feature_normalizer_is_unchanged(self):
        trajectories = torch.arange(2 * 3 * 43, dtype=torch.float32).reshape(2, 3, 43)
        normalizer = FeatureNormalizer.fit(trajectories)
        self.assertEqual(normalizer.mean.shape, (43,))
        torch.testing.assert_close(
            normalizer.denormalize(normalizer.normalize(trajectories)), trajectories
        )


if __name__ == "__main__":
    unittest.main()
