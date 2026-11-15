"""BC GeneralPolicy uses raw observations and internal LayerNorm."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.continual.bc_policy import (
    LEGACY_LEAKY_RELU_NEGATIVE_SLOPE,
    NLL_ATANH_EPSILON,
    GeneralPolicy,
    fit_bc,
)
from src.continual.trajectory_diffusion import FeatureNormalizer


def policy():
    torch.manual_seed(3)
    return GeneralPolicy(-np.ones(4), np.ones(4))


def dataset(offset=0.0):
    generator = torch.Generator().manual_seed(7)
    observations = torch.randn(12, 39, generator=generator) + offset
    actions = torch.tanh(torch.randn(12, 4, generator=generator))
    return TensorDataset(observations, actions)


class GeneralPolicyTests(unittest.TestCase):
    def test_relu_default_and_legacy_inspired_leaky_relu_architecture(self):
        relu = GeneralPolicy(-np.ones(4), np.ones(4))
        leaky = GeneralPolicy(
            -np.ones(4), np.ones(4), activation="leaky_relu"
        )
        self.assertEqual(relu.activation, "relu")
        self.assertEqual(
            [type(leaky.shared_network[index]) for index in (2, 4, 6, 8)],
            [nn.Tanh, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU],
        )
        self.assertTrue(
            all(
                leaky.shared_network[index].negative_slope
                == LEGACY_LEAKY_RELU_NEGATIVE_SLOPE
                for index in (4, 6, 8)
            )
        )
        self.assertEqual(leaky.hidden_sizes, (256, 256, 256, 256))
        self.assertIsInstance(leaky.shared_network[1], nn.LayerNorm)
        self.assertIsInstance(leaky.mean_head, nn.Linear)
        self.assertIsInstance(leaky.log_std_head, nn.Linear)

    def test_invalid_activation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "activation"):
            GeneralPolicy(-np.ones(4), np.ones(4), activation="gelu")

    def test_legacy_inspired_4x512_architecture_and_finite_nll(self):
        model = GeneralPolicy(
            -np.ones(4),
            np.ones(4),
            activation="leaky_relu",
            hidden_dim=512,
            initial_log_std=-1.0,
        )
        linear = [
            module for module in model.shared_network if isinstance(module, nn.Linear)
        ]
        self.assertEqual(
            [(module.in_features, module.out_features) for module in linear],
            [(39, 512), (512, 512), (512, 512), (512, 512)],
        )
        self.assertEqual(model.shared_network[1].normalized_shape, (512,))
        self.assertEqual(model.mean_head.in_features, 512)
        self.assertEqual(model.mean_head.out_features, 4)
        self.assertEqual(model.log_std_head.in_features, 512)
        self.assertTrue(torch.all(model.log_std_head.bias == -1))
        observations, actions = dataset().tensors
        loss = -model.normalized_log_prob(observations, actions).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
        observation = observations[0].numpy()
        for deterministic in (True, False):
            action = model.act(observation, deterministic=deterministic)
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue(np.all(action >= -1) and np.all(action <= 1))

    def test_invalid_hidden_dim_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            GeneralPolicy(-np.ones(4), np.ones(4), hidden_dim=384)

    def test_activation_changes_function_with_identical_weights(self):
        torch.manual_seed(31)
        relu = GeneralPolicy(-np.ones(4), np.ones(4))
        leaky = GeneralPolicy(
            -np.ones(4), np.ones(4), activation="leaky_relu"
        )
        leaky.load_state_dict(relu.state_dict())
        observations = torch.randn(32, 39)
        self.assertFalse(torch.equal(relu(observations), leaky(observations)))

    def test_configured_initial_log_std_changes_only_bias(self):
        torch.manual_seed(7)
        historical = GeneralPolicy(-np.ones(4), np.ones(4))
        torch.manual_seed(7)
        stabilized = GeneralPolicy(
            -np.ones(4), np.ones(4), initial_log_std=-1.0
        )
        differing = [
            name
            for name, value in historical.state_dict().items()
            if not torch.equal(value, stabilized.state_dict()[name])
        ]
        self.assertEqual(differing, ["log_std_head.bias"])
        self.assertTrue(torch.all(historical.log_std_head.bias == -3))
        self.assertTrue(torch.all(stabilized.log_std_head.bias == -1))

    def test_initial_log_std_rejects_values_outside_bounds(self):
        for value in (-21.0, 3.0, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "initial_log_std"
            ):
                GeneralPolicy(-np.ones(4), np.ones(4), initial_log_std=value)

    def test_four_layer_architecture_forward_shape_and_bounds(self):
        model = GeneralPolicy(-np.ones(4), np.ones(4))
        self.assertEqual(
            [type(module) for module in model.shared_network],
            [
                nn.Linear,
                nn.LayerNorm,
                nn.ReLU,
                nn.Linear,
                nn.ReLU,
                nn.Linear,
                nn.ReLU,
                nn.Linear,
                nn.ReLU,
            ],
        )
        self.assertIsInstance(model.mean_head, nn.Linear)
        self.assertIsInstance(model.log_std_head, nn.Linear)
        mean, log_std = model.distribution_parameters(torch.randn(7, 39))
        self.assertEqual(mean.shape, (7, 4))
        self.assertEqual(log_std.shape, (7, 4))
        actions = model(torch.randn(7, 39))
        self.assertEqual(actions.shape, (7, 4))
        self.assertTrue(torch.all(actions >= model.action_low))
        self.assertTrue(torch.all(actions <= model.action_high))

    def test_raw_single_and_batched_observations_and_layer_norm_order(self):
        model = policy()
        self.assertIsInstance(model.shared_network[0], nn.Linear)
        self.assertIsInstance(model.shared_network[1], nn.LayerNorm)
        self.assertIsInstance(model.shared_network[2], nn.ReLU)
        self.assertEqual(model.shared_network[1].normalized_shape, (256,))
        self.assertFalse(hasattr(model, "obs_mean"))
        self.assertFalse(hasattr(model, "obs_std"))

        raw = torch.randn(39)
        seen = []
        handle = model.shared_network[0].register_forward_pre_hook(
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

    def test_nll_training_is_finite_updates_parameters_and_preserves_bounds(self):
        model = policy()
        before = {name: value.clone() for name, value in model.named_parameters()}
        result = fit_bc(model, dataset(), dataset(), epochs=2)
        self.assertTrue(np.isfinite(result["train_loss"]))
        self.assertEqual(
            result["validation_loss"],
            min(record["validation_loss"] for record in result["history"]),
        )
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in model.named_parameters()
            )
        )
        actions = model(torch.randn(20, 39))
        self.assertTrue(torch.all(actions >= model.action_low))
        self.assertTrue(torch.all(actions <= model.action_high))

    def test_squashed_gaussian_log_prob_includes_tanh_jacobian(self):
        low = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
        high = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
        model = GeneralPolicy(low, high)
        observations = torch.randn(5, 39)
        normalized = torch.linspace(-0.8, 0.8, 20).reshape(5, 4)
        actions = model.scale_action(normalized)
        mean, log_std = model.distribution_parameters(observations)
        pre_tanh = torch.atanh(normalized)
        manual = (
            torch.distributions.Normal(mean, log_std.exp()).log_prob(pre_tanh)
            - torch.log1p(-normalized.square())
        ).sum(dim=-1)
        torch.testing.assert_close(
            model.normalized_log_prob(observations, actions), manual
        )

    def test_boundary_nll_sampling_and_log_std_bounds_are_finite(self):
        model = policy()
        observations = torch.randn(6, 39)
        actions = torch.tensor([[-1.0, 1.0, -1.0, 1.0]]).repeat(6, 1)
        loss = -model.normalized_log_prob(observations, actions).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )
        with torch.no_grad():
            model.log_std_head.bias.fill_(100.0)
        _, log_std = model.distribution_parameters(observations)
        self.assertTrue(torch.all(log_std == 2.0))
        observation = np.zeros(39, dtype=np.float32)
        deterministic = model.act(observation, deterministic=True)
        np.testing.assert_array_equal(
            deterministic,
            model(torch.from_numpy(observation)).detach().numpy(),
        )
        samples = [model.act(observation, deterministic=False) for _ in range(4)]
        self.assertTrue(all(np.isfinite(sample).all() for sample in samples))
        self.assertTrue(
            any(not np.array_equal(deterministic, sample) for sample in samples)
        )
        self.assertEqual(NLL_ATANH_EPSILON, 1e-4)

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
            torch.save({"kind": "general_policy_v2", "state_dict": {}}, old)
            with self.assertRaisesRegex(ValueError, "Deterministic"):
                GeneralPolicy.load(old)

    def test_checkpoint_round_trip_and_architecture_validation_for_both_modes(self):
        raw = torch.randn(3, 39)
        with tempfile.TemporaryDirectory() as tmp:
            model = GeneralPolicy(
                -np.ones(4), np.ones(4), initial_log_std=-1.0
            ).eval()
            path = Path(tmp) / "policy.pt"
            model.save(path)
            restored = GeneralPolicy.load(path, expected_use_layer_norm=True)
            self.assertTrue(restored.use_layer_norm)
            self.assertEqual(restored.initial_log_std, -1.0)
            torch.testing.assert_close(restored(raw), model(raw))
            with self.assertRaisesRegex(ValueError, "does not match"):
                GeneralPolicy.load(path, expected_use_layer_norm=False)

            legacy = Path(tmp) / "legacy.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload.pop("initial_log_std")
            torch.save(payload, legacy)
            historical = GeneralPolicy.load(legacy)
            self.assertEqual(historical.initial_log_std, -3.0)
            torch.testing.assert_close(historical(raw), model(raw))

    def test_activation_checkpoint_round_trip_legacy_default_and_mismatch(self):
        raw = torch.randn(3, 39)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaky.pt"
            model = GeneralPolicy(
                -np.ones(4), np.ones(4), activation="leaky_relu"
            ).eval()
            model.save(path)
            restored = GeneralPolicy.load(
                path, expected_activation="leaky_relu"
            )
            self.assertEqual(restored.activation, "leaky_relu")
            torch.testing.assert_close(restored(raw), model(raw))
            with self.assertRaisesRegex(ValueError, "activation does not match"):
                GeneralPolicy.load(path, expected_activation="relu")

            old = Path(tmp) / "old-relu.pt"
            relu = GeneralPolicy(-np.ones(4), np.ones(4)).eval()
            relu.save(old)
            payload = torch.load(old, map_location="cpu", weights_only=True)
            payload.pop("activation")
            payload["architecture"].pop("activation")
            payload["architecture"].pop("first_hidden_activation")
            payload["architecture"].pop("leaky_relu_negative_slope")
            torch.save(payload, old)
            old_restored = GeneralPolicy.load(old, expected_activation="relu")
            self.assertEqual(old_restored.activation, "relu")
            torch.testing.assert_close(old_restored(raw), relu(raw))

    def test_4x512_checkpoint_round_trip_and_width_mismatch(self):
        raw = torch.randn(3, 39)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy-4x512.pt"
            model = GeneralPolicy(
                -np.ones(4),
                np.ones(4),
                activation="leaky_relu",
                hidden_dim=512,
                initial_log_std=-1.0,
            ).eval()
            model.save(path)
            restored = GeneralPolicy.load(
                path,
                expected_activation="leaky_relu",
                expected_hidden_sizes=(512, 512, 512, 512),
            )
            self.assertEqual(restored.hidden_sizes, (512, 512, 512, 512))
            torch.testing.assert_close(restored(raw), model(raw))
            with self.assertRaisesRegex(ValueError, "architecture does not match"):
                GeneralPolicy.load(
                    path, expected_hidden_sizes=(256, 256, 256, 256)
                )

    def test_four_layer_save_reload_and_training_resume_equivalence(self):
        torch.manual_seed(13)
        model = GeneralPolicy(-np.ones(4), np.ones(4)).eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        raw = torch.randn(3, 39)
        expected = model(raw).detach()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            model.save(path, optimizer=optimizer, metadata={"stage": 1})
            restored, restored_optimizer, metadata = GeneralPolicy.load_training(path)
            self.assertEqual(restored.hidden_sizes, (256, 256, 256, 256))
            self.assertEqual(metadata, {"stage": 1})
            self.assertIsInstance(restored_optimizer, torch.optim.Adam)
            torch.testing.assert_close(restored(raw), expected)
            self.assertEqual(
                restored_optimizer.state_dict()["param_groups"],
                optimizer.state_dict()["param_groups"],
            )

    def test_corrupt_stored_architecture_has_clear_error(self):
        model = policy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            model.save(path)
            data = torch.load(path, map_location="cpu", weights_only=True)
            data["hidden_sizes"] = (256, 256, 256)
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "stored GeneralPolicy architecture"):
                GeneralPolicy.load(path)

    def test_diffusion_feature_normalizer_is_unchanged(self):
        trajectories = torch.arange(2 * 3 * 43, dtype=torch.float32).reshape(2, 3, 43)
        normalizer = FeatureNormalizer.fit(trajectories)
        self.assertEqual(normalizer.mean.shape, (43,))
        torch.testing.assert_close(
            normalizer.denormalize(normalizer.normalize(trajectories)), trajectories
        )


if __name__ == "__main__":
    unittest.main()
