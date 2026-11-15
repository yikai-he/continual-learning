"""Checkpoint and seeded-sampling contracts for trajectory diffusion."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.config import DiffusionModelConfig
from src.continual.trajectory_diffusion import (
    CHECKPOINT_KIND,
    LEGACY_CHECKPOINT_KIND,
    FeatureNormalizer,
    TemporalDenoiser,
    TemporalUNetDenoiser,
    TrajectoryDiffusion,
    fit_diffusion,
)


class TrajectoryDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = DiffusionModelConfig(
            horizon=4, features=43, steps=3, width=8, num_tasks=2
        )
        self.model = TrajectoryDiffusion(self.config)
        self.dynamic_normalizer = FeatureNormalizer(
            torch.arange(43, dtype=torch.float32), torch.ones(43)
        )
        self.goal_normalizer = FeatureNormalizer(
            torch.tensor([0.1, 0.2, 0.3]), torch.tensor([1.0, 2.0, 3.0])
        )

    def test_checkpoint_round_trip_includes_normalizers_and_metadata(self):
        metadata = {
            "task_order": ["reach-v3", "push-v3"],
            "generated_action_projection": "none",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diffusion.pt"
            self.model.save(
                path, self.dynamic_normalizer, self.goal_normalizer, metadata
            )
            loaded, dynamic, goal, loaded_metadata = TrajectoryDiffusion.load(
                path,
                expected_config=self.config,
                expected_action_space="raw",
                expected_clamp_epsilon=1e-4,
                expected_projection="none",
                expected_task_order=("reach-v3", "push-v3"),
            )

        self.assertEqual(loaded.config, self.config)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], value))
        self.assertTrue(torch.equal(dynamic.mean, self.dynamic_normalizer.mean))
        self.assertTrue(torch.equal(dynamic.scale, self.dynamic_normalizer.scale))
        self.assertTrue(torch.equal(goal.mean, self.goal_normalizer.mean))
        self.assertTrue(torch.equal(goal.scale, self.goal_normalizer.scale))
        self.assertEqual(loaded_metadata, metadata)
        self.assertEqual(loaded.checkpoint_kind, CHECKPOINT_KIND)

    def test_v2_checkpoint_loads_only_as_original_cnn(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.pt"
            legacy = Path(directory) / "legacy.pt"
            self.model.save(
                current,
                self.dynamic_normalizer,
                self.goal_normalizer,
                {"task_order": ["reach-v3", "push-v3"]},
            )
            payload = torch.load(current, map_location="cpu", weights_only=True)
            payload["kind"] = LEGACY_CHECKPOINT_KIND
            payload["config"].pop("backbone")
            payload["config"].pop("backbone_spec_version")
            torch.save(payload, legacy)
            loaded, _, _, _ = TrajectoryDiffusion.load(
                legacy, expected_config=self.config
            )

        self.assertIsInstance(loaded.denoiser, TemporalDenoiser)
        self.assertEqual(loaded.checkpoint_kind, LEGACY_CHECKPOINT_KIND)

    def test_v3_requires_explicit_backbone_identity(self):
        for missing in ("backbone", "backbone_spec_version"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                path = self._mutated_checkpoint(
                    directory, lambda payload: payload["config"].pop(missing)
                )
                with self.assertRaisesRegex(ValueError, "backbone identity"):
                    TrajectoryDiffusion.load(path)

    def test_v2_cannot_claim_unet(self):
        def mutate(payload):
            payload["kind"] = LEGACY_CHECKPOINT_KIND
            payload["config"]["backbone"] = "temporal_unet1d"

        with tempfile.TemporaryDirectory() as directory:
            path = self._mutated_checkpoint(directory, mutate)
            with self.assertRaisesRegex(ValueError, "v2 checkpoints support"):
                TrajectoryDiffusion.load(path)

    def test_unet_v3_round_trip(self):
        config = DiffusionModelConfig(
            horizon=9,
            features=43,
            steps=3,
            backbone="temporal_unet1d",
            width=8,
            num_tasks=2,
        )
        model = TrajectoryDiffusion(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unet.pt"
            model.save(path, self.dynamic_normalizer, self.goal_normalizer, {})
            loaded, _, _, _ = TrajectoryDiffusion.load(
                path, expected_config=config
            )
        self.assertIsInstance(loaded.denoiser, TemporalUNetDenoiser)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], value))

    def test_unet_shapes_conditions_loss_and_backward(self):
        for horizon in (4, 200, 201):
            with self.subTest(horizon=horizon):
                torch.manual_seed(17)
                model = TrajectoryDiffusion(
                    DiffusionModelConfig(
                        horizon=horizon,
                        steps=3,
                        backbone="temporal_unet1d",
                        width=4 if horizon == 4 else 8,
                        num_tasks=2,
                    )
                )
                values = torch.randn(2, horizon, 43)
                timestep = torch.tensor([0, 2])
                tasks = torch.tensor([0, 1])
                goals = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]])
                output = model(values, timestep, tasks, goals=goals)
                changed_time = model(values, timestep.flip(0), tasks, goals=goals)
                changed_task = model(values, timestep, tasks.flip(0), goals=goals)
                changed_goal = model(values, timestep, tasks, goals=goals.flip(0))
                loss = model.loss(
                    values,
                    generator=torch.Generator().manual_seed(23),
                    task_ids=tasks,
                    goals=goals,
                )
                loss.backward()
                self.assertEqual(output.shape, values.shape)
                self.assertTrue(torch.isfinite(output).all() and torch.isfinite(loss))
                self.assertFalse(torch.allclose(output, changed_time))
                self.assertFalse(torch.allclose(output, changed_task))
                self.assertFalse(torch.allclose(output, changed_goal))
                self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_horizon_below_four_is_rejected_only_for_unet(self):
        cnn = TrajectoryDiffusion(DiffusionModelConfig(horizon=1, width=4))
        self.assertIsInstance(cnn.denoiser, TemporalDenoiser)
        with self.assertRaisesRegex(ValueError, "Invalid diffusion"):
            TrajectoryDiffusion(
                DiffusionModelConfig(
                    horizon=3, width=4, backbone="temporal_unet1d"
                )
            )

    def test_unet_two_stage_fit_keeps_the_persistent_model(self):
        model = TrajectoryDiffusion(
            DiffusionModelConfig(
                horizon=4,
                steps=2,
                backbone="temporal_unet1d",
                width=4,
                num_tasks=2,
            )
        )
        identity = id(model.denoiser)
        train = torch.randn(2, 4, 43)
        validation = torch.randn(2, 4, 43)
        goals = torch.randn(2, 3)
        task_ids = torch.tensor([0, 1])
        initial = {name: value.clone() for name, value in model.state_dict().items()}

        for stage in (0, 1):
            fit_diffusion(
                model,
                train,
                validation,
                goals,
                goals,
                train_task_ids=task_ids,
                validation_task_ids=task_ids,
                epochs=1,
                batch_size=2,
                seed=stage,
            )

        self.assertEqual(id(model.denoiser), identity)
        self.assertTrue(
            any(
                not torch.equal(value, initial[name])
                for name, value in model.state_dict().items()
                if value.is_floating_point()
            )
        )

    def _mutated_checkpoint(self, directory, mutate):
        original = Path(directory) / "original.pt"
        changed = Path(directory) / "changed.pt"
        self.model.save(
            original,
            self.dynamic_normalizer,
            self.goal_normalizer,
            {"task_order": ["reach-v3", "push-v3"], "generated_action_projection": "none"},
        )
        payload = torch.load(original, map_location="cpu", weights_only=True)
        mutate(payload)
        torch.save(payload, changed)
        return changed

    def test_expected_model_configuration_rejects_conditioning_mismatch(self):
        incompatible = DiffusionModelConfig(
            horizon=4, features=43, steps=3, width=8, num_tasks=3
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._mutated_checkpoint(directory, lambda payload: None)
            with self.assertRaisesRegex(ValueError, "model configuration"):
                TrajectoryDiffusion.load(path, expected_config=incompatible)

    def test_invalid_normalization_is_rejected(self):
        def invalidate(payload):
            payload["dynamic_scale"][0] = 0

        with tempfile.TemporaryDirectory() as directory:
            path = self._mutated_checkpoint(directory, invalidate)
            with self.assertRaisesRegex(ValueError, "dynamic normalization"):
                TrajectoryDiffusion.load(path)

    def test_legacy_22d_checkpoint_is_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"kind": "trajectory_diffusion_22_v1"}, path)
            with self.assertRaisesRegex(ValueError, "legacy 22D checkpoints"):
                TrajectoryDiffusion.load(path)

    def test_schedule_mismatch_is_rejected(self):
        def change_schedule(payload):
            payload["state_dict"]["abar"][0] += 1e-3

        with tempfile.TemporaryDirectory() as directory:
            path = self._mutated_checkpoint(directory, change_schedule)
            with self.assertRaisesRegex(ValueError, "schedule"):
                TrajectoryDiffusion.load(path)

    def test_model_architecture_mismatch_is_rejected(self):
        def change_architecture(payload):
            payload["state_dict"]["denoiser.output.weight"] = payload["state_dict"][
                "denoiser.output.weight"
            ][:-1]

        with tempfile.TemporaryDirectory() as directory:
            path = self._mutated_checkpoint(directory, change_architecture)
            with self.assertRaisesRegex(ValueError, "model state"):
                TrajectoryDiffusion.load(path)

    def test_sampling_is_deterministic_for_seed_and_conditions(self):
        task_ids = torch.tensor([0, 1], dtype=torch.long)
        goals = torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.4]])

        first = self.model.sample(2, seed=19, task_ids=task_ids, goals=goals)
        second = self.model.sample(2, seed=19, task_ids=task_ids, goals=goals)
        different = self.model.sample(2, seed=20, task_ids=task_ids, goals=goals)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertEqual(first.shape, (2, 4, 43))

    def test_device_native_sampling_matches_legacy_on_cpu(self):
        task_ids = torch.tensor([0, 1], dtype=torch.long)
        goals = torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.4]])

        legacy = self.model.sample(2, seed=19, task_ids=task_ids, goals=goals)
        first = self.model.sample(
            2,
            seed=19,
            task_ids=task_ids,
            goals=goals,
            execution_mode="device-native",
        )
        second = self.model.sample(
            2,
            seed=19,
            task_ids=task_ids,
            goals=goals,
            execution_mode="device-native",
        )

        self.assertTrue(torch.equal(first, legacy))
        self.assertTrue(torch.equal(second, first))
        self.assertEqual(first.device.type, "cpu")
        self.assertTrue(torch.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
