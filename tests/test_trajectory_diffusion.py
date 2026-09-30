"""Checkpoint and seeded-sampling contracts for trajectory diffusion."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.config import DiffusionModelConfig
from src.continual.trajectory_diffusion import FeatureNormalizer, TrajectoryDiffusion


class TrajectoryDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = DiffusionModelConfig(
            horizon=4, features=22, steps=3, width=8, num_tasks=2
        )
        self.model = TrajectoryDiffusion(self.config)
        self.dynamic_normalizer = FeatureNormalizer(
            torch.arange(22, dtype=torch.float32), torch.ones(22)
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

    def test_sampling_is_deterministic_for_seed_and_conditions(self):
        task_ids = torch.tensor([0, 1], dtype=torch.long)
        goals = torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.4]])

        first = self.model.sample(2, seed=19, task_ids=task_ids, goals=goals)
        second = self.model.sample(2, seed=19, task_ids=task_ids, goals=goals)
        different = self.model.sample(2, seed=20, task_ids=task_ids, goals=goals)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertEqual(first.shape, (2, 4, 22))


if __name__ == "__main__":
    unittest.main()
