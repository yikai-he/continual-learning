"""Scientific contracts for direct 43D generator trajectories."""

import random
import unittest

import numpy as np
import torch

from src.continual.diffusion_data import (
    normalize_generator_data,
    prepare_generator_data,
    project_actions,
    transform_actions,
    validate_stage_trajectory_group,
)
from src.continual.generator_normalization import FeatureNormalizer


def valid_batch():
    current = torch.arange(2 * 4 * 18, dtype=torch.float32).reshape(2, 4, 18) / 100
    previous = torch.cat((current[:, :1], current[:, :-1]), dim=1)
    goals = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]])
    goals = goals[:, None].expand(-1, current.shape[1], -1).clone()
    actions = torch.linspace(-0.8, 0.8, 2 * 4 * 4).reshape(2, 4, 4)
    return torch.cat((current, previous, goals, actions), dim=-1)


class DirectTrajectoryTests(unittest.TestCase):
    def test_generator_preparation_normalizes_all_43_features_without_rng_use(self):
        physical = valid_batch()
        train, validation = physical[:1], physical[1:]
        transformed_train = transform_actions(train, "pre-tanh", 1e-4)
        transformed_validation = transform_actions(validation, "pre-tanh", 1e-4)
        train_goals = train[:, 0, 36:39]
        expected_normalizer = FeatureNormalizer.fit(transformed_train)
        expected_goal_normalizer = FeatureNormalizer.fit(train_goals[:, None, :])
        random.seed(61)
        np.random.seed(67)
        torch.manual_seed(71)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()

        result = prepare_generator_data(
            train, validation, action_space="pre-tanh", action_clamp_epsilon=1e-4
        )

        self.assertEqual(result.train.shape[-1], 43)
        torch.testing.assert_close(result.train, transformed_train)
        torch.testing.assert_close(result.validation, transformed_validation)
        torch.testing.assert_close(result.normalizer.mean, expected_normalizer.mean)
        torch.testing.assert_close(result.goal_normalizer.mean, expected_goal_normalizer.mean)
        self.assertEqual(random.getstate(), python_rng)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_rng[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
        self.assertEqual(numpy_after[2:], numpy_rng[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng)
        normalized = normalize_generator_data(result)
        self.assertEqual(normalized.train.shape[-1], 43)
        torch.testing.assert_close(
            normalized.train, expected_normalizer.normalize(transformed_train)
        )

    def test_action_transform_and_projection_change_only_last_four_features(self):
        physical = valid_batch()
        latent = transform_actions(physical, "pre-tanh")
        restored = transform_actions(latent, "pre-tanh", decode=True)
        torch.testing.assert_close(latent[..., :39], physical[..., :39])
        torch.testing.assert_close(restored, physical, atol=1e-6, rtol=1e-6)
        generated = physical.clone()
        generated[..., 39:] = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        projected = project_actions(generated, "clip")
        self.assertTrue(torch.equal(projected[..., :39], generated[..., :39]))
        self.assertTrue(torch.equal(projected[..., 39:], generated[..., 39:].clamp(-1, 1)))

    def test_stage_validation_accepts_learned_previous_state_and_goal_values(self):
        physical = valid_batch()
        changed = physical.clone()
        changed[0, 2, 18] += 2
        changed[0, 3, 36] += 3
        changed[1, 1, -1] = 1.5
        validate_stage_trajectory_group(
            changed, stage=1, task="old", source={"kind": "generated"}
        )
        validate_stage_trajectory_group(
            changed,
            stage=1,
            task="current",
            source={
                "kind": "real_expert",
                "collection_configurations": [{"task_hash": "unused"}],
            },
        )

    def test_stage_validation_retains_shape_and_finiteness_contracts(self):
        physical = valid_batch()
        with self.assertRaisesRegex(ValueError, "Invalid trajectory group shape"):
            validate_stage_trajectory_group(
                physical[..., :-1],
                stage=1,
                task="old",
                source={"kind": "generated"},
            )
        physical[0, 0, 18] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite trajectory data"):
            validate_stage_trajectory_group(
                physical,
                stage=1,
                task="current",
                source={"kind": "real_expert"},
            )


if __name__ == "__main__":
    unittest.main()
