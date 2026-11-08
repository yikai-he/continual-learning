"""Scientific contracts for physical and structured trajectory representations."""

import random
import unittest

import numpy as np
import torch

from src.continual.diffusion_data import (
    decode_trajectory,
    encode_trajectory,
    generated_action_diagnostics,
    normalize_generator_data,
    project_actions,
    prepare_generator_data,
    transform_actions,
    validate_stage_trajectory_group,
)


def valid_batch():
    current = torch.arange(2 * 4 * 18, dtype=torch.float32).reshape(2, 4, 18) / 100
    previous = torch.cat((current[:, :1], current[:, :-1]), dim=1)
    goals = torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]])
    goals = goals[:, None].expand(-1, current.shape[1], -1).clone()
    actions = torch.linspace(-0.8, 0.8, 2 * 4 * 4).reshape(2, 4, 4)
    return torch.cat((current, previous, goals), dim=-1), actions


class StructuredTrajectoryTests(unittest.TestCase):
    def test_generator_preparation_matches_exact_operations_without_rng_use(self):
        observations, actions = valid_batch()
        physical = torch.cat((observations, actions), dim=-1)
        train, validation = physical[:1], physical[1:]
        train_dynamic, train_goals = encode_trajectory(
            train[..., :39], train[..., 39:]
        )
        validation_dynamic, validation_goals = encode_trajectory(
            validation[..., :39], validation[..., 39:]
        )
        transformed_train = transform_actions(train_dynamic, "pre-tanh", 1e-4)
        transformed_validation = transform_actions(
            validation_dynamic, "pre-tanh", 1e-4
        )
        from src.continual.generator_normalization import FeatureNormalizer

        expected_goal_normalizer = FeatureNormalizer.fit(train_goals[:, None, :])
        expected_normalizer = FeatureNormalizer.fit(transformed_train)
        random.seed(61)
        np.random.seed(67)
        torch.manual_seed(71)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()

        result = prepare_generator_data(
            train,
            validation,
            action_space="pre-tanh",
            action_clamp_epsilon=1e-4,
        )

        torch.testing.assert_close(
            result.normalizer.mean, expected_normalizer.mean, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.normalizer.scale, expected_normalizer.scale, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.goal_normalizer.mean,
            expected_goal_normalizer.mean,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.goal_normalizer.scale,
            expected_goal_normalizer.scale,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.train,
            transformed_train,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.validation,
            transformed_validation,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.train_goals,
            train_goals,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.validation_goals,
            validation_goals,
            rtol=0,
            atol=0,
        )
        self.assertEqual(random.getstate(), python_rng)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_rng[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
        self.assertEqual(numpy_after[2:], numpy_rng[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)

        normalized = normalize_generator_data(result)
        torch.testing.assert_close(
            normalized.train,
            expected_normalizer.normalize(transformed_train),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            normalized.validation,
            expected_normalizer.normalize(transformed_validation),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            normalized.train_goals,
            expected_goal_normalizer.normalize(train_goals[:, None, :])[:, 0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            normalized.validation_goals,
            expected_goal_normalizer.normalize(validation_goals[:, None, :])[:, 0],
            rtol=0,
            atol=0,
        )
        self.assertEqual(random.getstate(), python_rng)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_rng[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
        self.assertEqual(numpy_after[2:], numpy_rng[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)

    def test_encode_decode_round_trip(self):
        observations, actions = valid_batch()

        dynamic, goals = encode_trajectory(observations, actions)
        decoded = decode_trajectory(dynamic, goals)

        self.assertEqual(dynamic.shape, (2, 4, 22))
        self.assertTrue(torch.equal(goals, observations[:, 0, 36:39]))
        self.assertTrue(torch.equal(decoded, torch.cat((observations, actions), dim=-1)))

    def test_encode_rejects_invalid_previous_state_and_varying_goal(self):
        observations, actions = valid_batch()
        invalid_previous = observations.clone()
        invalid_previous[0, 2, 18] += 1
        with self.assertRaisesRegex(ValueError, "Previous state"):
            encode_trajectory(invalid_previous, actions)

        invalid_goal = observations.clone()
        invalid_goal[1, 3, 36] += 1
        with self.assertRaisesRegex(ValueError, "Goal"):
            encode_trajectory(invalid_goal, actions)

    def test_action_transform_round_trip_and_projection(self):
        observations, actions = valid_batch()
        dynamic, _ = encode_trajectory(observations, actions)

        latent = transform_actions(dynamic, "pre-tanh")
        restored = transform_actions(latent, "pre-tanh", decode=True)
        self.assertTrue(torch.equal(latent[..., :18], dynamic[..., :18]))
        self.assertTrue(torch.allclose(restored, dynamic, atol=1e-6, rtol=1e-6))

        generated = dynamic.clone()
        generated[..., 18:] = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        unprojected = project_actions(generated, "none")
        projected = project_actions(generated, "clip")
        self.assertTrue(torch.equal(unprojected, generated))
        self.assertTrue(torch.equal(projected[..., :18], generated[..., :18]))
        self.assertTrue(
            torch.equal(projected[..., 18:], generated[..., 18:].clamp(-1, 1))
        )

    def test_stage_preflight_rejects_any_goal_drift_with_context(self):
        observations, actions = valid_batch()
        packed = torch.cat((observations, actions), dim=-1)
        source = {
            "kind": "real_expert",
            "collection_configurations": [
                {"task_bank_index": 7, "task_hash": "abc", "reset_seed": 4},
                {"task_bank_index": 8, "task_hash": "def", "reset_seed": 5},
            ],
        }
        validate_stage_trajectory_group(
            packed, stage=3, task="kuka-faucet-close-v3", source=source
        )
        for drift in (1e-8, 1e-2):
            invalid = packed.clone()
            invalid[0, 1, 36] += drift
            with self.assertRaisesRegex(
                ValueError,
                "stage=3.*trajectory=0.*source=current.*task_bank_index=7.*first_conflicting_timestep=1",
            ):
                validate_stage_trajectory_group(
                    invalid,
                    stage=3,
                    task="kuka-faucet-close-v3",
                    source=source,
                )

    def test_generated_action_diagnostics_preserve_existing_contract(self):
        observations, actions = valid_batch()
        before = torch.cat((observations, 2 * torch.ones_like(actions)), dim=-1)
        after = before.clone()
        after[..., -4:] = after[..., -4:].clamp(-1, 1)

        result = generated_action_diagnostics(before, after, "clip")

        self.assertEqual(result["generated_action_projection"], "clip")
        self.assertEqual(result["pre_projection_oob_component_fraction"], 1.0)
        self.assertEqual(result["post_projection_oob_component_fraction"], 0.0)
        self.assertEqual(result["clipped_component_fraction"], 1.0)

        changed_observation = after.clone()
        changed_observation[..., 0] += 1
        with self.assertRaisesRegex(ValueError, "changed replay shape or observations"):
            generated_action_diagnostics(before, changed_observation, "clip")

if __name__ == "__main__":
    unittest.main()
