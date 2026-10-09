"""Scientific contracts for physical and structured trajectory representations."""

import unittest

import torch

from src.continual.diffusion_data import (
    decode_trajectory,
    encode_trajectory,
    project_actions,
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


if __name__ == "__main__":
    unittest.main()
