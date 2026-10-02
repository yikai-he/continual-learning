"""Previous-task replay generation is deterministic and independently testable."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from src.continual.diffcrl_replay import ReplayRequest, generate_previous_task_replay


class IdentityNormalizer:
    def normalize(self, value):
        return value

    def denormalize(self, value):
        return value


class FakeDenoiser:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class FakeDiffusion:
    def __init__(self):
        self.denoiser = FakeDenoiser()
        self.calls = []

    def sample(self, count, *, seed, task_ids, goals):
        self.calls.append((count, seed, task_ids.clone(), goals.clone()))
        generator = torch.Generator().manual_seed(seed)
        result = torch.randn((count, 200, 22), generator=generator)
        result[..., :18] += task_ids[:, None, None]
        result[..., 18:] = 2.0
        return result


def request(**overrides):
    values = {
        "stage": 2,
        "trajectory_count": 3,
        "checkpoint": Path("/tmp/replay-checkpoint.pt"),
        "expected_state_hash": "expected-state",
            "task_names": ("reach-v3", "push-v3", "hammer-v3"),
            "backend": "metaworld-v3",
            "horizon": 200,
        "runtime_seed": 23,
        "device": "cpu",
        "reward_function_version": "v2",
        "action_space": "raw",
        "action_clamp_epsilon": 1e-4,
        "action_projection": "clip",
        "bc_clamp_epsilon": 1e-4,
        "progress": False,
    }
    values.update(overrides)
    return ReplayRequest(**values)


def metadata():
    names = ("reach-v3", "push-v3", "hammer-v3")
    return {
        "training_banks": {
            name: {
                "task_name": name,
                "ordered_task_hashes": [f"{name}-{i}" for i in range(5)],
            }
            for name in names
        },
        "train_configurations": {name: {"indices": [1, 3]} for name in names},
    }


class DiffCRLReplayTests(unittest.TestCase):
    def test_exact_order_seeds_projection_provenance_and_progress(self):
        model = FakeDiffusion()
        normalizer = IdentityNormalizer()
        load = Mock(return_value=(model, normalizer, normalizer, metadata()))
        progress = Mock()

        def goals(bank, selected, *, backend, reward_function_version):
            self.assertEqual(backend, "metaworld-v3")
            self.assertEqual(reward_function_version, "v2")
            base = 1.0 if bank["task_name"] == "reach-v3" else 2.0
            return [[base, float(index), -base] for index in selected]

        with patch(
            "src.continual.diffcrl_replay.TrajectoryDiffusion.load", load
        ), patch(
            "src.continual.diffcrl_replay.reconstruct_goals",
            side_effect=goals,
        ), patch(
            "src.continual.diffcrl_replay.file_hash",
            return_value="checkpoint-sha",
        ), patch(
            "src.continual.diffcrl_replay.state_hash",
            return_value="expected-state",
        ):
            result = generate_previous_task_replay(request(), progress_message=progress)

        self.assertEqual(list(result.groups), [0, 1])
        self.assertEqual(
            [tuple(group.shape) for group in result.groups.values()],
            [(3, 200, 43), (3, 200, 43)],
        )
        self.assertEqual([call[1] for call in model.calls], [102023, 102024])
        torch.testing.assert_close(model.calls[0][2], torch.tensor([0, 0, 0]))
        torch.testing.assert_close(model.calls[1][2], torch.tensor([1, 1, 1]))
        self.assertEqual(model.denoiser.devices, ["cpu"])
        load.assert_called_once_with(
            Path("/tmp/replay-checkpoint.pt"),
            expected_action_space="raw",
            expected_clamp_epsilon=1e-4,
            expected_projection="clip",
            expected_task_order=("reach-v3", "push-v3", "hammer-v3"),
        )
        self.assertEqual(progress.call_count, 2)
        for old_index, name in enumerate(("reach-v3", "push-v3")):
            source = result.sources[name]
            self.assertEqual(source["sampling_seed"], 102023 + old_index)
            self.assertEqual(source["condition"], old_index)
            self.assertEqual(source["checkpoint_sha256"], "checkpoint-sha")
            self.assertIsNone(source["generated_latent_abs_gt_bc_cap_fraction"])
            self.assertEqual(source["pre_projection_oob_component_fraction"], 1.0)
            self.assertEqual(source["post_projection_oob_component_fraction"], 0.0)
            self.assertEqual(source["clipped_component_fraction"], 1.0)
            self.assertEqual(len(source["training_configuration_indices"]), 3)

    def test_identical_request_produces_exact_tensors(self):
        def run_once():
            model = FakeDiffusion()
            normalizer = IdentityNormalizer()
            with patch(
                "src.continual.diffcrl_replay.TrajectoryDiffusion.load",
                return_value=(model, normalizer, normalizer, metadata()),
            ), patch(
                "src.continual.diffcrl_replay.reconstruct_goals",
                side_effect=lambda bank, selected, **_: [
                    [1.0, float(index), -1.0] for index in selected
                ],
            ), patch(
                "src.continual.diffcrl_replay.file_hash", return_value="sha"
            ), patch(
                "src.continual.diffcrl_replay.state_hash",
                return_value="expected-state",
            ):
                return generate_previous_task_replay(request())

        first, second = run_once(), run_once()
        self.assertEqual(first.sources, second.sources)
        for index in first.groups:
            torch.testing.assert_close(
                first.groups[index], second.groups[index], rtol=0, atol=0
            )

    def test_state_hash_mismatch_fails_before_sampling(self):
        model = FakeDiffusion()
        normalizer = IdentityNormalizer()
        with patch(
            "src.continual.diffcrl_replay.TrajectoryDiffusion.load",
            return_value=(model, normalizer, normalizer, metadata()),
        ), patch(
            "src.continual.diffcrl_replay.state_hash",
            return_value="different-state",
        ), self.assertRaisesRegex(RuntimeError, "Previous checkpoint differs"):
            generate_previous_task_replay(request())
        self.assertEqual(model.calls, [])

    def test_checkpoint_validation_failure_propagates(self):
        with patch(
            "src.continual.diffcrl_replay.TrajectoryDiffusion.load",
            side_effect=ValueError("Diffusion checkpoint configuration differs"),
        ), self.assertRaisesRegex(ValueError, "configuration differs"):
            generate_previous_task_replay(request())


if __name__ == "__main__":
    unittest.main()
