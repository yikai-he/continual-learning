"""Pure report construction and DiffCRL stage-finalization boundaries."""

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.continual.diffcrl import (
    DiffCRLTrainer,
    DiffusionTrainingResult,
    StageContext,
    StageData,
)
from src.continual.diffcrl_artifacts import StageArtifacts, finish_stage
from continual_manipulation.evaluation import EvaluationMatrix
from src.continual.reporting import build_diffcrl_stage_report
from continual_manipulation.sequences import TaskSequence
from src.continual.trajectory_diffusion import FeatureNormalizer
from src.support.reproducibility import file_hash


def report_inputs(directory: Path):
    config = SimpleNamespace(
        bc=SimpleNamespace(
            loss="mse",
            initial_log_std=-3.0,
            activation="relu",
            checkpoint_selection="validation_nll",
        ),
        continual=SimpleNamespace(
            replay_mode="diffusion",
            replay_budget=200,
            replay_allocation="generator-default",
            replay_task_weights=[],
            current_trajectory_schedule=[],
        ),
        diffusion=SimpleNamespace(
            action_space="normalized",
            action_clamp_epsilon=1e-4,
            generated_action_projection="clip",
        ),
    )
    sequence = TaskSequence.from_names(["reach-v3"])
    context = StageContext(
        index=0,
        task=sequence.task(0),
        trajectory_count=5,
        directory=directory,
        policy_before_hash="policy-before",
        diffusion_before_hash="diffusion-before",
    )
    data = StageData(
        train=torch.tensor([1.0]),
        validation=torch.tensor([2.0]),
        train_labels=torch.tensor([0]),
        validation_labels=torch.tensor([0]),
        sources={
            "reach-v3": {
                "kind": "real_expert",
            }
        },
        splits={
            "reach-v3": {
                "train_ids": [0, 1, 2, 3],
                "validation_ids": [4],
            }
        },
    )
    diffusion_training = DiffusionTrainingResult(
        normalizer=FeatureNormalizer(
            mean=torch.tensor([1.0, 2.0]), scale=torch.tensor([3.0, 4.0])
        ),
        goal_normalizer=FeatureNormalizer(
            mean=torch.tensor([5.0]), scale=torch.tensor([6.0])
        ),
        losses={"train": [1.25], "validation": [2.5]},
    )
    policy_training = {"train": [3.75], "validation": [4.5], "best_epoch": 0}
    artifacts = StageArtifacts(
        diffusion_path=directory / "diffusion.pt",
        policy_path=directory / "policy.pt",
    )
    return (
        config,
        sequence,
        context,
        data,
        diffusion_training,
        policy_training,
        artifacts,
    )


def expected_report(directory: Path):
    return {
        "stage": 0,
        "task": "reach-v3",
        "bc_normalization_mode": "layer-norm",
        "bc_use_layer_norm": True,
        "bc_loss": "mse",
        "bc_initial_log_std": -3.0,
        "bc_activation": "relu",
        "bc_clamp_epsilon": 1e-4,
        "diffusion_action_space": "normalized",
        "diffusion_action_clamp_epsilon": 1e-4,
        "generated_action_projection": "clip",
        "sources": {
            "reach-v3": {
                "kind": "real_expert",
            }
        },
        "splits": {
            "reach-v3": {
                "train_ids": [0, 1, 2, 3],
                "validation_ids": [4],
            }
        },
        "diffusion_before_state_sha256": "diffusion-before",
        "flow_matching_before_state_sha256": None,
        "policy_before_state_sha256": "policy-before",
        "policy_after_state_sha256": "policy-after",
        "diffusion_checkpoint": str(directory / "diffusion.pt"),
        "diffusion_sha256": "diffusion-hash",
        "flow_matching_checkpoint": None,
        "flow_matching_sha256": None,
        "generator_mode": "diffusion",
        "generator_checkpoint": str(directory / "diffusion.pt"),
        "policy_checkpoint": str(directory / "policy.pt"),
        "policy_sha256": "policy-hash",
        "normalization_mean": [1.0, 2.0],
        "normalization_scale": [3.0, 4.0],
        "goal_normalization_mean": [5.0],
        "goal_normalization_scale": [6.0],
        "diffusion_losses": {"train": [1.25], "validation": [2.5]},
        "flow_matching_losses": None,
        "bc_losses": {"train": [3.75], "validation": [4.5], "best_epoch": 0},
        "generator_training_seconds": 0.0,
    }


class StageReportTests(unittest.TestCase):
    def test_fixed_weighted_report_records_requested_and_actual_allocation(self):
        directory = Path("/fixture/stage_2_door-open-v3")
        config, _, _, data, diffusion, policy, artifacts = report_inputs(directory)
        config.continual = SimpleNamespace(
            replay_mode="diffusion",
            replay_allocation="fixed-weighted",
            replay_budget=100,
            replay_task_weights=[1.0, 1.0, 2.0],
            trajectories_per_task=100,
            tasks=["drawer-close-v3", "window-close-v3", "door-open-v3"],
            current_trajectory_schedule=[100, 50, 50],
        )
        config.runtime = SimpleNamespace(seed=0)
        sequence = TaskSequence.from_names(config.continual.tasks)
        context = StageContext(
            index=2,
            task=sequence.task(2),
            trajectory_count=50,
            directory=directory,
            policy_before_hash="policy-before",
            diffusion_before_hash="diffusion-before",
        )
        data.sources = {
            "drawer-close-v3": {"kind": "generated", "trajectory_count": 33},
            "window-close-v3": {"kind": "generated", "trajectory_count": 67},
            "door-open-v3": {"kind": "real_expert", "trajectory_count": 100},
        }
        data.replay_allocation = {0: 33, 1: 67}
        data.splits["door-open-v3"] = {
            "trajectories": 50,
            "train_samples": 8000,
            "validation_samples": 2000,
        }

        report = build_diffcrl_stage_report(
            config,
            context,
            data,
            diffusion,
            policy,
            artifacts,
            policy_after_state_sha256="policy-after",
            diffusion_sha256="diffusion-hash",
            policy_sha256="policy-hash",
        )

        self.assertEqual(report["replay_allocation_mode"], "fixed-weighted")
        self.assertEqual(report["requested_current_trajectory_count"], 50)
        self.assertEqual(report["actual_current_trajectory_count"], 50)
        self.assertEqual(report["current_trajectory_schedule"], [100, 50, 50])
        self.assertEqual(report["runtime_seed"], 0)
        self.assertEqual(
            report["replay_allocation"],
            {"drawer-close-v3": 33, "window-close-v3": 67},
        )
        self.assertEqual(report["replay_task_weights"], [1.0, 1.0, 2.0])
        self.assertEqual(
            report["historical_tasks_considered"],
            ["drawer-close-v3", "window-close-v3"],
        )
        self.assertEqual(report["normalized_historical_weights"], [0.5, 0.5])
        self.assertEqual(
            report["actual_generated_trajectory_counts"],
            {"drawer-close-v3": 33, "window-close-v3": 67},
        )

    def test_builder_matches_exact_schema_without_mutating_inputs_or_state(self):
        directory = Path("/fixture/stage_0_reach-v3")
        inputs = report_inputs(directory)
        config, sequence, context, data, diffusion, policy, artifacts = inputs
        sources_before = copy.deepcopy(data.sources)
        splits_before = copy.deepcopy(data.splits)
        diffusion_losses_before = copy.deepcopy(diffusion.losses)
        policy_losses_before = copy.deepcopy(policy)
        tensors_before = [
            value.clone()
            for value in (
                data.train,
                data.validation,
                data.train_labels,
                data.validation_labels,
                diffusion.normalizer.mean,
                diffusion.normalizer.scale,
                diffusion.goal_normalizer.mean,
                diffusion.goal_normalizer.scale,
            )
        ]
        trainer = object.__new__(DiffCRLTrainer)
        trainer.matrix = EvaluationMatrix(sequence)
        trainer.previous_diffusion = Path("/fixture/previous.pt")
        trainer.next_stage = 0
        matrix_before = copy.deepcopy(trainer.matrix.to_dict())
        random.seed(29)
        np.random.seed(31)
        torch.manual_seed(37)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()

        report = build_diffcrl_stage_report(
            config,
            context,
            data,
            diffusion,
            policy,
            artifacts,
            policy_after_state_sha256="policy-after",
            diffusion_sha256="diffusion-hash",
            policy_sha256="policy-hash",
        )

        self.assertEqual(report, expected_report(directory))
        self.assertEqual(data.sources, sources_before)
        self.assertEqual(data.splits, splits_before)
        self.assertEqual(diffusion.losses, diffusion_losses_before)
        self.assertEqual(policy, policy_losses_before)
        for actual, before in zip(
            (
                data.train,
                data.validation,
                data.train_labels,
                data.validation_labels,
                diffusion.normalizer.mean,
                diffusion.normalizer.scale,
                diffusion.goal_normalizer.mean,
                diffusion.goal_normalizer.scale,
            ),
            tensors_before,
        ):
            torch.testing.assert_close(actual, before, rtol=0, atol=0)
        self.assertEqual(trainer.matrix.to_dict(), matrix_before)
        self.assertEqual(trainer.previous_diffusion, Path("/fixture/previous.pt"))
        self.assertEqual(trainer.next_stage, 0)
        self.assertEqual(random.getstate(), python_rng)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_rng[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
        self.assertEqual(numpy_after[2:], numpy_rng[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)

    def test_finish_persists_exact_report_and_advances_state_after_integrity_check(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            inputs = report_inputs(directory)
            config, sequence, context, data, diffusion, policy, artifacts = inputs
            artifacts.diffusion_path.write_bytes(b"diffusion checkpoint")
            artifacts.policy_path.write_bytes(b"policy checkpoint")
            expert_path = directory / "expert.pt"
            expert_path.write_bytes(b"expert checkpoint")
            trainer = object.__new__(DiffCRLTrainer)
            trainer.config = config
            trainer.policy = object()
            trainer.matrix = EvaluationMatrix(sequence)
            trainer.experts = {"reach-v3": expert_path}
            trainer.expert_hashes = {"reach-v3": file_hash(expert_path)}
            trainer.previous_diffusion = directory / "previous.pt"
            trainer.next_stage = 0
            trainer.output = directory
            trainer.training_banks = {}
            trainer.train_configurations = {}
            expected = expected_report(directory)
            expected["diffusion_sha256"] = file_hash(artifacts.diffusion_path)
            expected["policy_sha256"] = file_hash(artifacts.policy_path)

            from src.continual import diffcrl_artifacts

            real_file_hash = diffcrl_artifacts.file_hash
            with patch(
                "src.continual.diffcrl_artifacts.state_hash",
                return_value="policy-after",
            ), patch(
                "src.continual.diffcrl_artifacts.file_hash",
                wraps=real_file_hash,
            ) as hashed, patch(
                "src.continual.diffcrl_artifacts.print_stage_report"
            ) as printed:
                actual = finish_stage(
                    trainer, context, data, diffusion, policy, artifacts
                )

            self.assertEqual(actual, expected)
            expected_bytes = (json.dumps(expected, indent=2) + "\n").encode()
            self.assertEqual((directory / "report.json").read_bytes(), expected_bytes)
            matrix = trainer.matrix.to_dict()
            self.assertEqual(
                json.loads((directory / "evaluation_matrix.json").read_text()), matrix
            )
            self.assertEqual(
                (directory / "evaluation_matrix.json").read_bytes(),
                (json.dumps(matrix, indent=2) + "\n").encode(),
            )
            self.assertEqual(trainer.previous_diffusion, artifacts.diffusion_path)
            self.assertEqual(trainer.next_stage, 1)
            hashed_paths = [Path(call.args[0]) for call in hashed.call_args_list]
            self.assertEqual(hashed_paths.count(artifacts.policy_path), 1)
            self.assertEqual(hashed_paths.count(artifacts.diffusion_path), 1)
            marker = json.loads(
                (directory / "stage_0_COMPLETE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["artifacts"]["policy"]["sha256"],
                expected["policy_sha256"],
            )
            self.assertEqual(
                marker["artifacts"]["diffusion"]["sha256"],
                expected["diffusion_sha256"],
            )
            printed.assert_called_once_with(
                trainer.matrix,
                0,
                task_training_steps=None,
                global_timesteps=None,
                training_detail="Training trajectories/task: 5",
            )

    def test_integrity_failure_preserves_existing_write_and_transition_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            inputs = report_inputs(directory)
            config, sequence, context, data, diffusion, policy, artifacts = inputs
            artifacts.diffusion_path.write_bytes(b"diffusion checkpoint")
            artifacts.policy_path.write_bytes(b"policy checkpoint")
            expert_path = directory / "expert.pt"
            expert_path.write_bytes(b"changed expert")
            trainer = object.__new__(DiffCRLTrainer)
            trainer.config = config
            trainer.policy = object()
            trainer.matrix = EvaluationMatrix(sequence)
            trainer.experts = {"reach-v3": expert_path}
            trainer.expert_hashes = {"reach-v3": "original-expert-hash"}
            previous = directory / "previous.pt"
            trainer.previous_diffusion = previous
            trainer.next_stage = 0

            with patch(
                "src.continual.diffcrl_artifacts.state_hash",
                return_value="policy-after",
            ), patch(
                "src.continual.diffcrl_artifacts.print_stage_report"
            ) as printed, self.assertRaisesRegex(
                RuntimeError, "Source expert changed during experiment"
            ):
                finish_stage(trainer, context, data, diffusion, policy, artifacts)

            self.assertTrue((directory / "report.json").is_file())
            self.assertTrue((directory / "evaluation_matrix.json").is_file())
            self.assertEqual(trainer.previous_diffusion, previous)
            self.assertEqual(trainer.next_stage, 0)
            printed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
