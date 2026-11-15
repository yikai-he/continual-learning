"""State-isolation contracts for optional best-MSE evaluation."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.continual.bc_policy import GeneralPolicy
from src.continual.diffcrl import DiffCRLTrainer
from src.continual.diffcrl import DiffusionTrainingResult, StageContext, StageData
from src.continual.diffcrl_artifacts import (
    StageArtifacts,
    finish_stage,
    save_stage_checkpoints,
    write_stage_marker,
)
from continual_manipulation.evaluation import EvaluationMatrix
from src.continual.generator_normalization import FeatureNormalizer
from continual_manipulation.sequences import TaskSequence
from src.support.reproducibility import state_hash
from src.support.io import write_json


def test_best_mse_evaluation_is_isolated_and_uses_authoritative_protocol():
    torch.manual_seed(17)
    sequence = TaskSequence.from_names(["reach-v3"])
    trainer = object.__new__(DiffCRLTrainer)
    trainer.policy = GeneralPolicy(-np.ones(4), np.ones(4))
    trainer.policy_optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=1e-3)
    trainer.best_mse_policy_state = deepcopy(trainer.policy.state_dict())
    trainer.best_mse_policy_state["mean_head.bias"] = torch.ones(4)
    trainer.best_mse_matrix = EvaluationMatrix(sequence)
    trainer.config = SimpleNamespace(
        evaluation=SimpleNamespace(
            episodes=7, seed=1234, mode="fixed-tasks", task_set_seed=5678
        ),
        environment=SimpleNamespace(
            reward_function_version="v2", backend="metaworld-v3"
        ),
    )
    trainer.progress = False
    policy_before = state_hash(trainer.policy)
    optimizer_before = deepcopy(trainer.policy_optimizer.state_dict())
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    captured = {}

    def fake_evaluate(policy, matrix, stage, **kwargs):
        captured.update(kwargs)
        captured["alternative_hash"] = state_hash(policy)
        with torch.no_grad():
            policy.mean_head.bias.add_(10)
        torch.rand(3)
        np.random.rand(3)
        return matrix

    with patch("src.continual.diffcrl.evaluate_stage", side_effect=fake_evaluate):
        trainer._evaluate_best_mse_stage(0)

    assert state_hash(trainer.policy) == policy_before
    assert trainer.policy_optimizer.state_dict() == optimizer_before
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_state[1])
    assert numpy_after[2:] == numpy_state[2:]
    assert captured == {
        "episodes": 7,
        "seed": 1234,
        "evaluation_mode": "fixed-tasks",
        "task_set_seed": 5678,
        "reward_function_version": "v2",
        "backend": "metaworld-v3",
        "progress": False,
        "alternative_hash": captured["alternative_hash"],
    }
    assert captured["alternative_hash"] != policy_before


def test_dual_checkpoints_keep_kind_and_record_shared_fit_metadata(tmp_path):
    sequence = TaskSequence.from_names(["reach-v3"])
    coordinator = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(seed=0, run_name="dual-test"),
            bc=SimpleNamespace(
                loss="squashed_gaussian_nll",
                initial_log_std=-3.0,
                task_conditioning="none",
                hidden_dim=256,
                dual_checkpoint_selection=True,
                    checkpoint_selection="validation_nll",
            ),
            diffusion=SimpleNamespace(
                generated_action_projection="none",
                action_space="raw",
                action_clamp_epsilon=1e-4,
            ),
        ),
        policy=GeneralPolicy(-np.ones(4), np.ones(4)),
        policy_optimizer=None,
        diffusion=None,
        flow_matching=None,
        output=Path(tmp_path),
        sequence=sequence,
        training_banks={},
        train_configurations={},
    )
    coordinator.policy_optimizer = torch.optim.Adam(
        coordinator.policy.parameters(), lr=1e-3
    )
    coordinator.best_mse_policy_state = deepcopy(coordinator.policy.state_dict())
    coordinator.best_mse_policy_state["mean_head.bias"] = torch.ones(4)
    context = StageContext(
        index=0,
        task=sequence.task(0),
        trajectory_count=2,
        directory=Path(tmp_path) / "stage_0_reach-v3",
        policy_before_hash=None,
        diffusion_before_hash=None,
    )
    generator = DiffusionTrainingResult(
        normalizer=FeatureNormalizer(torch.zeros(1), torch.ones(1)),
        goal_normalizer=None,
        losses=None,
    )
    training = {
        "best_epoch": 4,
        "validation_loss": -2.0,
        "validation_action_mse": 0.08,
        "best_mse_checkpoint": {
            "epoch": 9,
            "validation_loss": 1.5,
            "validation_action_mse": 0.03,
        },
    }
    artifacts = save_stage_checkpoints(coordinator, context, generator, training)

    authoritative = torch.load(artifacts.policy_path, weights_only=True)
    alternative = torch.load(artifacts.best_mse_policy_path, weights_only=True)
    assert authoritative["kind"] == alternative["kind"]
    assert authoritative["metadata"]["checkpoint_selection"]["metric"] == "validation_nll"
    assert alternative["metadata"]["checkpoint_selection"]["metric"] == "validation_action_mse"
    assert (
        authoritative["metadata"]["checkpoint_selection"][
            "shared_training_trajectory_id"
        ]
        == alternative["metadata"]["checkpoint_selection"][
            "shared_training_trajectory_id"
        ]
    )
    assert "optimizer_state" in authoritative
    assert "optimizer_state" not in alternative
    loaded = GeneralPolicy.load(artifacts.best_mse_policy_path)
    assert state_hash(loaded) == state_hash(
        SimpleNamespace(state_dict=lambda: coordinator.best_mse_policy_state)
    )

    authoritative_directory = Path(tmp_path) / "best-mse-authoritative"
    authoritative_directory.mkdir()
    coordinator.output = authoritative_directory
    coordinator.config.bc.dual_checkpoint_selection = False
    coordinator.config.bc.checkpoint_selection = "validation_physical_mse"
    coordinator.config.runtime.run_name = "best-mse-inheritance-test"
    mse_artifacts = save_stage_checkpoints(
        coordinator, context, generator, training
    )
    mse_checkpoint = torch.load(mse_artifacts.policy_path, weights_only=True)
    selection = mse_checkpoint["metadata"]["checkpoint_selection"]
    assert mse_artifacts.best_mse_policy_path is None
    assert selection == {
        "metric": "validation_physical_mse",
        "training_objective": "squashed_gaussian_nll",
        "selected_epoch": 4,
        "validation_nll": -2.0,
        "validation_action_mse": 0.08,
        "shared_training_trajectory_id": (
            "best-mse-inheritance-test-seed0-stage0-shared-nll-fit"
        ),
    }


def test_three_stage_dual_matrix_updates_preserve_immutable_stage_artifacts(tmp_path):
    sequence = TaskSequence.from_names(
        ["reach-v3", "button-press-v3", "window-open-v3"]
    )
    expert = tmp_path / "expert.pt"
    expert.write_bytes(b"expert")
    from src.support.reproducibility import file_hash

    config = SimpleNamespace(
        bc=SimpleNamespace(
            loss="squashed_gaussian_nll",
            initial_log_std=-3.0,
            activation="relu",
            dual_checkpoint_selection=True,
            checkpoint_selection="validation_nll",
        ),
        continual=SimpleNamespace(
            replay_mode="expert-recollection",
            replay_budget=3,
            replay_allocation="fixed-total",
            trajectories_per_task=3,
            tasks=list(sequence.task_names),
            replay_task_weights=[],
            current_trajectory_schedule=[],
        ),
        diffusion=SimpleNamespace(
            action_space="raw",
            action_clamp_epsilon=1e-4,
            generated_action_projection="none",
        ),
        runtime=SimpleNamespace(device="cpu"),
    )
    trainer = SimpleNamespace(
        config=config,
        policy=GeneralPolicy(-np.ones(4), np.ones(4)),
        matrix=EvaluationMatrix(sequence),
        best_mse_matrix=EvaluationMatrix(sequence),
        experts={"expert": expert},
        expert_hashes={"expert": file_hash(expert)},
        previous_diffusion=None,
        previous_flow_matching=None,
        next_stage=0,
        output=tmp_path,
        training_banks={},
        train_configurations={},
        stage_complete_callback=None,
    )
    generator = DiffusionTrainingResult(
        normalizer=FeatureNormalizer(torch.zeros(1), torch.ones(1)),
        goal_normalizer=None,
        losses=None,
    )
    result = {
        "mean_return": 1.0,
        "std_return": 0.0,
        "success_rate": 1.0,
        "mean_episode_length": 1.0,
    }
    marker_bytes = []
    for stage, task in enumerate(sequence.tasks):
        directory = tmp_path / f"stage_{stage}_{task.task_name}"
        directory.mkdir()
        policy_path = tmp_path / f"policy_{stage}.pt"
        best_mse_policy_path = tmp_path / f"policy_best_mse_{stage}.pt"
        policy_path.write_bytes(f"nll-{stage}".encode())
        best_mse_policy_path.write_bytes(f"mse-{stage}".encode())
        artifacts = StageArtifacts(
            diffusion_path=None,
            policy_path=policy_path,
            best_mse_policy_path=best_mse_policy_path,
        )
        context = StageContext(
            index=stage,
            task=task,
            trajectory_count=3,
            directory=directory,
            policy_before_hash=f"authoritative-before-{stage}",
            diffusion_before_hash=None,
        )
        data = StageData(
            train=torch.zeros(1),
            validation=torch.zeros(1),
            train_labels=torch.zeros(1, dtype=torch.long),
            validation_labels=torch.zeros(1, dtype=torch.long),
            sources={task.task_name: {"kind": "real_expert"}},
            splits={task.task_name: {"train_ids": [0], "validation_ids": [1]}},
        )
        training = {
            "best_epoch": 2,
            "validation_loss": -1.0,
            "validation_action_mse": 0.2,
            "best_mse_checkpoint": {
                "epoch": 4,
                "validation_loss": 0.5,
                "validation_action_mse": 0.1,
            },
        }
        for task_index in range(stage + 1):
            trainer.matrix.record(stage, task_index, dict(result))
            alternative = dict(result)
            alternative["mean_return"] = float(stage * 10 + task_index)
            trainer.best_mse_matrix.record(stage, task_index, alternative)

        with patch(
            "src.continual.diffcrl_artifacts.state_hash",
            return_value=f"authoritative-after-{stage}",
        ), patch("src.continual.diffcrl_artifacts.print_stage_report"):
            finish_stage(
                trainer, context, data, generator, training, artifacts
            )

        root = EvaluationMatrix.load(tmp_path / "evaluation_matrix_best_mse.json")
        for completed_stage in range(stage + 1):
            for task_index in range(completed_stage + 1):
                assert root.cells[completed_stage][task_index]["mean_return"] == float(
                    completed_stage * 10 + task_index
                )
        local_matrix = directory / "evaluation_matrix_best_mse.json"
        with pytest.raises(FileExistsError):
            write_json(local_matrix, {})
        marker = tmp_path / f"stage_{stage}_COMPLETE.json"
        marker_bytes.append(marker.read_bytes())
        with pytest.raises(FileExistsError):
            write_stage_marker(
                tmp_path,
                stage=stage,
                task=task.task_name,
                policy_path=policy_path,
                diffusion_path=None,
                report_path=directory / "report.json",
                matrix_path=directory / "evaluation_matrix.json",
            )
        assert [
            (tmp_path / f"stage_{index}_COMPLETE.json").read_bytes()
            for index in range(stage + 1)
        ] == marker_bytes
        assert context.policy_before_hash == f"authoritative-before-{stage}"

    assert trainer.next_stage == 3


def test_run_finalization_replaces_existing_cumulative_best_mse_matrix(tmp_path):
    sequence = TaskSequence.from_names(
        ["reach-v3", "button-press-v3", "window-open-v3"]
    )
    trainer = object.__new__(DiffCRLTrainer)
    trainer.sequence = sequence
    trainer.output = Path(tmp_path)
    trainer.config = SimpleNamespace(runtime=SimpleNamespace(device="cpu"))
    trainer.matrix = EvaluationMatrix(sequence)
    trainer.best_mse_matrix = EvaluationMatrix(sequence)
    timing = {
        "expert_collection_seconds": 0.0,
        "replay_generation_seconds": 0.0,
        "generator_training_seconds": 0.0,
        "bc_training_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "total_stage_seconds": 0.0,
        "replay_neural_function_evaluations": 0,
    }
    result = {
        "mean_return": 1.0,
        "std_return": 0.0,
        "success_rate": 1.0,
        "mean_episode_length": 1.0,
    }
    for stage in range(3):
        for task_index in range(stage + 1):
            trainer.matrix.record(stage, task_index, dict(result))
            alternative = dict(result)
            alternative["mean_return"] = float(stage + task_index)
            trainer.best_mse_matrix.record(stage, task_index, alternative)
        report = tmp_path / f"report_{stage}.json"
        write_json(report, {"timing": timing})
        trainer.matrix.stage_provenance[str(stage)] = {"report": str(report)}

    write_json(
        tmp_path / "evaluation_matrix_best_mse.json",
        {"kind": "stale-cumulative-matrix"},
    )
    with patch.object(trainer, "train_task") as train_task, patch(
        "src.continual.diffcrl.print_final_report"
    ), patch("src.continual.diffcrl.complete_run"):
        returned = trainer.run()

    assert returned is trainer.matrix
    assert train_task.call_count == 3
    final_alternative = EvaluationMatrix.load(
        tmp_path / "evaluation_matrix_best_mse.json"
    )
    assert final_alternative.cells[2][2]["mean_return"] == 4.0
