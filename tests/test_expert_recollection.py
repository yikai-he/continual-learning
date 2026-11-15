"""Focused contracts for fresh historical SAC-expert recollection."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.config import config_from_dict, default_config
from src.continual.diffcrl import DiffCRLTrainer, StageContext
from src.continual.expert_recollection import (
    EXPERT_RECOLLECTION_SEED_OFFSET,
    ExpertRecollectionRequest,
    collect_historical_expert_replay,
)
from continual_manipulation.evaluation.rollout import Trajectory
from continual_manipulation.sequences import TaskSequence
from continual_manipulation import EnvironmentSettings


TASKS = ("reach-v3", "push-v3")


def trajectory(task: str, value: float = 0.0) -> Trajectory:
    observations = np.zeros((200, 39), dtype=np.float32)
    observations[:, :18] = value
    observations[1:, 18:36] = observations[:-1, :18]
    observations[0, 18:36] = observations[0, :18]
    observations[:, 36:39] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return Trajectory(
        observations=observations,
        actions=np.zeros((200, 4), dtype=np.float32),
        rewards=np.ones(200),
        task_id=task,
        episode_return=200.0,
        success=True,
        final_success=False,
        length=200,
    )


def request(tmp_path, *, stage=1, counts=None):
    counts = {0: 3} if counts is None else counts
    experts = {task: tmp_path / f"{task}.zip" for task in TASKS}
    return ExpertRecollectionRequest(
        stage=stage,
        trajectory_counts=counts,
        task_names=TASKS,
        experts=experts,
        expert_hashes={task: f"hash-{task}" for task in TASKS},
        expert_identities={task: SimpleNamespace() for task in TASKS},
        environment=EnvironmentSettings(),
        runtime_seed=7,
        device="cpu",
        horizon=200,
        collection_mode="all",
        max_attempt_multiplier=2,
        progress=False,
    )


def test_recollection_uses_fresh_deterministic_rollouts_and_records_provenance(tmp_path):
    env = SimpleNamespace(close=Mock())
    captured = {}

    def collect(_env, _policy, task, **kwargs):
        captured.update(kwargs)
        configurations = []
        for attempt in range(kwargs["episodes"]):
            kwargs["on_reset"](_env, np.zeros(39), kwargs["initial_seed"] + attempt, attempt)
            configurations.append(attempt)
        episodes = [trajectory(task, float(index)) for index in configurations]
        return episodes, {
            "mode": "all",
            "expert_action_mode": "deterministic",
            "requested_trajectories": len(episodes),
            "maximum_attempts": len(episodes),
            "actual_attempts": len(episodes),
            "successful_attempts": len(episodes),
            "failed_attempts": 0,
            "accepted_successful": len(episodes),
            "accepted_unsuccessful": 0,
            "accepted_fallback": 0,
            "rejected_trajectories": 0,
            "selected_attempt_indices": configurations,
            "selected_seeds": [kwargs["initial_seed"] + i for i in configurations],
            "selected_success_fraction": 1.0,
            "selected_mean_return": 200.0,
            "selected_return_std": 0.0,
        }

    with patch("src.continual.expert_recollection.SAC.load", return_value=object()) as load, patch(
        "src.continual.expert_recollection.make_env", return_value=env
    ) as make, patch(
        "src.continual.expert_recollection.validate_expert_spaces"
    ), patch(
        "src.continual.expert_recollection.SB3SACPolicy", return_value=object()
    ), patch(
        "src.continual.expert_recollection.training_bank",
        return_value={"backend": "metaworld-v3"},
    ) as bank, patch(
        "src.continual.expert_recollection.selected_configuration",
        side_effect=lambda _env, _bank, _obs, attempt, reset_seed=None: {
            "configuration_index": attempt,
            "task_data_sha256": f"configuration-{attempt}",
            "reset_seed": reset_seed,
            "recorded_goal": [0.1, 0.2, 0.3],
        },
    ), patch(
        "src.continual.expert_recollection.collect_trajectories", side_effect=collect
    ):
        groups, sources = collect_historical_expert_replay(request(tmp_path))

    source = sources[TASKS[0]]
    expected_seed = 7 + EXPERT_RECOLLECTION_SEED_OFFSET + 10_000
    assert groups[0].shape == (3, 200, 43)
    assert captured["initial_seed"] == expected_seed
    assert captured["deterministic"] is True
    assert source["kind"] == "expert_recollection"
    assert source["checkpoint_sha256"] == f"hash-{TASKS[0]}"
    assert source["reset_seeds"] == [expected_seed, expected_seed + 1, expected_seed + 2]
    assert source["source_attempt_ids"] == [0, 1, 2]
    assert source["success_count"] == 3
    assert source["final_success_count"] == 0
    assert source["unique_configuration_count"] == 3
    assert len(source["trajectory_sha256"]) == 3
    load.assert_called_once_with(request(tmp_path).experts[TASKS[0]], device="cpu")
    make.assert_called_once_with("metaworld-v3", TASKS[0], 7, reward_function_version="v2")
    bank.assert_called_once_with(TASKS[0], 7, backend="metaworld-v3")
    env.close.assert_called_once()


def test_recollection_dispatch_and_fixed_total_allocation(tmp_path):
    config = default_config("diffcrl")
    config = replace(
        config,
        continual=replace(
            config.continual,
            tasks=list(TASKS),
            replay_mode="expert-recollection",
            replay_allocation="fixed-total",
            replay_budget=5,
        ),
    )
    trainer = object.__new__(DiffCRLTrainer)
    trainer.config = config
    trainer.sequence = TaskSequence.from_names(TASKS)
    trainer.horizon = 200
    trainer.progress = False
    trainer.output = tmp_path
    trainer.experts = {task: tmp_path / f"{task}.zip" for task in TASKS}
    trainer.expert_hashes = {task: "hash" for task in TASKS}
    trainer.expert_identities = {task: SimpleNamespace() for task in TASKS}
    trainer.environment_settings = EnvironmentSettings()
    trainer.diffusion = None
    trainer.flow_matching = None
    context = StageContext(1, trainer.sequence.task(1), 100, tmp_path, None, None)

    with patch(
        "src.continual.diffcrl.collect_historical_expert_replay",
        return_value=({}, {}),
    ) as collect:
        trainer._generate_replay(context)
    sent = collect.call_args.args[0]
    assert sent.stage == 1
    assert sent.trajectory_counts == {0: 5}
    assert sent.task_names == TASKS


def test_stage_zero_does_not_dispatch_recollection(tmp_path):
    trainer = object.__new__(DiffCRLTrainer)
    trainer.config = replace(
        default_config("diffcrl"),
        continual=replace(
            default_config("diffcrl").continual,
            replay_mode="expert-recollection",
        ),
    )
    with patch("src.continual.diffcrl.collect_historical_expert_replay") as collect:
        result = trainer._generate_replay(
            StageContext(0, TaskSequence.from_names(["reach-v3"]).task(0), 2, tmp_path, None, None)
        )
    assert result.groups == {}
    collect.assert_not_called()


def test_legacy_real_mode_is_rejected_without_reinterpretation():
    values = {
        "experiment": "diffcrl",
        "continual": {"replay_mode": "real"},
        "runtime": {"output": "runs/test"},
    }
    with pytest.raises(ValueError, match="retired stored-trajectory protocol"):
        config_from_dict(values)
