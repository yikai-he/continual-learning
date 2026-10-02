"""Focused contracts for standalone KUKA expert collection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from gymnasium.spaces import Box
from stable_baselines3 import SAC

from src.continual.collector import collect_trajectory
from src.continual.expert_collection import (
    CollectionConfig,
    collect_expert_dataset,
    collect_successful_trajectories,
    load_collection_config,
)
from src.continual.policy_adapter import SB3SACPolicy
from src.continual.schema import Trajectory, validate_trajectory
from src.envs import GOAL_SLICE, make_env


KUKA_EXPERT_CHECKPOINT = os.environ.get("KUKA_EXPERT_CHECKPOINT")


def trajectory(*, success: bool, final_success: bool) -> Trajectory:
    observations = np.zeros((300, 39), dtype=np.float32)
    observations[:, 36:39] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    actions = np.zeros((300, 4), dtype=np.float32)
    rewards = np.ones(300, dtype=np.float64)
    return Trajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        task_id="kuka-reach-v2",
        episode_return=300.0,
        success=success,
        final_success=final_success,
        length=300,
    )


def test_success_filtered_collection_keeps_attempt_ledger_and_transient_success() -> None:
    failed = trajectory(success=False, final_success=False)
    transient = trajectory(success=True, final_success=False)
    sustained = trajectory(success=True, final_success=True)
    with patch(
        "src.continual.expert_collection.collect_trajectory",
        side_effect=[failed, transient, sustained],
    ) as collect:
        accepted, attempts = collect_successful_trajectories(
            object(),
            object(),
            "kuka-reach-v2",
            target_successes=2,
            initial_seed=40000,
        )

    assert accepted == [transient, sustained]
    assert [seed for seed, _, _ in attempts] == [40000, 40001, 40002]
    assert [accepted for _, _, accepted in attempts] == [False, True, True]
    assert accepted[0].final_success is False
    assert all(item.length == 300 for item in accepted)
    assert collect.call_count == 3
    assert [call.kwargs["seed"] for call in collect.call_args_list] == [
        40000,
        40001,
        40002,
    ]


def test_collection_configs_preserve_fixed_mode_and_define_full_contract(
    tmp_path: Path,
) -> None:
    expert = tmp_path / "expert.zip"
    expert.touch()

    def write_config(
        name: str, *, trajectories: int, seed: int, successful_only: bool
    ) -> Path:
        path = tmp_path / name
        path.write_text(
            yaml.safe_dump(
                {
                    "experiment": "trajectory_collection",
                    "environment": {
                        "backend": "kuka-v2",
                        "reward_function_version": "v2",
                    },
                    "continual": {
                        "tasks": ["kuka-reach-v2"],
                        "experts": {"kuka-reach-v2": str(expert)},
                        "trajectories_per_task": trajectories,
                    },
                    "collection": {
                        "horizon": 300,
                        "deterministic": True,
                        "successful_only": successful_only,
                    },
                    "runtime": {
                        "seed": seed,
                        "device": "auto",
                        "output": str(tmp_path / f"{name}-output"),
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    sanity_path = write_config(
        "sanity.yaml", trajectories=20, seed=30000, successful_only=False
    )
    full_path = write_config(
        "full.yaml", trajectories=200, seed=40000, successful_only=True
    )
    sanity = load_collection_config(sanity_path)
    full = load_collection_config(full_path)
    assert sanity.trajectories == 20
    assert sanity.successful_only is False
    assert full.trajectories == 200
    assert full.successful_only is True
    assert full.initial_seed == 40000
    assert full.horizon == 300
    assert full.output == (tmp_path / "full.yaml-output").resolve()
    assert full.expert == expert.resolve()


def test_saved_dataset_excludes_failure_but_ledger_retains_attempt(
    tmp_path: Path,
) -> None:
    failed = trajectory(success=False, final_success=False)
    transient = trajectory(success=True, final_success=False)
    sustained = trajectory(success=True, final_success=True)
    observation_space = Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    model = SimpleNamespace(
        observation_space=observation_space,
        action_space=action_space,
    )
    env = SimpleNamespace(
        observation_space=observation_space,
        action_space=action_space,
        spec=SimpleNamespace(max_episode_steps=300),
        close=lambda: None,
    )
    attempts = [
        (40000, failed, False),
        (40001, transient, True),
        (40002, sustained, True),
    ]
    output = tmp_path / "dataset"
    config = CollectionConfig(
        task="kuka-reach-v2",
        backend="kuka-v2",
        reward_function_version="v2",
        expert=tmp_path / "expert.zip",
        trajectories=2,
        initial_seed=40000,
        horizon=300,
        deterministic=True,
        successful_only=True,
        device="cpu",
        output=output,
    )
    with patch(
        "src.continual.expert_collection.SAC.load", return_value=model
    ), patch(
        "src.continual.expert_collection.make_env", return_value=env
    ), patch(
        "src.continual.expert_collection.collect_successful_trajectories",
        return_value=([transient, sustained], attempts),
    ):
        report = collect_expert_dataset(config)

    assert np.load(output / "observations.npy").shape == (2, 300, 39)
    assert np.load(output / "actions.npy").shape == (2, 300, 4)
    assert np.load(output / "rewards.npy").shape == (2, 300)
    with np.load(output / "real_trajectories.npz") as archive:
        assert "episode_0_observations" in archive.files
        assert "episode_1_observations" in archive.files
        assert "episode_2_observations" not in archive.files
    ledger = json.loads((output / "attempt_ledger.json").read_text())
    assert [entry["accepted"] for entry in ledger] == [False, True, True]
    assert [entry["seed"] for entry in ledger] == [40000, 40001, 40002]
    assert ledger[1]["success"] is True
    assert ledger[1]["final_success"] is False
    assert report["attempt_count"] == 3
    assert report["trajectory_count"] == 2


@pytest.mark.skipif(
    not KUKA_EXPERT_CHECKPOINT or not Path(KUKA_EXPERT_CHECKPOINT).is_file(),
    reason="set KUKA_EXPERT_CHECKPOINT to run the KUKA integration test",
)
def test_kuka_expert_collects_one_complete_valid_trajectory() -> None:
    model = SAC.load(Path(KUKA_EXPERT_CHECKPOINT), device="cpu")
    policy = SB3SACPolicy(model)
    env = make_env("kuka-v2", "kuka-reach-v2", 30000, reward_function_version="v2")
    try:
        assert model.observation_space == env.observation_space
        assert model.action_space == env.action_space
        trajectory = collect_trajectory(
            env, policy, "kuka-reach-v2", seed=30000, deterministic=True
        )
        validate_trajectory(trajectory, horizon=300)
        assert trajectory.length == 300
        assert trajectory.observations.shape == (300, 39)
        assert trajectory.actions.shape == (300, 4)
        assert trajectory.observations.dtype == np.float32
        assert np.isfinite(trajectory.observations).all()
        assert np.isfinite(trajectory.actions).all()
        assert np.all(trajectory.actions >= env.action_space.low)
        assert np.all(trajectory.actions <= env.action_space.high)
        assert np.any(trajectory.observations[0, GOAL_SLICE])
        assert np.array_equal(
            trajectory.observations[:, GOAL_SLICE],
            np.broadcast_to(trajectory.observations[0, GOAL_SLICE], (300, 3)),
        )
        assert isinstance(trajectory.success, (bool, np.bool_))
        assert isinstance(trajectory.final_success, (bool, np.bool_))
    finally:
        env.close()
