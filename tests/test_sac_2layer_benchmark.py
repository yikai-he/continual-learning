"""Synthetic-only validation for the ten-task SAC benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from torch import nn

from scripts.run_sac_2layer_benchmark import (
    BenchmarkEvalCallback,
    _bank_provenance,
    _hydrate,
    _write_metadata,
)
from src.config import EarlyStoppingConfig
from src.sac.transferable_actor import TransferableSACPolicy
from src.sac.two_layer_benchmark import (
    TASKS,
    aggregate_results,
    load_benchmark_config,
    stopping_reason,
    success_auc,
    validate_installed_tasks,
    write_aggregate,
)


ROOT = Path(__file__).resolve().parents[1]


class SpacesOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(39,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # pragma: no cover
        raise AssertionError("Benchmark tests must not execute an environment.")

    def step(self, action):  # pragma: no cover
        raise AssertionError("Benchmark tests must not execute an environment.")


def _model(seed=0):
    return SAC(
        TransferableSACPolicy,
        SpacesOnlyEnv(),
        policy_kwargs={"net_arch": [256, 256], "log_std_init": -3.0},
        seed=seed,
        device="cpu",
        verbose=0,
    )


def test_exact_tasks_and_resolved_configuration():
    config = load_benchmark_config(ROOT / "configs/sac_2layer_benchmark.yaml")
    assert config.tasks == TASKS
    assert config.algorithm == "SAC"
    assert config.policy.type == "TransferableSACPolicy"
    assert config.policy.net_arch == [256, 256]
    assert config.seeds == [0, 1, 2]
    assert asdict(config.training) == {
        "max_timesteps": 1_000_000, "learning_rate": 3e-4,
        "batch_size": 256, "buffer_size": 1_000_000,
        "learning_starts": 10_000, "gamma": .99, "tau": .005,
        "train_freq": 1, "gradient_steps": 1, "ent_coef": "auto",
        "device": "cpu", "torch_num_threads": 1,
    }


def test_installed_task_validation_uses_registry_without_environment():
    fake = MagicMock()
    fake.MT1.ENV_NAMES = tuple(TASKS)
    with patch.dict("sys.modules", {"metaworld": fake}):
        validate_installed_tasks()
    fake.MT1.assert_not_called()


def test_actor_and_critics_have_required_architecture_and_no_transfer():
    left, right = _model(17), _model(17)
    actor = left.policy.actor
    assert isinstance(actor.latent_pi[0], nn.Linear)
    assert isinstance(actor.latent_pi[1], nn.LayerNorm)
    assert isinstance(actor.latent_pi[2], nn.ReLU)
    assert isinstance(actor.latent_pi[3], nn.Linear)
    assert isinstance(actor.latent_pi[4], nn.ReLU)
    assert actor.mu is not actor.log_std
    assert not any(isinstance(module, nn.LayerNorm) for module in left.policy.critic.modules())
    for first, second in zip(left.policy.parameters(), right.policy.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert left.policy.actor.optimizer.state == {}
    assert left.policy.critic.optimizer.state == {}
    assert left.replay_buffer.size() == 0


def _callback(tmp_path, *, minimum=100_000, patience=3):
    callback = BenchmarkEvalCallback(
        MagicMock(), eval_freq=20_000, n_eval_episodes=1, seed=10_000,
        log_path=tmp_path, best_model_save_path=tmp_path,
        best_success_model_save_path=tmp_path,
        early_stopping=EarlyStoppingConfig(
            enabled=True, min_steps=minimum,
            success_threshold=.95, patience=patience,
        ),
    )
    callback.model = MagicMock()
    callback._logger = MagicMock()
    return callback


def test_early_stopping_counter_minimum_and_best_checkpoint_order(tmp_path):
    callback = _callback(tmp_path)
    callback._evaluate = MagicMock(side_effect=[
        ([1.0], [10], [True]), ([2.0], [10], [True]),
        ([3.0], [10], [True]),
    ])
    decisions = []
    for step in (60_000, 80_000, 100_000):
        callback.num_timesteps = step
        decisions.append(callback._record_evaluation())
    assert decisions == [True, True, False]
    assert callback.stop_timestep == 100_000
    # The qualifying model is persisted inside super() before it returns False.
    assert callback.model.save.call_args_list[5].args[0] == tmp_path / "best_success_model"


def test_maximum_step_reason_and_auc():
    assert stopping_reason(False, 1_000_000, 1_000_000) == "max_timesteps"
    assert stopping_reason(True, 100_000, 1_000_000) == "success_criterion"
    with pytest.raises(ValueError, match="ended before"):
        stopping_reason(False, 900_000, 1_000_000)
    assert success_auc([0, 20_000, 40_000], [0, .5, 1]) == pytest.approx(.5)


def test_resume_hydrates_only_history_at_or_before_checkpoint(tmp_path):
    evaluation_dir = tmp_path / "evaluations"
    evaluation_dir.mkdir()
    np.savez(
        evaluation_dir / "evaluations.npz", timesteps=[0, 20_000, 40_000],
        results=[[0.], [1.], [2.]], ep_lengths=[[10], [10], [10]],
        successes=[[False], [True], [True]], return_stddevs=[0., 0., 0.],
    )
    (evaluation_dir / "success_diagnostics.json").write_text(json.dumps([
        {"step": 0, "episodes": [{"value": 0}]},
        {"step": 20_000, "episodes": [{"value": 1}]},
        {"step": 40_000, "episodes": [{"value": 2}]},
    ]))
    callback = _callback(evaluation_dir, minimum=0, patience=3)
    settings = SimpleNamespace(
        threshold=.95, min_timesteps=0, consecutive_evaluations=3
    )
    _hydrate(callback, tmp_path, 20_000, settings)
    assert callback.timesteps == [0, 20_000]
    assert callback.consecutive_successful_evaluations == 1
    assert callback.success_diagnostics == [[{"value": 0}], [{"value": 1}]]


def test_resume_restores_an_already_triggered_stop(tmp_path):
    evaluation_dir = tmp_path / "evaluations"
    evaluation_dir.mkdir()
    np.savez(
        evaluation_dir / "evaluations.npz", timesteps=[60_000, 80_000, 100_000],
        results=[[1.], [2.], [3.]], ep_lengths=[[10], [10], [10]],
        successes=[[True], [True], [True]], return_stddevs=[0., 0., 0.],
    )
    callback = _callback(evaluation_dir, minimum=100_000, patience=3)
    settings = SimpleNamespace(
        threshold=.95, min_timesteps=100_000, consecutive_evaluations=3
    )
    _hydrate(callback, tmp_path, 100_000, settings)
    assert callback.early_stopped
    assert callback.stop_timestep == 100_000
    assert callback.trigger_success_rate == 1.0


def test_all_four_task_banks_must_be_pairwise_disjoint():
    def hashes(tasks):
        return tasks

    with patch("scripts.run_sac_2layer_benchmark.backend_tasks", side_effect=lambda task, seed: [f"h{seed}"]), patch(
        "scripts.run_sac_2layer_benchmark.bank_hashes", side_effect=hashes
    ):
        value = _bank_provenance("hammer-v3", 0, 10_000, [20_000, 20_001])
        assert value["all_banks_pairwise_disjoint"] is True
    with patch("scripts.run_sac_2layer_benchmark.backend_tasks", return_value=["same"]), patch(
        "scripts.run_sac_2layer_benchmark.bank_hashes", side_effect=hashes
    ), pytest.raises(ValueError, match="overlap"):
        _bank_provenance("hammer-v3", 0, 10_000, [20_000, 20_001])


def test_aggregation_is_read_only_and_outputs_refuse_overwrite(tmp_path):
    run = tmp_path / TASKS[0] / "seed_0"
    run.mkdir(parents=True)
    (run / "RUN_COMPLETE").write_text("complete\n")
    result = {
        "actual_training_steps": 100_000, "stopping_reason": "success_criterion",
        "best_validation_success": 1.0,
        "final_test": {"success_rate": .98}, "best_success_checkpoint": "best.zip",
        "training_duration_seconds": 12.5, "success_auc": .8,
    }
    (run / "result.json").write_text(json.dumps(result))
    rows = aggregate_results(tmp_path, [TASKS[0]], [0])
    json_path, csv_path = tmp_path / "summary.json", tmp_path / "summary.csv"
    write_aggregate(rows, json_path, csv_path)
    assert json.loads(json_path.read_text())["runs"][0]["final_test_success"] == .98
    with pytest.raises(FileExistsError):
        write_aggregate(rows, json_path, csv_path)


def test_metadata_creation_refuses_overwrite_but_owned_update_is_allowed(tmp_path):
    path = tmp_path / "metadata.json"
    _write_metadata(path, {"phase": "initial"}, initial=True)
    with pytest.raises(FileExistsError):
        _write_metadata(path, {"phase": "duplicate"}, initial=True)
    _write_metadata(path, {"phase": "complete"}, initial=False)
    assert json.loads(path.read_text()) == {"phase": "complete"}
