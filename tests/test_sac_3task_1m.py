"""Synthetic contract tests for the full-budget three-task SAC experiment."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.run_sac_architecture_benchmark import (
    BenchmarkEvalCallback,
    LatestModelCallback,
    _final_test,
    run_one,
)
from src.config import EarlyStoppingConfig
from src.sac.architecture_benchmark import (
    THREE_TASK_1M_TASKS,
    load_benchmark_config,
    run_directory,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sac_4layer_3task_1m.yaml"


def test_three_task_full_budget_configuration():
    config = load_benchmark_config(CONFIG)
    assert config.tasks == THREE_TASK_1M_TASKS == [
        "reach-v3", "hammer-v3", "handle-press-side-v3"
    ]
    assert config.seeds == [0]
    assert config.training.max_timesteps == 1_000_000
    assert config.training.device == "cpu"
    assert config.training.torch_num_threads == 1
    assert config.policy.net_arch == [256] * 4
    assert config.early_stopping.enabled is False
    assert run_directory(config, "hammer-v3", 0).as_posix().endswith(
        "runs/sac_4layer_3task_1m/hammer-v3/seed_0"
    )


def _disabled_callback(tmp_path):
    callback = BenchmarkEvalCallback(
        MagicMock(),
        eval_freq=20_000,
        n_eval_episodes=1,
        seed=10_000,
        log_path=tmp_path,
        best_model_save_path=tmp_path,
        best_success_model_save_path=tmp_path,
        early_stopping=EarlyStoppingConfig(
            enabled=False, min_steps=100_000, success_threshold=0.95, patience=3
        ),
    )
    callback.model = MagicMock()
    callback._logger = MagicMock()
    return callback


def test_disabled_early_stopping_never_terminates(tmp_path):
    callback = _disabled_callback(tmp_path)
    callback.num_timesteps = 1_000_000
    for _ in range(10):
        assert callback._update_early_stopping(1.0) is True
    assert callback.early_stopped is False


def test_existing_run_is_never_overwritten(tmp_path):
    config = load_benchmark_config(CONFIG)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with patch(
        "scripts.run_sac_architecture_benchmark.run_directory",
        return_value=occupied,
    ), pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_one(config, "reach-v3", 0)


def test_low_disk_resume_is_explicitly_rejected():
    config = load_benchmark_config(CONFIG)
    with pytest.raises(ValueError, match="cannot resume.*replay buffers"):
        run_one(config, "reach-v3", 0, resume=True)


class _FileModel:
    def __init__(self):
        self.saves = 0
        self.num_timesteps = 0
        self.save_replay_buffer = MagicMock()

    def save(self, path):
        self.saves += 1
        Path(path).write_bytes(f"model-{self.saves}".encode())


def test_latest_model_is_replaced_without_replay_or_history(tmp_path):
    model = _FileModel()
    callback = LatestModelCallback(50_000, tmp_path / "latest_model.zip")
    callback.model = model
    callback._on_training_start()
    callback.num_timesteps = 50_000
    assert callback._on_step()
    first = (tmp_path / "latest_model.zip").read_bytes()
    callback.num_timesteps = 100_000
    assert callback._on_step()
    assert (tmp_path / "latest_model.zip").read_bytes() != first
    assert list(tmp_path.glob("*.zip")) == [tmp_path / "latest_model.zip"]
    model.save_replay_buffer.assert_not_called()


def test_best_success_selection_and_evaluation_csv_are_retained(tmp_path):
    callback = BenchmarkEvalCallback(
        MagicMock(),
        eval_freq=20_000,
        n_eval_episodes=2,
        seed=10_000,
        log_path=tmp_path,
        best_model_save_path=tmp_path,
        best_success_model_save_path=tmp_path,
        early_stopping=EarlyStoppingConfig(enabled=False),
        save_best_return=False,
        best_success_model_name="best_model",
        atomic_model_replacement=True,
    )
    model = _FileModel()
    model.logger = MagicMock()
    callback.model = model
    callback._logger = MagicMock()
    callback._evaluate = MagicMock(
        side_effect=[
            ([1.0, 1.0], [10, 10], [True, False]),
            ([0.0, 0.0], [10, 10], [True, False]),
            ([3.0, 3.0], [10, 10], [True, False]),
            ([100.0, 100.0], [10, 10], [False, False]),
        ]
    )
    snapshots = []
    for step in (0, 20_000, 40_000, 60_000):
        callback.num_timesteps = step
        assert callback._record_evaluation()
        snapshots.append((tmp_path / "best_model.zip").read_bytes())
    assert snapshots[1] == snapshots[0]  # Equal success, lower return: no update.
    assert snapshots[2] != snapshots[1]  # Equal success, higher return: update.
    assert snapshots[3] == snapshots[2]  # Lower success cannot win on return.
    assert model.saves == 2
    assert len(callback.timesteps) == 4
    csv_rows = (tmp_path / "evaluation_history.csv").read_text().splitlines()
    assert len(csv_rows) == 5
    assert list(tmp_path.glob("*.zip")) == [tmp_path / "best_model.zip"]


def test_final_test_uses_two_disjoint_fifty_episode_banks():
    config = load_benchmark_config(CONFIG)

    def result(*args, task_set_seed, seed, **kwargs):
        return {
            "episode_returns": [float(task_set_seed)] * 50,
            "episode_successes": [True] * 50,
            "seeds": list(range(seed, seed + 50)),
            "ordered_task_hashes": [f"{task_set_seed}-{i}" for i in range(50)],
        }

    with patch(
        "scripts.run_sac_architecture_benchmark.evaluate_task", side_effect=result
    ) as evaluate:
        model = MagicMock()
        model.observation_space = SimpleNamespace(shape=(39,))
        model.action_space = SimpleNamespace(shape=(4,))
        final = _final_test(config, "reach-v3", model)
    assert evaluate.call_count == 2
    assert final["episodes"] == 100
    assert final["task_set_seeds"] == [20_000, 20_001]
    assert len(set(final["ordered_task_hashes"])) == 100
