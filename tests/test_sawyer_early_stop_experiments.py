"""Contracts for the three prepared Sawyer SAC Early Stopping runs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import json
import zipfile

import numpy as np

from src.config import load_config, output_directory
from src.support.reproducibility import file_hash
from src.support.expert_manifest import load_expert_manifest
from train_sac import FixedSeedEvalCallback, validate_fixed_evaluation_preflight


ROOT = Path(__file__).resolve().parents[1]


CASES = (
    ("sawyer_stick_pull_continuation_early_stop.yaml", "stick-pull-v3", 500_000, 120_000, 1_300_000, "resume"),
    ("sawyer_push_continuation_early_stop.yaml", "push-v3", 1_000_000, 400_000, 1_540_000, "resume"),
    ("sawyer_push_back_from_faucet_close_early_stop.yaml", "push-back-v3", 1_000_000, 200_000, 560_000, "sac_actor"),
)


def test_prepared_configs_are_valid_isolated_and_hash_bound():
    outputs = set()
    for filename, task, budget, minimum, source_step, mode in CASES:
        config = load_config(ROOT / "configs/sac" / filename, expected_experiment="sac")
        assert config.continual.tasks == [task]
        assert config.sac.steps == budget
        assert config.sac.early_stopping.min_steps == minimum
        assert config.sac.early_stopping.success_threshold == 1.0
        assert config.sac.early_stopping.patience == 3
        assert config.sac.early_stopping.plateau_patience is None
        assert config.sac.early_stopping.min_delta == 0.02
        assert config.evaluation.episodes == 50
        assert config.evaluation.frequency == 40_000
        assert config.evaluation.task_set_seed == 10_000
        assert config.evaluation.final_task_set_seeds == [20_000, 20_001]
        source_name = config.resume.checkpoint if mode == "resume" else config.actor_init.checkpoint
        source = ROOT / source_name
        manifest = load_expert_manifest(source.parent / "manifest.json")
        assert file_hash(source) == manifest["checkpoint_sha256"]
        with zipfile.ZipFile(source) as archive:
            source_data = json.loads(archive.read("data"))
        assert source_data["num_timesteps"] == source_step
        outputs.add(output_directory(config))
    assert len(outputs) == 3


def test_sequential_launcher_order_and_safe_default():
    launcher = (ROOT / "tools/experiments/train_sawyer_early_stop_three.sh").read_text()
    positions = [launcher.index(name) for name in ("stick-pull-v3", "push-v3", "push-back-v3")]
    assert positions == sorted(positions)
    assert "--preflight-only|--dry-run|--run" in launcher
    assert 'case "${1:-}"' in launcher
    assert "validate_fixed_evaluation_preflight" in launcher
    assert 'evaluation_check["task_bank_size"] == 50' in launcher
    assert 'evaluation_check["task_bank_repetitions"] == 1' in launcher


def test_target_requires_three_consecutive_exactly_perfect_evaluations():
    config = load_config(
        ROOT / "configs/sac/sawyer_push_continuation_early_stop.yaml",
        expected_experiment="sac",
    )
    callback = FixedSeedEvalCallback(
        MagicMock(),
        eval_freq=config.evaluation.frequency,
        n_eval_episodes=1,
        seed=0,
        log_path=Path("evaluations"),
        best_success_model_save_path=Path("best"),
        early_stopping=config.sac.early_stopping,
    )
    callback.training_start_timestep = 0
    callback.num_timesteps = config.sac.early_stopping.min_steps
    assert callback._update_early_stopping(1.0)
    assert callback._update_early_stopping(1.0)
    assert callback._update_early_stopping(49 / 50)
    assert callback.consecutive_successful_evaluations == 0
    assert callback._update_early_stopping(1.0)
    assert callback._update_early_stopping(1.0)
    assert not callback._update_early_stopping(1.0)
    assert callback.stopping_reason == "target-success"


def test_zero_success_never_plateau_stops_and_budget_completion_is_normal():
    config = load_config(
        ROOT / "configs/sac/sawyer_push_back_from_faucet_close_early_stop.yaml",
        expected_experiment="sac",
    )
    callback = FixedSeedEvalCallback(
        MagicMock(),
        eval_freq=config.evaluation.frequency,
        n_eval_episodes=1,
        seed=0,
        log_path=Path("evaluations"),
        best_success_model_save_path=Path("best"),
        early_stopping=config.sac.early_stopping,
    )
    callback.training_start_timestep = 0
    for timestep in range(40_000, config.sac.steps + 1, 40_000):
        callback.num_timesteps = timestep
        assert callback._update_early_stopping(0.0)
    assert not callback.early_stopped
    callback.record_maximum_budget_completion(config.sac.steps)
    assert callback.stopping_reason == "maximum-training-steps"


def test_preflight_validates_one_episode_per_unique_configuration():
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.task = None

        def set_task(self, task):
            self.task = task

        def reset(self, seed=None):
            return np.asarray([self.task, seed % 2]), {}

    tasks = list(range(50))
    identities = [
        {"task_data_sha256": f"hash-{index}"} for index in range(50)
    ]
    with patch("train_sac.fixed_mt1_tasks", return_value=(tasks, identities)), patch(
        "train_sac.disable_task_sampling"
    ), patch("train_sac.validate_active_task"):
        result = validate_fixed_evaluation_preflight(
            Env(),
            task_name="push-v3",
            task_set_seed=10_000,
            episodes=50,
            first_reset_seed=20_000,
        )
    assert result["task_bank_size"] == 50
    assert result["task_bank_repetitions"] == 1
    assert result["task_bank_indices"] == tuple(range(50))
    assert result["reset_seeds"][-1] == 20_049
    assert result["same_pair_reset_reproducible"]
    assert result["second_pass_reset_seed_changed_observation"] is None


def test_preflight_rejects_nondeterministic_identical_reset_pair():
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.calls = 0

        def set_task(self, _task):
            pass

        def reset(self, seed=None):
            self.calls += 1
            return np.asarray([self.calls]), {}

    with patch(
        "train_sac.fixed_mt1_tasks",
        return_value=([object()], [{"task_data_sha256": "hash"}]),
    ), patch("train_sac.disable_task_sampling"), patch(
        "train_sac.validate_active_task"
    ):
        try:
            validate_fixed_evaluation_preflight(
                Env(),
                task_name="push-v3",
                task_set_seed=10_000,
                episodes=1,
                first_reset_seed=20_000,
            )
        except ValueError as error:
            assert "not reproducible" in str(error)
        else:
            raise AssertionError("Preflight accepted a non-reproducible reset.")
