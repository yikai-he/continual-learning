"""Contracts for the remaining four KUKA SAC expert tasks."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.sac.architecture_benchmark import (
    KUKA_REMAINING_FOUR_TASK_1M_TASKS,
    LOW_DISK_EXPERIMENTS,
    environment_task_id,
    load_benchmark_config,
    run_directory,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sac_kuka_remaining_4task_1m.yaml"
REFERENCE = ROOT / "configs/sac_kuka_5task_1m.yaml"


def test_remaining_profile_matches_five_task_training_contract():
    config = load_benchmark_config(CONFIG)
    reference = load_benchmark_config(REFERENCE)
    assert config.tasks == KUKA_REMAINING_FOUR_TASK_1M_TASKS == [
        "drawer-close-v3",
        "window-close-v3",
        "door-open-v3",
        "peg-unplug-side-v3",
    ]
    assert "push-v3" not in config.tasks
    assert config.seeds == [0]
    assert config.environment.env_backend == "kuka-v3"
    assert config.policy == reference.policy
    assert config.policy.net_arch == [256] * 4
    assert config.training == reference.training
    assert config.training.max_timesteps == 1_000_000
    assert config.evaluation == reference.evaluation
    assert config.early_stopping.enabled is False
    assert config.checkpoint_frequency == 50_000
    assert config.experiment in LOW_DISK_EXPERIMENTS
    assert config.output == "runs/sac_kuka_remaining_4task_1m"


def test_peg_unplug_reward_contract_is_explicit_and_default():
    environment = load_benchmark_config(CONFIG).environment
    assert environment.peg_unplug_reward_variant == "original"
    assert environment.peg_unplug_progress_weight == 0.0
    assert environment.peg_unplug_progress_gamma == 0.99


def test_all_labels_resolve_to_kuka_ids_and_isolated_paths():
    config = load_benchmark_config(CONFIG)
    for task in config.tasks:
        assert environment_task_id(config, task) == f"kuka-{task}"
        assert run_directory(config, task, 0).as_posix().endswith(
            f"runs/sac_kuka_remaining_4task_1m/{task}/seed_0"
        )


def test_low_disk_resume_is_rejected_without_environment_creation():
    from scripts.run_sac_architecture_benchmark import run_one

    with pytest.raises(ValueError, match="cannot resume.*replay buffers"):
        run_one(load_benchmark_config(CONFIG), "drawer-close-v3", 0, resume=True)


def test_final_test_uses_kuka_peg_options_and_independent_banks():
    from scripts.run_sac_architecture_benchmark import _final_test

    config = load_benchmark_config(CONFIG)

    def result(*args, task_set_seed, backend, **kwargs):
        assert backend == "kuka-v3"
        assert args[1].task_name == "kuka-peg-unplug-side-v3"
        assert kwargs["peg_unplug_reward_variant"] == "original"
        assert kwargs["peg_unplug_progress_weight"] == 0.0
        assert kwargs["peg_unplug_progress_gamma"] == 0.99
        return {
            "episode_returns": [1.0] * 50,
            "episode_successes": [True] * 50,
            "seeds": list(range(50)),
            "ordered_task_hashes": [f"{task_set_seed}-{i}" for i in range(50)],
        }

    with patch(
        "scripts.run_sac_architecture_benchmark.evaluate_task", side_effect=result
    ) as evaluate:
        model = MagicMock()
        model.observation_space = SimpleNamespace(shape=(39,))
        model.action_space = SimpleNamespace(shape=(4,))
        final = _final_test(config, "peg-unplug-side-v3", model)
    assert evaluate.call_count == 2
    assert final["episodes"] == 100
    assert final["task_set_seeds"] == [20_000, 20_001]
    assert len(set(final["ordered_task_hashes"])) == 100
