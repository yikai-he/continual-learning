"""Synthetic contract tests for the additional four-task SAC experiment."""

from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.sac.architecture_benchmark import (
    FOUR_TASK_1M_TASKS,
    LOW_DISK_EXPERIMENTS,
    THREE_TASK_1M_TASKS,
    load_benchmark_config,
    run_directory,
    validate_installed_tasks,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sac_4layer_4task_1m.yaml"
REFERENCE_CONFIG = ROOT / "configs/sac_4layer_3task_1m.yaml"


def test_four_task_full_budget_configuration_matches_reference_settings():
    config = load_benchmark_config(CONFIG)
    reference = load_benchmark_config(REFERENCE_CONFIG)

    assert config.tasks == FOUR_TASK_1M_TASKS == [
        "button-press-topdown-v3",
        "faucet-open-v3",
        "drawer-close-v3",
        "window-close-v3",
    ]
    assert len(config.tasks) == 4
    assert all(task.endswith("-v3") for task in config.tasks)
    assert config.seeds == [0]
    assert config.policy == reference.policy
    assert config.policy.net_arch == [256] * 4
    assert config.training == reference.training
    assert config.training.max_timesteps == 1_000_000
    assert config.training.device == "cpu"
    assert config.training.torch_num_threads == 1
    assert config.evaluation == reference.evaluation
    assert config.evaluation.frequency == 20_000
    assert config.evaluation.episodes == 50
    assert config.evaluation.validation_task_set_seed == 10_000
    assert config.evaluation.final_task_set_seeds == [20_000, 20_001]
    assert config.early_stopping == reference.early_stopping
    assert config.early_stopping.enabled is False
    assert config.checkpoint_frequency == reference.checkpoint_frequency == 50_000
    assert config.experiment in LOW_DISK_EXPERIMENTS
    assert config.output == "runs/sac_4layer_4task_1m"


def test_four_task_ids_are_supported_by_metaworld_v3_registry_without_simulation():
    fake_metaworld = MagicMock()
    fake_metaworld.MT1.ENV_NAMES = tuple(FOUR_TASK_1M_TASKS)
    with patch.dict("sys.modules", {"metaworld": fake_metaworld}):
        validate_installed_tasks(FOUR_TASK_1M_TASKS)
    fake_metaworld.MT1.assert_not_called()


def test_four_task_run_directories_are_isolated():
    config = load_benchmark_config(CONFIG)
    for task in FOUR_TASK_1M_TASKS:
        assert run_directory(config, task, 0).as_posix().endswith(
            f"runs/sac_4layer_4task_1m/{task}/seed_0"
        )


def test_existing_three_task_profile_remains_unchanged():
    config = load_benchmark_config(REFERENCE_CONFIG)
    assert config.tasks == THREE_TASK_1M_TASKS == [
        "reach-v3",
        "hammer-v3",
        "handle-press-side-v3",
    ]
    assert config.experiment == "sac_4layer_3task_1m"
    assert config.output == "runs/sac_4layer_3task_1m"
    assert asdict(config)["early_stopping"]["enabled"] is False
