"""Synthetic contracts for the five-task KUKA SAC expert profile."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.run_sac_architecture_benchmark import _validate_backend_identity
from src.sac.architecture_benchmark import (
    KUKA_FIVE_TASK_1M_TASKS,
    LOW_DISK_EXPERIMENTS,
    environment_task_id,
    load_benchmark_config,
    run_directory,
    validate_installed_tasks,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sac_kuka_5task_1m.yaml"
SAWYER_REFERENCE = ROOT / "configs/sac_4layer_3task_1m.yaml"


def test_kuka_profile_is_fixed_full_budget_and_matches_sac_baseline():
    config = load_benchmark_config(CONFIG)
    sawyer = load_benchmark_config(SAWYER_REFERENCE)
    assert config.tasks == KUKA_FIVE_TASK_1M_TASKS
    assert len(config.tasks) == 5
    assert config.seeds == [0]
    assert config.environment.env_backend == "kuka-v3"
    assert config.environment.hammer_reward_variant == "original"
    assert config.environment.hammer_nail_progress_weight == 0.0
    assert config.policy == sawyer.policy
    assert config.policy.net_arch == [256] * 4
    assert config.training == sawyer.training
    assert config.training.max_timesteps == 1_000_000
    assert config.training.device == "cpu"
    assert config.training.torch_num_threads == 1
    assert config.evaluation == sawyer.evaluation
    assert config.early_stopping.enabled is False
    assert config.checkpoint_frequency == 50_000
    assert config.experiment in LOW_DISK_EXPERIMENTS


def test_kuka_labels_resolve_only_to_registered_kuka_ids_and_paths():
    config = load_benchmark_config(CONFIG)
    expected = [f"kuka-{task}" for task in KUKA_FIVE_TASK_1M_TASKS]
    assert [environment_task_id(config, task) for task in config.tasks] == expected
    for task in config.tasks:
        assert run_directory(config, task, 0).as_posix().endswith(
            f"runs/sac_kuka_5task_1m/{task}/seed_0"
        )


def test_kuka_registry_validation_does_not_use_standard_mt1():
    registry = {f"kuka-{task}": object for task in KUKA_FIVE_TASK_1M_TASKS}
    fake_registration = SimpleNamespace(KUKA_ENVIRONMENTS=registry)
    with patch.dict(
        "sys.modules", {"metaworld_kuka.registration": fake_registration}
    ):
        validate_installed_tasks(KUKA_FIVE_TASK_1M_TASKS, backend="kuka-v3")


def test_backend_identity_rejects_sawyer_even_with_matching_spaces():
    config = load_benchmark_config(CONFIG)
    expected = type("ExpectedKuka", (), {})
    expected.__module__ = "metaworld_kuka.envs.fake"
    sawyer = type("SawyerLookalike", (), {})
    sawyer.__module__ = "metaworld.envs.fake"
    fake_registration = SimpleNamespace(
        KUKA_ENVIRONMENTS={"kuka-reach-v3": expected}
    )
    with patch.dict(
        "sys.modules", {"metaworld_kuka.registration": fake_registration}
    ), pytest.raises(RuntimeError, match="non-KUKA"):
        _validate_backend_identity(
            SimpleNamespace(unwrapped=sawyer()), config, "kuka-reach-v3"
        )


def test_existing_sawyer_profile_remains_on_standard_backend():
    config = load_benchmark_config(SAWYER_REFERENCE)
    assert config.environment.env_backend == "metaworld-v3"
    assert environment_task_id(config, "reach-v3") == "reach-v3"
    assert config.training.max_timesteps == 1_000_000
    assert config.evaluation.frequency == 20_000
    assert config.evaluation.episodes == 50
    assert config.evaluation.final_task_set_seeds == [20_000, 20_001]
    assert config.experiment in LOW_DISK_EXPERIMENTS
    assert asdict(config)["early_stopping"]["enabled"] is False


def test_low_disk_resume_is_rejected_before_environment_creation():
    from scripts.run_sac_architecture_benchmark import run_one

    with pytest.raises(ValueError, match="cannot resume.*replay buffers"):
        run_one(load_benchmark_config(CONFIG), "reach-v3", 0, resume=True)


def test_kuka_final_test_uses_backend_and_two_fifty_episode_banks():
    from scripts.run_sac_architecture_benchmark import _final_test

    config = load_benchmark_config(CONFIG)

    def result(*args, task_set_seed, seed, backend, **kwargs):
        assert backend == "kuka-v3"
        assert args[1].task_name == "kuka-reach-v3"
        return {
            "episode_returns": [1.0] * 50,
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
