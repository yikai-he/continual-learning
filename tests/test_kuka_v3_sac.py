"""Focused coverage for isolated KUKA-v3 single-task SAC support."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import SAC_NET_ARCH, config_from_dict, load_config
from src.env_adapters.kuka_v3 import KUKA_V3_TASKS
from src.envs import horizon_for_backend, make_env


def _sac_config(task: str) -> dict:
    return {
        "experiment": "sac",
        "environment": {"backend": "kuka-v3"},
        "continual": {"tasks": [task]},
    }


def test_kuka_v3_training_configs() -> None:
    full = load_config(
        Path("configs/sac/kuka_reach_v3_1m.yaml"), expected_experiment="sac"
    )
    assert full.environment.backend == "kuka-v3"
    assert full.continual.tasks == ["kuka-reach-v3"]
    assert full.sac.steps == 1_000_000
    assert full.sac.learning_starts == 10_000
    assert SAC_NET_ARCH == (256, 256, 256, 256)
    assert horizon_for_backend("kuka-v3") == 200


@pytest.mark.parametrize(
    ("filename", "task"),
    [
        ("kuka_button_press_topdown_v3_1m.yaml", "kuka-button-press-topdown-v3"),
        ("kuka_peg_unplug_side_v3_1m.yaml", "kuka-peg-unplug-side-v3"),
    ],
)
def test_new_kuka_v3_formal_training_configs(filename: str, task: str) -> None:
    config = load_config(Path("configs/sac") / filename, expected_experiment="sac")
    assert config.environment.backend == "kuka-v3"
    assert config.environment.reward_function_version == "v2"
    assert config.continual.tasks == [task]
    assert config.sac.steps == 1_000_000
    assert config.sac.checkpoint_freq == 50_000
    assert config.sac.early_stopping.enabled
    assert config.sac.early_stopping.min_steps == 100_000
    assert config.sac.early_stopping.success_threshold == 0.95
    assert config.sac.early_stopping.patience == 3
    assert config.evaluation.frequency == 20_000
    assert config.evaluation.episodes == 20
    assert config.runtime.seed == 0
    assert config.runtime.output == f"runs/reward_v2/{task}"


def test_kuka_push_sawyer_v2_1m_experiment_contract() -> None:
    config = load_config(
        Path("configs/sac/kuka_push_v3_sawyer_v2_1m.yaml"),
        expected_experiment="sac",
    )
    assert config.environment.backend == "kuka-v3"
    assert config.environment.reward_function_version == "v2"
    assert config.continual.tasks == ["kuka-push-v3"]
    assert config.sac.steps == 1_000_000
    assert config.sac.learning_rate == 0.001
    assert config.sac.buffer_size == 1_000_000
    assert config.sac.learning_starts == 10_000
    assert config.sac.batch_size == 128
    assert SAC_NET_ARCH == (256, 256, 256, 256)
    assert config.sac.train_freq == config.sac.gradient_steps == 1
    assert config.sac.checkpoint_freq == 50_000
    assert not config.sac.early_stopping.enabled
    assert config.sac.early_stopping.min_steps == 100_000
    assert config.sac.early_stopping.success_threshold == 0.95
    assert config.sac.early_stopping.patience == 3
    assert config.evaluation.mode == "fixed-tasks"
    assert config.evaluation.episodes == 50
    assert config.evaluation.frequency == 20_000
    assert config.evaluation.task_set_seed == 10_000
    assert config.runtime.seed == 0
    assert config.runtime.output == "runs/training_baselines/kuka-push-v3/sawyer-v2-1m"
    assert config.runtime.run_name == "seed_0_fresh_1m"


def test_kuka_push_200k_pilot_differs_only_in_budget_and_output_identity() -> None:
    baseline = load_config(
        Path("configs/sac/kuka_push_v3_sawyer_v2_1m.yaml"),
        expected_experiment="sac",
    )
    pilot = load_config(
        Path("configs/sac/kuka_push_v3_sawyer_v2_pilot_200k.yaml"),
        expected_experiment="sac",
    )
    assert pilot.environment == baseline.environment
    assert pilot.continual == baseline.continual
    assert pilot.evaluation == baseline.evaluation
    assert pilot.runtime.seed == baseline.runtime.seed == 0
    assert pilot.runtime.device == baseline.runtime.device
    assert pilot.sac == replace(baseline.sac, steps=200_000)
    assert pilot.runtime.output == "runs/training_pilots/kuka-push-v3/sawyer-v2-200k"
    assert pilot.runtime.run_name == "seed_0_fresh_200k_pilot"


def test_obsolete_push_reward_is_rejected_for_all_push_tasks() -> None:
    values = _sac_config("kuka-push-v3")
    values["environment"]["reward_function_version"] = "v3_1"
    with pytest.raises(ValueError, match="reward_function_version"):
        config_from_dict(values, expected_experiment="sac")
    values["continual"]["tasks"] = ["kuka-push-cube-v3"]
    with pytest.raises(ValueError, match="reward_function_version"):
        config_from_dict(values, expected_experiment="sac")


@pytest.mark.parametrize(
    "field",
    (
        "hammer_reward_variant",
        "hammer_nail_progress_weight",
        "peg_unplug_reward_variant",
        "peg_unplug_progress_weight",
        "peg_unplug_progress_gamma",
    ),
)
def test_removed_task_specific_reward_options_are_rejected(field: str) -> None:
    values = _sac_config("kuka-hammer-v3")
    values["environment"][field] = "removed"
    with pytest.raises(ValueError, match="is not used"):
        config_from_dict(values, expected_experiment="sac")


def test_main_hammer_configs_use_fixed_reward() -> None:
    development = load_config(
        Path("configs/sac/dev/dev_kuka_hammer_v3.yaml"), expected_experiment="sac"
    )
    formal = load_config(
        Path("configs/sac/kuka_hammer_v3_1m.yaml"), expected_experiment="sac"
    )
    for config in (development, formal):
        assert config.environment.backend == "kuka-v3"
        assert config.environment.reward_function_version == "v2"
        assert config.continual.tasks == ["kuka-hammer-v3"]
        assert config.sac.steps == 1_000_000
        assert config.sac.early_stopping.enabled
        assert config.sac.early_stopping.min_steps == 100_000
        assert config.sac.early_stopping.success_threshold == 0.95
        assert config.sac.early_stopping.patience == 3


@pytest.mark.parametrize(
    ("filename", "task"),
    [
        ("kuka_push_v3_short_50k.yaml", "kuka-push-v3"),
        ("kuka_handle_press_side_v3_short_50k.yaml", "kuka-handle-press-side-v3"),
        ("kuka_button_press_v3_short_50k.yaml", "kuka-button-press-v3"),
        ("kuka_hammer_v3_short_50k.yaml", "kuka-hammer-v3"),
        ("kuka_drawer_close_v3_short_50k.yaml", "kuka-drawer-close-v3"),
        ("kuka_faucet_open_v3_short_50k.yaml", "kuka-faucet-open-v3"),
        ("kuka_faucet_close_v3_short_50k.yaml", "kuka-faucet-close-v3"),
        ("kuka_window_open_v3_short_50k.yaml", "kuka-window-open-v3"),
        ("kuka_window_close_v3_short_50k.yaml", "kuka-window-close-v3"),
        ("kuka_door_open_v3_short_50k.yaml", "kuka-door-open-v3"),
        ("kuka_pick_place_v3_short_50k.yaml", "kuka-pick-place-v3"),
    ],
)
def test_short_configs_only_change_task_and_output_identity(
    filename: str, task: str
) -> None:
    config_path = Path("configs/sac/archive/short_probes") / filename
    config = load_config(
        config_path,
        expected_experiment="sac",
        allow_legacy_fixed_fields=True,
    )
    assert config.continual.tasks == [task]
    assert config.runtime.output == f"runs/reward_v2/{task}"
    assert config.runtime.run_name == "seed_0_short_50k"

    with Path(
        "configs/sac/archive/short_probes/kuka_reach_v3_short_50k.yaml"
    ).open() as stream:
        reach_values = yaml.safe_load(stream)
    with config_path.open() as stream:
        task_values = yaml.safe_load(stream)
    reach_values["continual"]["tasks"] = [task]
    reach_values["runtime"]["output"] = f"runs/reward_v2/{task}"
    assert task_values == reach_values


@pytest.mark.parametrize("task", KUKA_V3_TASKS)
def test_each_kuka_v3_task_is_accepted_individually(task: str) -> None:
    config = config_from_dict(_sac_config(task), expected_experiment="sac")
    assert config.continual.tasks == [task]


def test_multiple_kuka_v3_tasks_are_rejected() -> None:
    values = _sac_config("kuka-reach-v3")
    values["continual"]["tasks"] = ["kuka-reach-v3", "kuka-push-v3"]
    with pytest.raises(ValueError, match="exactly one task"):
        config_from_dict(values, expected_experiment="sac")


@pytest.mark.parametrize("task", ["kuka-unknown-v3", "push-v3"])
def test_unknown_or_mixed_kuka_v3_task_is_rejected(task: str) -> None:
    with pytest.raises(ValueError, match="kuka-v3 backend supports only"):
        config_from_dict(_sac_config(task), expected_experiment="sac")


def test_kuka_v3_is_accepted_for_diffcrl() -> None:
    values = {
        "experiment": "diffcrl",
        "environment": {"backend": "kuka-v3"},
        "continual": {
            "tasks": ["kuka-reach-v3"],
            "experts": {"kuka-reach-v3": "/tmp/expert.zip"},
        },
        "runtime": {"output": "/tmp/kuka-diffcrl-test"},
    }
    config = config_from_dict(values, expected_experiment="diffcrl")
    assert config.environment.backend == "kuka-v3"


@pytest.mark.parametrize("task", KUKA_V3_TASKS)
def test_kuka_v3_factory_contract(task: str) -> None:
    env = make_env("kuka-v3", task, 0)
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (39,)
        assert observation.dtype == np.float32
        assert env.action_space.shape == (4,)
        assert np.isfinite(observation).all()
        assert np.any(observation[36:39])
        step = env.step(env.action_space.sample())
        assert len(step) == 5
        assert np.isfinite(step[0]).all()
        assert np.isfinite(step[1])
    finally:
        env.close()
