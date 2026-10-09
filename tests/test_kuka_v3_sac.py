"""Focused coverage for isolated KUKA-v3 single-task SAC support."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import config_from_dict, load_config
from src.env_adapters.kuka_v3 import KUKA_V3_TASKS
from src.envs import horizon_for_backend, make_env


def _sac_config(task: str) -> dict:
    return {
        "experiment": "sac",
        "environment": {"backend": "kuka-v3"},
        "continual": {"tasks": [task]},
    }


def test_kuka_v3_training_configs() -> None:
    short = load_config(
        Path("configs/sac/kuka_reach_v3_short_50k.yaml"),
        expected_experiment="sac",
    )
    full = load_config(
        Path("configs/sac/kuka_reach_v3_1m.yaml"), expected_experiment="sac"
    )
    assert short.environment.backend == full.environment.backend == "kuka-v3"
    assert short.continual.tasks == full.continual.tasks == ["kuka-reach-v3"]
    assert short.sac.steps == 50_000
    assert full.sac.steps == 1_000_000
    assert short.sac.learning_starts == full.sac.learning_starts == 10_000
    assert short.sac.net_arch == full.sac.net_arch == [256, 256, 256, 256]
    assert horizon_for_backend("kuka-v3") == 200


def test_kuka_push_reward_v3_probe_matches_v2_sac_settings() -> None:
    v2 = load_config(
        Path("configs/sac/kuka_push_v3_short_50k.yaml"),
        expected_experiment="sac",
    )
    v3 = load_config(
        Path("configs/sac/kuka_push_v3_reward_v3_50k.yaml"),
        expected_experiment="sac",
    )
    assert v3.environment.backend == v2.environment.backend == "kuka-v3"
    assert v3.environment.reward_function_version == "v3"
    assert v2.environment.reward_function_version == "v2"
    assert v3.sac == v2.sac
    assert v3.continual == v2.continual
    assert v3.evaluation == v2.evaluation
    assert v3.runtime.seed == v2.runtime.seed == 0
    assert v3.runtime.device == v2.runtime.device == "auto"
    assert v3.runtime.output == "runs/reward_v3/kuka-push-v3"
    assert v3.runtime.run_name == "seed_0_50k_probe"


def test_kuka_push_reward_v3_1_probe_only_changes_reward_identity() -> None:
    v3 = load_config(
        Path("configs/sac/kuka_push_v3_reward_v3_50k.yaml"),
        expected_experiment="sac",
    )
    v3_1 = load_config(
        Path("configs/sac/kuka_push_v3_reward_v3_1_50k.yaml"),
        expected_experiment="sac",
    )
    assert v3_1.environment.backend == v3.environment.backend == "kuka-v3"
    assert v3_1.environment.reward_function_version == "v3_1"
    assert v3.environment.reward_function_version == "v3"
    assert v3_1.sac == v3.sac
    assert v3_1.continual == v3.continual
    assert v3_1.evaluation == v3.evaluation
    assert v3_1.runtime.seed == v3.runtime.seed == 0
    assert v3_1.runtime.device == v3.runtime.device == "auto"
    assert v3_1.runtime.output == "runs/reward_v3_1/kuka-push-v3"
    assert v3_1.runtime.run_name == "seed_0_50k_probe"


def test_hammer_reward_ablation_configs_are_controlled() -> None:
    paths = [
        Path("configs/sac/kuka_hammer_v3_ablation_original_400k.yaml"),
        Path("configs/sac/kuka_hammer_v3_ablation_progress_w2_400k.yaml"),
        Path("configs/sac/kuka_hammer_v3_ablation_progress_w4_400k.yaml"),
    ]
    control, weight_2, weight_4 = [
        load_config(path, expected_experiment="sac") for path in paths
    ]
    assert [config.sac.steps for config in (control, weight_2, weight_4)] == [
        400_000
    ] * 3
    assert control.sac == weight_2.sac == weight_4.sac
    assert control.continual == weight_2.continual == weight_4.continual
    assert control.evaluation == weight_2.evaluation == weight_4.evaluation
    assert control.evaluation.mode == "fixed-tasks"
    assert control.evaluation.task_set_seed == 10_000
    assert control.runtime.seed == weight_2.runtime.seed == weight_4.runtime.seed == 0
    assert control.environment.hammer_reward_variant == "original"
    assert control.environment.hammer_nail_progress_weight == 0.0
    assert weight_2.environment.hammer_reward_variant == "nail_progress"
    assert weight_2.environment.hammer_nail_progress_weight == 2.0
    assert weight_4.environment.hammer_reward_variant == "nail_progress"
    assert weight_4.environment.hammer_nail_progress_weight == 4.0


def test_main_hammer_configs_select_progress_weight_4() -> None:
    development = load_config(
        Path("configs/sac/dev_kuka_hammer_v3.yaml"), expected_experiment="sac"
    )
    formal = load_config(
        Path("configs/sac/kuka_hammer_v3_1m.yaml"), expected_experiment="sac"
    )
    for config in (development, formal):
        assert config.environment.backend == "kuka-v3"
        assert config.environment.reward_function_version == "v2"
        assert config.environment.hammer_reward_variant == "nail_progress"
        assert config.environment.hammer_nail_progress_weight == 4.0
        assert config.continual.tasks == ["kuka-hammer-v3"]
        assert config.sac.steps == 1_000_000
        assert config.sac.early_stopping.enabled
        assert config.sac.early_stopping.min_steps == 100_000
        assert config.sac.early_stopping.success_threshold == 0.95
        assert config.sac.early_stopping.patience == 3


def test_original_hammer_reward_remains_explicitly_selectable() -> None:
    config = load_config(
        Path("configs/sac/kuka_hammer_v3_ablation_original_250k.yaml"),
        expected_experiment="sac",
    )
    assert config.environment.hammer_reward_variant == "original"
    assert config.environment.hammer_nail_progress_weight == 0.0


def test_hammer_reward_options_rejected_for_other_tasks() -> None:
    values = _sac_config("kuka-reach-v3")
    values["environment"].update(
        hammer_reward_variant="nail_progress", hammer_nail_progress_weight=2.0
    )
    with pytest.raises(ValueError, match="require KUKA-v3 Hammer SAC"):
        config_from_dict(values, expected_experiment="sac")


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
    config_path = Path("configs/sac") / filename
    config = load_config(config_path, expected_experiment="sac")
    assert config.continual.tasks == [task]
    assert config.runtime.output == f"runs/reward_v2/{task}"
    assert config.runtime.run_name == "seed_0_short_50k"

    with Path("configs/sac/kuka_reach_v3_short_50k.yaml").open() as stream:
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
