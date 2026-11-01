"""Controlled KUKA ten-task GeneralPolicy architecture comparison."""

from dataclasses import asdict
from pathlib import Path

from src.config import load_config
from src.continual.replay_allocation import replay_counts


ROOT = Path(__file__).resolve().parents[1]
TWO = ROOT / "configs/diffcrl/diffcrl_kuka_10task_15plus15_2layer.yaml"
FOUR = ROOT / "configs/diffcrl/diffcrl_kuka_10task_15plus15_4layer.yaml"
TASKS = [
    "kuka-hammer-v3",
    "kuka-window-close-v3",
    "kuka-button-press-v3",
    "kuka-peg-unplug-side-v3",
    "kuka-faucet-open-v3",
    "kuka-handle-press-side-v3",
    "kuka-drawer-close-v3",
    "kuka-button-press-topdown-v3",
    "kuka-door-open-v3",
    "kuka-reach-v3",
]


def _without_architecture_and_output(config):
    value = asdict(config)
    del value["bc"]["hidden_sizes"]
    del value["runtime"]["output"]
    return value


def test_resolved_architecture_configs_differ_only_as_permitted() -> None:
    two = load_config(TWO, expected_experiment="diffcrl")
    four = load_config(FOUR, expected_experiment="diffcrl")
    assert two.bc.hidden_sizes == [256, 256]
    assert four.bc.hidden_sizes == [256, 256, 256, 256]
    assert two.runtime.output == "runs/formal/kuka_10task_15plus15_2layer/"
    assert four.runtime.output == "runs/formal/kuka_10task_15plus15_4layer/"
    assert _without_architecture_and_output(two) == _without_architecture_and_output(
        four
    )


def test_kuka_ten_task_protocol_and_experts_are_exact() -> None:
    for config in (load_config(TWO), load_config(FOUR)):
        assert config.environment.backend == "kuka-v3"
        assert config.environment.reward_function_version == "v2"
        assert config.environment.hammer_reward_variant == "nail_progress"
        assert config.environment.hammer_nail_progress_weight == 4.0
        assert (
            config.environment.peg_unplug_reward_variant
            == "peg_unplug_progress_v1"
        )
        assert config.environment.peg_unplug_progress_weight == 2.0
        assert config.environment.peg_unplug_progress_gamma == 0.99
        assert config.continual.tasks == TASKS
        assert set(config.continual.experts) == set(TASKS)
        assert all(
            path == f"runs/experts/kuka-v3/{task.removeprefix('kuka-')}/policy.zip"
            for task, path in config.continual.experts.items()
        )
        assert config.continual.trajectories_per_task == 15
        assert config.continual.replay_mode == "diffusion"
        assert config.continual.replay_allocation == "fixed-total"
        assert config.continual.replay_budget == 15
        assert config.continual.replay_priority_strategy == "uniform"
        assert config.evaluation.episodes == 50
        assert config.evaluation.seed == 10_000
        assert config.evaluation.mode == "fixed-tasks"
        assert config.evaluation.task_set_seed == 10_000
        assert config.runtime.seed == 0
        assert config.runtime.device == "cpu"


def test_every_active_replay_allocation_sums_to_fifteen() -> None:
    expected = {
        1: [15],
        2: [8, 7],
        3: [5, 5, 5],
        4: [4, 4, 4, 3],
        5: [3, 3, 3, 3, 3],
        6: [3, 3, 3, 2, 2, 2],
        7: [3, 2, 2, 2, 2, 2, 2],
        8: [2, 2, 2, 2, 2, 2, 2, 1],
        9: [2, 2, 2, 2, 2, 2, 1, 1, 1],
    }
    for stage, counts in expected.items():
        actual = replay_counts(
            range(stage),
            generator="diffusion",
            allocation="fixed-total",
            trajectories_per_task=15,
            total_budget=15,
            strategy="uniform",
        )
        assert list(actual.values()) == counts
        assert sum(actual.values()) == 15
