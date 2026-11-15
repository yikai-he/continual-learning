"""Contracts for the KUKA CW10-inspired DDPM/Flow Matching comparison."""

from dataclasses import replace
from pathlib import Path

from src.config import load_config
from src.continual.replay_allocation import replay_counts
from continual_manipulation.sequences import KUKA_CW10_INSPIRED_5


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/diffcrl"


def _load(method: str, seed: int):
    middle = (
        "ddpm_30current_30replay"
        if method == "ddpm"
        else "flow_matching_30current_30replay_fm10"
    )
    return load_config(CONFIG_DIR / f"kuka_cw10_inspired_5_{middle}_seed{seed}.yaml")


def test_all_six_configs_resolve_the_controlled_contract() -> None:
    for method in ("ddpm", "flow"):
        for seed in range(3):
            config = _load(method, seed)
            assert tuple(config.continual.tasks) == KUKA_CW10_INSPIRED_5
            assert config.environment.backend == "kuka-v3"
            assert config.environment.reward_function_version == "v2"
            assert config.bc.initial_log_std == -1.0
            assert config.bc.loss == "squashed-gaussian-nll"
            assert config.bc.epochs == 100
            assert config.continual.trajectories_per_task == 30
            assert config.continual.replay_allocation == "fixed-total"
            assert config.continual.replay_budget == 30
            assert config.diffusion.steps == 100
            assert config.evaluation.episodes == 50
            assert config.evaluation.mode == "fixed-tasks"
            assert config.evaluation.task_set_seed == 10_000
            assert config.runtime.seed == seed
            assert config.runtime.run_name == f"seed_{seed}"
            if method == "ddpm":
                assert config.continual.replay_mode == "diffusion"
            else:
                assert config.continual.replay_mode == "flow-matching"
                assert config.flow_matching.integration_steps == 10
                assert config.flow_matching.ode_solver == "euler"


def test_seed_configs_differ_only_in_runtime_identity() -> None:
    for method in ("ddpm", "flow"):
        base = _load(method, 0)
        for seed in (1, 2):
            candidate = _load(method, seed)
            assert candidate == replace(
                base,
                runtime=replace(base.runtime, seed=seed, run_name=f"seed_{seed}"),
            )


def test_fixed_total_replay_allocation_is_exact() -> None:
    expected = {
        1: {0: 30},
        2: {0: 15, 1: 15},
        3: {0: 10, 1: 10, 2: 10},
        4: {0: 8, 1: 8, 2: 7, 3: 7},
    }
    for stage, allocation in expected.items():
        assert replay_counts(
            range(stage),
            generator="diffusion",
            allocation="fixed-total",
            trajectories_per_task=30,
            total_budget=30,
        ) == allocation
        assert replay_counts(
            range(stage),
            generator="flow-matching",
            allocation="fixed-total",
            trajectories_per_task=30,
            total_budget=30,
        ) == allocation
