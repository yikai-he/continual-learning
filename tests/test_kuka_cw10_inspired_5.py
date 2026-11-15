"""Configuration contracts for the CW10-inspired five-task KUKA sequence."""

import json
from pathlib import Path

import torch

from src.config import load_config
from src.config.schema import DiffusionModelConfig
from continual_manipulation.sequences import (
    KUKA_CW10_INSPIRED_5,
    resolve_named_sequence,
)
from src.continual.trajectory_diffusion import TrajectoryDiffusion
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
)
from src.support.expert_manifest import validate_expert_checkpoint
from src.support.reproducibility import file_hash


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/diffcrl/kuka_cw10_inspired_5_ddpm_seed0.yaml"
SMOKE_CONFIG = ROOT / "configs/diffcrl/kuka_cw10_inspired_5_ddpm_smoke.yaml"
REGISTRY = ROOT / "runs/experts/expert_registry.json"


def test_named_sequence_preserves_cw10_relative_order() -> None:
    expected = (
        "kuka-hammer-v3",
        "kuka-faucet-close-v3",
        "kuka-handle-press-side-v3",
        "kuka-window-close-v3",
        "kuka-peg-unplug-side-v3",
    )
    assert KUKA_CW10_INSPIRED_5 == expected
    assert resolve_named_sequence("kuka_cw10_inspired_5") == expected


def test_seed0_config_resolves_stable_bc_and_fixed_total_replay() -> None:
    config = load_config(CONFIG, expected_experiment="diffcrl")
    assert tuple(config.continual.tasks) == KUKA_CW10_INSPIRED_5
    assert config.environment.backend == "kuka-v3"
    assert config.bc.loss == "squashed-gaussian-nll"
    assert config.bc.initial_log_std == -1.0
    assert config.continual.replay_mode == "diffusion"
    assert config.continual.replay_allocation == "fixed-total"
    assert config.continual.replay_budget == 30
    assert config.evaluation.mode == "fixed-tasks"
    assert config.evaluation.task_set_seed == 10_000
    assert config.runtime.seed == 0

    smoke = load_config(SMOKE_CONFIG, expected_experiment="diffcrl")
    assert tuple(smoke.continual.tasks) == KUKA_CW10_INSPIRED_5
    assert smoke.bc.initial_log_std == -1.0
    assert smoke.continual.trajectories_per_task == 2
    assert smoke.continual.replay_allocation == "fixed-total"
    assert smoke.evaluation.episodes == 2


def test_exact_experts_strictly_authenticate_and_match_registry() -> None:
    config = load_config(CONFIG)
    registry = {
        entry["task"]: entry
        for entry in json.loads(REGISTRY.read_text(encoding="utf-8"))["experts"]
    }
    for task in KUKA_CW10_INSPIRED_5:
        checkpoint = ROOT / config.continual.experts[task]
        identity = validate_expert_checkpoint(
            checkpoint,
            expected_task=task,
            expected_backend="kuka-v3",
            expected_reward_function_version="v2",
            expected_observation_shape=(39,),
            expected_action_shape=(4,),
            expected_horizon=200,
            allow_legacy=False,
        )
        assert not identity.legacy
        assert identity.algorithm == "SAC"
        assert registry[task]["checkpoint"] == config.continual.experts[task]
        assert registry[task]["sha256"] == file_hash(checkpoint)


def test_both_generators_represent_all_five_task_conditions() -> None:
    task_ids = torch.arange(len(KUKA_CW10_INSPIRED_5), dtype=torch.long)
    ddpm = TrajectoryDiffusion(DiffusionModelConfig(num_tasks=5))
    flow = TrajectoryFlowMatching(FlowMatchingConfig(num_tasks=5))
    ddpm.validate_tasks(task_ids, count=5)
    flow.validate_tasks(task_ids, count=5)
