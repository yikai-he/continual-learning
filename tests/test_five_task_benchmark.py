"""Contracts for the controlled KUKA five-task benchmark matrix."""

from dataclasses import asdict
from pathlib import Path

import pytest

from src.config import load_config
from src.continual.replay_allocation import replay_counts
from src.continual.stage_dataset import split_indices
from src.continual.trajectory_diffusion import DiffusionConfig, TrajectoryDiffusion
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
)
from continual_manipulation.environments import horizon_for_backend, make_env
from src.support.expert_manifest import validate_expert_checkpoint

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs/diffcrl/five_task_benchmark"
TASKS = [
    "kuka-drawer-close-v3",
    "kuka-window-close-v3",
    "kuka-door-open-v3",
    "kuka-peg-unplug-side-v3",
    "kuka-reach-v3",
]
VARIANTS = {
    "ddpm_cnn": ("diffusion", "temporal_cnn", 64),
    "ddpm_unet": ("diffusion", "temporal_unet1d", 32),
    "fm_cnn": ("flow-matching", "temporal_cnn", 64),
    "fm_unet": ("flow-matching", "temporal_unet1d", 32),
    "expert_recollection": ("expert-recollection", None, None),
    "no_replay": ("none", None, None),
}


def test_complete_five_task_matrix_is_strict_and_collision_free(
    tmp_path, monkeypatch
):
    # Configured outputs are relative paths. Resolve the collision assertion in
    # an isolated workspace so existing scientific runs cannot affect the test.
    monkeypatch.chdir(tmp_path)
    paths = sorted(CONFIGS.glob("kuka5_reach_*.yaml"))
    assert len(paths) == 18
    outputs = set()
    for variant, (mode, backbone, width) in VARIANTS.items():
        for seed in range(3):
            path = CONFIGS / f"kuka5_reach_{variant}_seed{seed}.yaml"
            config = load_config(path, expected_experiment="diffcrl")
            assert config.continual.tasks == TASKS
            assert config.continual.replay_mode == mode
            assert config.continual.trajectories_per_task == 100
            assert config.continual.replay_allocation == "fixed-total"
            assert config.continual.replay_budget == 100
            assert config.runtime.seed == seed
            assert config.bc.loss == "deterministic_mse"
            assert config.bc.task_conditioning == "none"
            assert (
                config.evaluation.mode,
                config.evaluation.episodes,
                config.evaluation.seed,
                config.evaluation.task_set_seed,
            ) == ("fixed-tasks", 50, 10000, 10000)
            if variant.startswith("ddpm"):
                assert (config.diffusion.backbone, config.diffusion.width) == (
                    backbone,
                    width,
                )
            if variant.startswith("fm"):
                assert (config.flow_matching.backbone, config.flow_matching.width) == (
                    backbone,
                    width,
                )
                assert (
                    config.flow_matching.ode_solver,
                    config.flow_matching.integration_steps,
                ) == ("euler", 10)
            output = Path(config.runtime.output) / config.runtime.run_name
            assert output not in outputs and not output.exists()
            outputs.add(output)


def test_reach_environment_and_strict_expert_identity():
    env = make_env(
        "kuka-v3", "kuka-reach-v3", 0, reward_function_version="v2"
    )
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == env.observation_space.shape == (39,)
        assert env.action_space.shape == (4,)
        assert horizon_for_backend("kuka-v3") == 200
    finally:
        env.close()

    identity = validate_expert_checkpoint(
        ROOT / "runs/experts/kuka-v3/reach-v3/policy.zip",
        expected_task="kuka-reach-v3",
        expected_backend="kuka-v3",
        expected_reward_function_version="v2",
        expected_observation_shape=(39,),
        expected_action_shape=(4,),
        expected_horizon=200,
        allow_legacy=False,
    )
    assert not identity.legacy
    assert identity.checkpoint_sha256 == (
        "79fdf365debd4c85da2af5544939bdc7acf57449fdf25edca1f4cf0445a2fbe9"
    )


def test_reach_configs_only_replace_fifth_task_expert_and_run_identity():
    for variant in VARIANTS:
        for seed in range(3):
            hammer = asdict(
                load_config(
                    CONFIGS / f"kuka5_{variant}_seed{seed}.yaml",
                    expected_experiment="diffcrl",
                )
            )
            reach = asdict(
                load_config(
                    CONFIGS / f"kuka5_reach_{variant}_seed{seed}.yaml",
                    expected_experiment="diffcrl",
                )
            )
            assert hammer["continual"]["tasks"][-1] == "kuka-hammer-v3"
            assert reach["continual"]["tasks"][-1] == "kuka-reach-v3"
            assert reach["continual"]["experts"]["kuka-reach-v3"] == (
                "runs/experts/kuka-v3/reach-v3/policy.zip"
            )
            reach["continual"]["tasks"] = hammer["continual"]["tasks"]
            reach["continual"]["experts"] = hammer["continual"]["experts"]
            reach["runtime"]["run_name"] = hammer["runtime"]["run_name"]
            assert reach == hammer


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (1, {0: 100}),
        (2, {0: 50, 1: 50}),
        (3, {0: 34, 1: 33, 2: 33}),
        (4, {0: 25, 1: 25, 2: 25, 3: 25}),
    ],
)
def test_exact_uniform_allocation(stage, expected):
    assert replay_counts(
        range(stage),
        generator="diffusion",
        allocation="fixed-total",
        trajectories_per_task=100,
        total_budget=100,
    ) == expected


def test_per_task_split_counts_match_stage_dataset_semantics():
    assert tuple(map(len, split_indices(100, 0))) == (80, 20)
    allocations = {
        0: [],
        1: [100],
        2: [50, 50],
        3: [34, 33, 33],
        4: [25, 25, 25, 25],
    }
    expected_totals = {
        0: (80, 20),
        1: (160, 40),
        2: (160, 40),
        3: (159, 41),
        4: (160, 40),
    }
    for stage, replay_group_sizes in allocations.items():
        group_sizes = [100, *replay_group_sizes]
        split_sizes = [tuple(map(len, split_indices(size, 0))) for size in group_sizes]
        assert tuple(map(sum, zip(*split_sizes))) == expected_totals[stage]


def test_generator_parameter_counts_for_five_task_conditioning():
    models = {
        "ddpm_cnn": TrajectoryDiffusion(DiffusionConfig(num_tasks=5)),
        "ddpm_unet": TrajectoryDiffusion(
            DiffusionConfig(num_tasks=5, backbone="temporal_unet1d", width=32)
        ),
        "fm_cnn": TrajectoryFlowMatching(FlowMatchingConfig(num_tasks=5)),
        "fm_unet": TrajectoryFlowMatching(
            FlowMatchingConfig(num_tasks=5, backbone="temporal_unet1d", width=32)
        ),
    }
    assert {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()} == {
        "ddpm_cnn": 164331,
        "ddpm_unet": 344299,
        "fm_cnn": 164331,
        "fm_unet": 344299,
    }
