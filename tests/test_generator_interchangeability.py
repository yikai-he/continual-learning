"""Shared continual-pipeline contracts for interchangeable generators."""

from dataclasses import replace
from pathlib import Path

import torch

from src.config import REPLAY_PRIORITY_STRATEGY, default_config, load_config
from src.continual.bc_policy import GeneralPolicy
from src.continual.replay_allocation import replay_counts
from src.continual.stage_dataset import (
    bc_dataset,
    concatenate_trajectory_groups,
    prepare_trajectory_group,
    validate_trajectory_group,
)


HAMMER_FIRST_TASKS = [
    "hammer-v3",
    "button-press-v3",
    "faucet-close-v3",
    "handle-press-side-v3",
    "window-close-v3",
]


def physical_group(count: int, seed: int) -> torch.Tensor:
    values = torch.randn((count, 200, 43), generator=torch.Generator().manual_seed(seed))
    values[..., 18:36] = torch.cat(
        (values[:, :1, :18], values[:, :-1, :18]), dim=1
    )
    values[..., 36:39] = values[:, :1, 36:39]
    values[..., 39:] = values[..., 39:].tanh()
    return values.float()


def shared_config(generator: str):
    base = default_config("diffcrl")
    return replace(
        base,
        continual=replace(
            base.continual,
            replay_mode=generator,
            trajectories_per_task=50,
            replay_allocation="fixed-total",
            replay_budget=50,
        ),
    )


def test_generator_switch_preserves_shared_pipeline_configuration():
    ddpm = shared_config("diffusion")
    flow = shared_config("flow-matching")
    assert ddpm.continual.trajectories_per_task == flow.continual.trajectories_per_task
    assert ddpm.continual.replay_budget == flow.continual.replay_budget
    assert REPLAY_PRIORITY_STRATEGY == "uniform"
    assert ddpm.bc == flow.bc
    assert ddpm.evaluation == flow.evaluation

    ddpm_policy = GeneralPolicy((-torch.ones(4)).numpy(), torch.ones(4).numpy())
    flow_policy = GeneralPolicy((-torch.ones(4)).numpy(), torch.ones(4).numpy())
    assert tuple(ddpm_policy.state_dict()) == tuple(flow_policy.state_dict())
    assert sum(p.numel() for p in ddpm_policy.parameters()) == sum(
        p.numel() for p in flow_policy.parameters()
    )


def test_both_generators_build_the_same_100_trajectory_stage_dataset():
    outputs = []
    for generator in ("diffusion", "flow-matching"):
        counts = replay_counts(
            range(2),
            generator=generator,
            allocation="fixed-total",
            trajectories_per_task=50,
            total_budget=50,
        )
        groups = {
            0: physical_group(counts[0], 1),
            1: physical_group(counts[1], 2),
            2: physical_group(50, 3),
        }
        assert sum(len(group) for group in groups.values()) == 100
        preparations = {}
        for task_id, group in groups.items():
            validate_trajectory_group(group, count=len(group))
            preparations[task_id] = prepare_trajectory_group(
                group,
                count=len(group),
                seed=1000 + task_id,
                action_low=-torch.ones(4),
                action_high=torch.ones(4),
                clamp_epsilon=1e-4,
            )
        stage = concatenate_trajectory_groups(groups, preparations)
        train_bc = bc_dataset(stage.train)
        validation_bc = bc_dataset(stage.validation)
        assert train_bc.tensors[0].shape[1:] == (39,)
        assert train_bc.tensors[1].shape[1:] == (4,)
        assert validation_bc.tensors[0].shape[1:] == (39,)
        assert validation_bc.tensors[1].shape[1:] == (4,)
        outputs.append(
            (
                stage.train.shape,
                stage.validation.shape,
                train_bc.tensors[0].shape,
                validation_bc.tensors[0].shape,
            )
        )
    assert outputs[0] == outputs[1]


def test_stage_zero_has_only_50_real_trajectories():
    for generator in ("diffusion", "flow-matching"):
        assert replay_counts(
            [],
            generator=generator,
            allocation="fixed-total",
            trajectories_per_task=50,
            total_budget=0,
        ) == {}
        assert physical_group(50, 4).shape == (50, 200, 43)


def test_paired_experiment_configs_share_all_downstream_settings():
    root = Path(__file__).parents[1] / "configs" / "diffcrl"
    ddpm = load_config(root / "ddpm_5task_fixed_budget_50.yaml")
    flow = load_config(root / "flow_matching_5task_fixed_budget_50.yaml")
    assert ddpm.continual.tasks == flow.continual.tasks == HAMMER_FIRST_TASKS
    assert ddpm.continual.trajectories_per_task == 50
    assert flow.continual.trajectories_per_task == 50
    assert ddpm.continual.replay_allocation == "fixed-total"
    assert flow.continual.replay_allocation == "fixed-total"
    assert ddpm.continual.replay_budget == flow.continual.replay_budget == 50
    assert ddpm.bc == flow.bc
    assert ddpm.evaluation == flow.evaluation
    assert ddpm.diffusion.epochs == flow.flow_matching.epochs == 300
    assert ddpm.diffusion.steps == 100
    assert flow.flow_matching.integration_steps == 10
    assert "hammer_first" in ddpm.runtime.output
    assert "hammer_first" in flow.runtime.output
    assert "hammer_first" in ddpm.runtime.run_name
    assert "hammer_first" in flow.runtime.run_name


def test_paired_five_stage_smoke_configs_are_small_and_isolated():
    root = Path(__file__).parents[1] / "configs" / "diffcrl"
    ddpm = load_config(root / "ddpm_5task_fixed_budget_smoke.yaml")
    flow = load_config(root / "flow_matching_5task_fixed_budget_smoke.yaml")
    assert ddpm.continual.tasks == flow.continual.tasks == HAMMER_FIRST_TASKS
    assert ddpm.continual.experts == flow.continual.experts
    assert ddpm.continual.trajectories_per_task == 4
    assert flow.continual.trajectories_per_task == 4
    assert ddpm.continual.replay_allocation == "fixed-total"
    assert flow.continual.replay_allocation == "fixed-total"
    assert ddpm.continual.replay_budget == flow.continual.replay_budget == 4
    assert REPLAY_PRIORITY_STRATEGY == "uniform"
    assert ddpm.diffusion.epochs == flow.flow_matching.epochs == 2
    assert ddpm.bc == flow.bc
    assert ddpm.evaluation == flow.evaluation
    assert ddpm.runtime.output != flow.runtime.output
    assert "hammer_first" in ddpm.runtime.output
    assert "hammer_first" in flow.runtime.output
