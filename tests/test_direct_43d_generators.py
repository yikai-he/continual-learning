"""Shared contracts for the only supported direct-43D generators."""

import pytest
import torch

from src.config import DiffusionModelConfig
from src.continual.generator_normalization import FeatureNormalizer
from src.continual.trajectory_diffusion import TrajectoryDiffusion
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
)


@pytest.mark.parametrize("kind", ["ddpm", "flow"])
def test_generators_train_and_sample_complete_43d_trajectories(kind):
    clean = torch.randn(2, 3, 43)
    labels = torch.tensor([0, 1], dtype=torch.long)
    goals = torch.randn(2, 3)
    if kind == "ddpm":
        model = TrajectoryDiffusion(
            DiffusionModelConfig(
                horizon=3, features=43, steps=2, width=8, num_tasks=2
            )
        )
    else:
        model = TrajectoryFlowMatching(
            FlowMatchingConfig(
                horizon=3,
                features=43,
                width=8,
                num_tasks=2,
                integration_steps=2,
            )
        )

    loss = model.loss(
        clean,
        generator=torch.Generator().manual_seed(7),
        task_ids=labels,
        goals=goals,
    )
    sample = model.sample(2, seed=9, task_ids=labels, goals=goals)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert sample.shape == (2, 3, 43)
    assert torch.isfinite(sample).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TrajectoryDiffusion(DiffusionModelConfig(features=22)),
        lambda: TrajectoryFlowMatching(FlowMatchingConfig(features=40)),
    ],
)
def test_reduced_feature_models_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("kind", ["ddpm", "flow"])
def test_direct_43d_checkpoint_round_trip(kind, tmp_path):
    dynamic = FeatureNormalizer(torch.zeros(43), torch.ones(43))
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3))
    metadata = {
        "task_order": ["task"],
        "training_banks": {},
        "train_configurations": {},
        "generated_action_projection": "none",
        "action_space": "raw",
        "action_clamp_epsilon": 1e-4,
    }
    if kind == "ddpm":
        model = TrajectoryDiffusion(
            DiffusionModelConfig(
                horizon=3, features=43, steps=2, width=8, num_tasks=1
            )
        )
    else:
        model = TrajectoryFlowMatching(
            FlowMatchingConfig(horizon=3, features=43, width=8, num_tasks=1)
        )
    path = tmp_path / f"{kind}.pt"

    model.save(path, dynamic, goal, metadata)
    loaded, loaded_dynamic, loaded_goal, loaded_metadata = type(model).load(path)

    assert loaded.config.features == 43
    torch.testing.assert_close(loaded_dynamic.mean, dynamic.mean)
    torch.testing.assert_close(loaded_dynamic.scale, dynamic.scale)
    torch.testing.assert_close(loaded_goal.mean, goal.mean)
    assert loaded_metadata == metadata
