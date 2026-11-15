"""Exact pre-refactor training and checkpoint baselines for both generators."""

from unittest.mock import patch

import pytest
import torch

from src.config import DiffusionModelConfig
from src.continual.trajectory_diffusion import (
    FeatureNormalizer,
    TrajectoryDiffusion,
    fit_diffusion,
)
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
    fit_flow_matching,
)
from src.support.reproducibility import state_hash


CHECKPOINT_KEYS = [
    "kind",
    "config",
    "state_dict",
    "dynamic_mean",
    "dynamic_scale",
    "goal_mean",
    "goal_scale",
    "metadata",
]


@pytest.fixture(scope="module", autouse=True)
def _deterministic_cpu_threads():
    """Match the single-threaded CPU setting used by DiffCRLTrainer."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _data():
    generator = torch.Generator().manual_seed(12345)
    return (
        torch.randn(5, 4, 43, generator=generator),
        torch.randn(2, 4, 43, generator=generator),
        torch.randn(5, 3, generator=generator),
        torch.randn(2, 3, generator=generator),
        torch.tensor([0, 1, 0, 1, 0]),
        torch.tensor([1, 0]),
    )


def _generator(kind):
    torch.manual_seed(77)
    if kind == "ddpm":
        return TrajectoryDiffusion(
            DiffusionModelConfig(
                horizon=4, features=43, steps=3, width=8, num_tasks=2
            )
        ), fit_diffusion
    return TrajectoryFlowMatching(
        FlowMatchingConfig(
            horizon=4,
            features=43,
            width=8,
            num_tasks=2,
            integration_steps=3,
            time_embedding_scale=20.0,
        )
    ), fit_flow_matching


@pytest.mark.parametrize("kind", ["ddpm", "flow"])
def test_training_minibatches_losses_and_sampling_are_seeded(kind):
    train, validation, train_goals, validation_goals, train_ids, validation_ids = (
        _data()
    )
    def run_once():
        model, fit = _generator(kind)
        orders = []
        original_randperm = torch.randperm

        def record_randperm(*args, **kwargs):
            value = original_randperm(*args, **kwargs)
            orders.append(value.tolist())
            return value

        with patch("torch.randperm", side_effect=record_randperm):
            result = fit(
                model,
                train,
                validation,
                train_goals,
                validation_goals,
                train_task_ids=train_ids,
                validation_task_ids=validation_ids,
                epochs=4,
                batch_size=2,
                learning_rate=0.01,
                seed=19,
            )
        sampled = model.sample(
            2,
            seed=29,
            task_ids=torch.tensor([0, 1]),
            goals=torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]]),
        )
        return orders, result, state_hash(model), sampled

    first = run_once()
    second = run_once()

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
    result = first[1]
    assert 1 <= result["best_epoch"] <= 4
    assert torch.isfinite(
        torch.tensor([result["train_loss"], result["validation_loss"]])
    ).all()

    sampled = first[3]
    assert sampled.shape == (2, 4, 43)
    assert torch.isfinite(sampled).all()


@pytest.mark.parametrize("kind", ["ddpm", "flow"])
def test_exact_normalizers_checkpoint_envelope_and_round_trip(kind, tmp_path):
    train, _, train_goals, _, _, _ = _data()
    model, _ = _generator(kind)
    dynamic = FeatureNormalizer.fit(train)
    goal = FeatureNormalizer.fit(train_goals[:, None, :])
    metadata = {
        "task_order": ["a", "b"],
        "training_banks": {},
        "train_configurations": {},
        "generated_action_projection": "none",
        "action_space": "raw",
        "action_clamp_epsilon": 1e-4,
    }
    path = tmp_path / f"{kind}.pt"

    model.save(path, dynamic, goal, metadata)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    assert list(checkpoint) == CHECKPOINT_KEYS
    assert "optimizer_state" not in checkpoint
    assert checkpoint["metadata"] == metadata
    assert dynamic.mean.shape == dynamic.scale.shape == (43,)
    assert goal.mean.shape == goal.scale.shape == (3,)

    loaded, loaded_dynamic, loaded_goal, loaded_metadata = type(model).load(path)
    assert state_hash(loaded) == state_hash(model)
    assert torch.equal(loaded_dynamic.mean, dynamic.mean)
    assert torch.equal(loaded_dynamic.scale, dynamic.scale)
    assert torch.equal(loaded_goal.mean, goal.mean)
    assert torch.equal(loaded_goal.scale, goal.scale)
    assert loaded_metadata == metadata
