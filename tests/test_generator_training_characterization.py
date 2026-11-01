"""Exact pre-refactor training and checkpoint baselines for both generators."""

import hashlib
import json
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


EXPECTED = {
    "ddpm": {
        "initial": "8fddbf256b59241da6ccb6e927cf9a9040201bcdddc50798d54d22d2c48014af",
        "best_epoch": 3,
        "result": "996c1eafe7bd3a9c2312a6c6e835009545ce994afa1969312c5027a9fb69a5d2",
        "state": "f3f4b2ebe3a1c533e78541dbfa91d70463fc41f932146eb690d17676b8c46d46",
        "sample": "9270e36319b44b5924a1938d51727465cd4a7dd621ceb35a145e6cc5bd3c2c43",
    },
    "flow": {
        "initial": "03ccf9df96e8dacef11e45839171575f75e0bd3ab3e2b6c2a8125634cb354747",
        "best_epoch": 1,
        "result": "bc6fcba160195e67dcc4a5a78cec7b5ce1f543d76f5ce35eda298fe70a7d8474",
        "state": "fe10b7f639253fcc5e0d9f3591df1bfc34a9845c2654800a31fa8b5df33d9977",
        "sample": "c38c6e3d147b7f001154923da07270c7d5f9a86325a390903d3266b2b777ea25",
    },
}

EXPECTED_ORDERS = [
    [0, 3, 1, 2, 4],
    [3, 1, 0, 4, 2],
    [0, 4, 3, 1, 2],
    [4, 2, 3, 1, 0],
]

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


def _hash_tensor(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _hash_json(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _data():
    generator = torch.Generator().manual_seed(12345)
    return (
        torch.randn(5, 4, 22, generator=generator),
        torch.randn(2, 4, 22, generator=generator),
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
                horizon=4, features=22, steps=3, width=8, num_tasks=2
            )
        ), fit_diffusion
    return TrajectoryFlowMatching(
        FlowMatchingConfig(
            horizon=4,
            features=22,
            width=8,
            num_tasks=2,
            integration_steps=3,
            time_embedding_scale=20.0,
        )
    ), fit_flow_matching


@pytest.mark.parametrize("kind", ["ddpm", "flow"])
def test_exact_training_state_minibatches_losses_and_sampling(kind):
    train, validation, train_goals, validation_goals, train_ids, validation_ids = (
        _data()
    )
    model, fit = _generator(kind)
    assert state_hash(model) == EXPECTED[kind]["initial"]
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

    assert orders == EXPECTED_ORDERS
    assert result["best_epoch"] == EXPECTED[kind]["best_epoch"]
    assert _hash_json(result) == EXPECTED[kind]["result"]
    assert state_hash(model) == EXPECTED[kind]["state"]

    sampled = model.sample(
        2,
        seed=29,
        task_ids=torch.tensor([0, 1]),
        goals=torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]]),
    )
    assert _hash_tensor(sampled) == EXPECTED[kind]["sample"]


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
    assert _hash_tensor(dynamic.mean) == (
        "6abf890b050c42c61d6ae4d15e1ced4602331b9e2f3c75f764a1a761d69da753"
    )
    assert _hash_tensor(dynamic.scale) == (
        "cebcce711d0e72eb59eb90961baaba07aff88be5b9128543f41935b4039acffb"
    )
    assert _hash_tensor(goal.mean) == (
        "353badc6e464bcad7a9b52e53d6dad99cc2179fb15690d41ad1f91c05c30bd7f"
    )
    assert _hash_tensor(goal.scale) == (
        "2c8c30576447851f179c21908f8f517f3b655d2008c235a5e3b268d357c249a8"
    )

    loaded, loaded_dynamic, loaded_goal, loaded_metadata = type(model).load(path)
    assert state_hash(loaded) == state_hash(model)
    assert torch.equal(loaded_dynamic.mean, dynamic.mean)
    assert torch.equal(loaded_dynamic.scale, dynamic.scale)
    assert torch.equal(loaded_goal.mean, goal.mean)
    assert torch.equal(loaded_goal.scale, goal.scale)
    assert loaded_metadata == metadata
