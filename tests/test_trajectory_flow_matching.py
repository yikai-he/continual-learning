"""Unit contracts for conditional trajectory Flow Matching."""

from pathlib import Path
from types import MethodType

import pytest
import torch

from src.continual.trajectory_flow_matching import (
    CHECKPOINT_KIND,
    FlowMatchingConfig,
    TrajectoryFlowMatching,
    linear_conditional_flow,
)
from src.continual.trajectory_diffusion import FeatureNormalizer


@pytest.fixture
def config():
    return FlowMatchingConfig(
        horizon=4,
        features=22,
        width=8,
        num_tasks=2,
        integration_steps=3,
        time_embedding_scale=20.0,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(7)
    return TrajectoryFlowMatching(config)


def conditions(count=2):
    return (
        torch.arange(count, dtype=torch.long) % 2,
        torch.tensor([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]])[:count].clone(),
    )


def test_linear_interpolation_and_velocity_target_are_exact():
    clean = torch.tensor([[[2.0]], [[4.0]]])
    noise = torch.tensor([[[-2.0]], [[0.0]]])
    time = torch.tensor([0.0, 0.25], dtype=torch.float32)

    x_t, velocity = linear_conditional_flow(clean, noise, time)

    torch.testing.assert_close(x_t, torch.tensor([[[-2.0]], [[1.0]]]), rtol=0, atol=0)
    torch.testing.assert_close(
        velocity, torch.tensor([[[4.0]], [[4.0]]]), rtol=0, atol=0
    )
    endpoint, _ = linear_conditional_flow(clean, noise, torch.ones(2))
    torch.testing.assert_close(endpoint, clean, rtol=0, atol=0)


def test_loss_uses_exact_linear_path_velocity_target(model, monkeypatch):
    clean = torch.randn(2, 4, 22)
    task_ids, goals = conditions()
    captured = {}

    def forward(self, x_t, time, task_ids=None, *, goals=None):
        captured.update(x_t=x_t.clone(), time=time.clone())
        return torch.zeros_like(x_t)

    monkeypatch.setattr(model, "forward", MethodType(forward, model))
    expected_generator = torch.Generator().manual_seed(19)
    expected_time = torch.rand((2,), generator=expected_generator)
    expected_noise = torch.randn(clean.shape, generator=expected_generator)
    expected_x, expected_velocity = linear_conditional_flow(
        clean, expected_noise, expected_time
    )

    loss = model.loss(
        clean,
        generator=torch.Generator().manual_seed(19),
        task_ids=task_ids,
        goals=goals,
    )

    torch.testing.assert_close(captured["time"], expected_time, rtol=0, atol=0)
    torch.testing.assert_close(captured["x_t"], expected_x, rtol=0, atol=0)
    torch.testing.assert_close(loss, expected_velocity.square().mean())


def test_continuous_time_embedding_supports_unit_interval(model):
    embedding = model.velocity_model.time_embedding
    values = embedding(torch.tensor([0.0, 0.125, 0.5, 1.0]))
    assert values.shape == (4, model.config.width)
    assert torch.isfinite(values).all()
    assert not torch.equal(values[0], values[-1])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        embedding(torch.tensor([-0.01, 1.01]))


def test_forward_shapes_are_finite_and_conditions_affect_output(model):
    x_t = torch.randn(2, 4, 22)
    time = torch.tensor([0.2, 0.8])
    task_ids, goals = conditions()

    base = model(x_t, time, task_ids, goals=goals)
    changed_task = model(x_t, time, task_ids.flip(0), goals=goals)
    changed_goal = model(x_t, time, task_ids, goals=goals.flip(0))

    assert base.shape == x_t.shape
    assert torch.isfinite(base).all()
    assert not torch.allclose(base, changed_task)
    assert not torch.allclose(base, changed_goal)


def test_default_model_accepts_full_200_by_22_trajectories():
    model = TrajectoryFlowMatching(FlowMatchingConfig(width=4, num_tasks=1))
    output = model(
        torch.zeros(1, 200, 22),
        torch.tensor([0.5]),
        torch.zeros(1, dtype=torch.long),
        goals=torch.zeros(1, 3),
    )
    assert output.shape == (1, 200, 22)


def test_sampling_is_deterministic_finite_and_counts_exact_nfe(model):
    task_ids, goals = conditions()
    first = model.sample(2, seed=23, task_ids=task_ids, goals=goals)
    assert model.last_sample_nfe == 3
    second = model.sample(2, seed=23, task_ids=task_ids, goals=goals)
    assert model.last_sample_nfe == 3
    different = model.sample(2, seed=24, task_ids=task_ids, goals=goals)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, different)
    assert first.shape == (2, 4, 22)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize(
    ("solver", "expected_nfe"),
    (("euler", 3), ("midpoint", 6), ("heun2", 6), ("rk4", 12), ("dopri5", None)),
)
def test_supported_solvers_are_deterministic_finite_and_count_nfe(model, solver, expected_nfe):
    task_ids, goals = conditions()
    first = model.sample(
        2,
        seed=23,
        task_ids=task_ids,
        goals=goals,
        solver=solver,
    )
    first_nfe = model.last_sample_nfe
    second = model.sample(
        2,
        seed=23,
        task_ids=task_ids,
        goals=goals,
        solver=solver,
    )
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert first.device.type == "cpu"
    assert first.shape == (2, 4, 22)
    assert torch.isfinite(first).all()
    assert model.last_sample_nfe == first_nfe
    assert first_nfe > 0
    if expected_nfe is not None:
        assert first_nfe == expected_nfe


def test_euler_matches_previous_manual_integration(model):
    task_ids, goals = conditions()
    steps = 5
    expected = torch.randn((2, 4, 22), generator=torch.Generator().manual_seed(31))
    with torch.no_grad():
        for step in range(steps):
            time = torch.full((2,), step / steps, dtype=torch.float32)
            expected = expected + model.velocity_model(expected, time, task_ids, goals) / steps
    sampled = model.sample(2, seed=31, task_ids=task_ids, goals=goals, integration_steps=steps)
    torch.testing.assert_close(sampled, expected, rtol=0, atol=2e-6)
    assert model.last_sample_nfe == 5


def test_checkpoint_round_trip_includes_normalizers_and_metadata(model, tmp_path):
    dynamic = FeatureNormalizer(torch.arange(22.0), torch.ones(22))
    goal = FeatureNormalizer(torch.tensor([1.0, 2.0, 3.0]), torch.ones(3))
    metadata = {"task_order": ["reach-v3", "push-v3"], "path": "linear"}
    path = tmp_path / "flow.pt"

    model.save(path, dynamic, goal, metadata)
    loaded, loaded_dynamic, loaded_goal, loaded_metadata = (
        TrajectoryFlowMatching.load(path)
    )

    assert loaded.config == model.config
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(loaded_dynamic.mean, dynamic.mean, rtol=0, atol=0)
    torch.testing.assert_close(loaded_dynamic.scale, dynamic.scale, rtol=0, atol=0)
    torch.testing.assert_close(loaded_goal.mean, goal.mean, rtol=0, atol=0)
    assert loaded_metadata == metadata
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["kind"] == CHECKPOINT_KIND


def test_incompatible_checkpoint_kind_is_rejected(tmp_path):
    path = Path(tmp_path) / "wrong.pt"
    torch.save({"kind": "trajectory_diffusion_22_v1"}, path)
    with pytest.raises(ValueError, match=CHECKPOINT_KIND):
        TrajectoryFlowMatching.load(path)
