"""Unit contracts for conditional trajectory Flow Matching."""

import math
from pathlib import Path
from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from src.continual.trajectory_flow_matching import (
    BACKBONE_SPEC_VERSION,
    CHECKPOINT_KIND,
    LEGACY_CHECKPOINT_KIND,
    FlowMatchingConfig,
    TemporalUNetVelocityModel,
    TemporalVelocityModel,
    TrajectoryFlowMatching,
    fit_flow_matching,
    linear_conditional_flow,
)
from src.continual.trajectory_diffusion import FeatureNormalizer


@pytest.fixture
def config():
    return FlowMatchingConfig(
        horizon=4,
        features=43,
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


def strict_metadata():
    return {
        "task_order": ["a", "b"],
        "training_banks": {},
        "train_configurations": {},
        "generated_action_projection": "none",
        "action_space": "raw",
        "action_clamp_epsilon": 1e-4,
    }


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
    clean = torch.randn(2, 4, 43)
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
    x_t = torch.randn(2, 4, 43)
    time = torch.tensor([0.2, 0.8])
    task_ids, goals = conditions()

    base = model(x_t, time, task_ids, goals=goals)
    changed_task = model(x_t, time, task_ids.flip(0), goals=goals)
    changed_goal = model(x_t, time, task_ids, goals=goals.flip(0))

    assert base.shape == x_t.shape
    assert torch.isfinite(base).all()
    assert not torch.allclose(base, changed_task)
    assert not torch.allclose(base, changed_goal)


def test_default_model_accepts_full_200_by_43_trajectories():
    model = TrajectoryFlowMatching(FlowMatchingConfig(width=4, num_tasks=1))
    output = model(
        torch.zeros(1, 200, 43),
        torch.tensor([0.5]),
        torch.zeros(1, dtype=torch.long),
        goals=torch.zeros(1, 3),
    )
    assert output.shape == (1, 200, 43)


def test_default_cnn_matches_pre_backbone_numeric_regression():
    previous_threads = torch.get_num_threads()
    try:
        # CPU convolution and MSE reductions differ by a few ULPs with thread
        # count. Pin this strict numeric fixture to one deterministic order.
        torch.set_num_threads(1)
        torch.manual_seed(20261113)
        model = TrajectoryFlowMatching(
            FlowMatchingConfig(
                horizon=7,
                features=43,
                width=8,
                num_tasks=2,
                integration_steps=3,
                time_embedding_scale=20.0,
            )
        )
        generator = torch.Generator().manual_seed(99)
        values = torch.randn((2, 7, 43), generator=generator)
        time = torch.tensor([0.2, 0.8])
        task_ids = torch.tensor([0, 1])
        goals = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, 0.6]])
        output = model(values, time, task_ids, goals=goals)
        loss = model.loss(
            values,
            generator=torch.Generator().manual_seed(77),
            task_ids=task_ids,
            goals=goals,
        )

        assert model.config.backbone == "temporal_cnn"
        assert isinstance(model.velocity_model, TemporalVelocityModel)
        assert sum(parameter.numel() for parameter in model.parameters()) == 3531
        assert output.sum().item() == pytest.approx(-20.620710372924805, abs=1e-7)
        assert output.square().sum().sqrt().item() == pytest.approx(
            7.474085807800293, abs=1e-7
        )
        assert loss.item() == pytest.approx(2.065772294998169, abs=1e-7)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("horizon", [200, 201])
def test_unet_shapes_conditions_loss_and_backward_for_even_and_odd_horizons(horizon):
    torch.manual_seed(17)
    model = TrajectoryFlowMatching(
        FlowMatchingConfig(
            horizon=horizon,
            width=8,
            num_tasks=2,
            integration_steps=3,
            backbone="temporal_unet1d",
        )
    )
    values = torch.randn(2, horizon, 43)
    time = torch.tensor([0.25, 0.75])
    task_ids, goals = conditions()

    output = model(values, time, task_ids, goals=goals)
    changed_task = model(values, time, task_ids.flip(0), goals=goals)
    changed_goal = model(values, time, task_ids, goals=goals.flip(0))
    loss = model.loss(
        values,
        generator=torch.Generator().manual_seed(23),
        task_ids=task_ids,
        goals=goals,
    )
    loss.backward()

    assert isinstance(model.velocity_model, TemporalUNetVelocityModel)
    assert output.shape == values.shape
    assert torch.isfinite(output).all() and torch.isfinite(loss)
    assert not torch.allclose(output, changed_task)
    assert not torch.allclose(output, changed_goal)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_minimum_unet_configuration_forward_and_backward():
    model = TrajectoryFlowMatching(
        FlowMatchingConfig(
            horizon=4,
            width=4,
            num_tasks=1,
            backbone="temporal_unet1d",
        )
    )
    values = torch.randn(1, 4, 43)
    loss = model.loss(
        values,
        generator=torch.Generator().manual_seed(5),
        task_ids=torch.zeros(1, dtype=torch.long),
        goals=torch.zeros(1, 3),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_unet_shared_body_matches_pre_extraction_forward_operations():
    torch.manual_seed(29)
    model = TrajectoryFlowMatching(
        FlowMatchingConfig(
            horizon=9,
            width=8,
            num_tasks=2,
            backbone="temporal_unet1d",
        )
    ).velocity_model
    values = torch.randn(2, 9, 43)
    time = torch.tensor([0.2, 0.8])
    tasks, goals = conditions()

    actual = model(values, time, tasks, goals)
    condition = (
        model.time_embedding(time)
        + model.task_embedding(tasks)
        + model.goal_projection(goals)
    )
    projected = [layer(condition)[:, :, None] for layer in model.condition_projections]
    position = torch.linspace(-1, 1, values.shape[1])[None, None].expand(2, 1, -1)
    hidden = model.input(torch.cat((values.transpose(1, 2), position), dim=1))
    first = model.encoder_blocks[0](hidden, projected[0])
    hidden = model.downsamples[0](first)
    second = model.encoder_blocks[1](hidden, projected[1])
    hidden = model.downsamples[1](second)
    for block in model.bottleneck_blocks:
        hidden = block(hidden, projected[2])
    hidden = model.upsample_convolutions[0](
        F.interpolate(hidden, size=second.shape[-1], mode="nearest")
    )
    hidden = model.decoder_blocks[0]((hidden + second) / math.sqrt(2), projected[1])
    hidden = model.upsample_convolutions[1](
        F.interpolate(hidden, size=first.shape[-1], mode="nearest")
    )
    hidden = model.decoder_blocks[1]((hidden + first) / math.sqrt(2), projected[0])
    expected = model.output(hidden).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_horizon_below_four_is_rejected_only_for_unet():
    cnn = TrajectoryFlowMatching(FlowMatchingConfig(horizon=1, width=4))
    assert isinstance(cnn.velocity_model, TemporalVelocityModel)
    with pytest.raises(ValueError, match="Invalid flow matching configuration"):
        TrajectoryFlowMatching(
            FlowMatchingConfig(
                horizon=3, width=4, backbone="temporal_unet1d"
            )
        )


def test_unet_tiny_cpu_fit_and_sampling_smoke():
    torch.manual_seed(31)
    model = TrajectoryFlowMatching(
        FlowMatchingConfig(
            horizon=9,
            width=8,
            num_tasks=2,
            integration_steps=2,
            backbone="temporal_unet1d",
        )
    )
    train = torch.randn(4, 9, 43)
    validation = torch.randn(2, 9, 43)
    train_goals = torch.randn(4, 3)
    validation_goals = torch.randn(2, 3)
    train_ids = torch.tensor([0, 1, 0, 1])
    validation_ids = torch.tensor([0, 1])

    result = fit_flow_matching(
        model,
        train,
        validation,
        train_goals,
        validation_goals,
        train_task_ids=train_ids,
        validation_task_ids=validation_ids,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        seed=7,
    )
    sampled = model.sample(
        2,
        seed=11,
        task_ids=validation_ids,
        goals=validation_goals,
    )

    assert 1 <= result["best_epoch"] <= 2
    assert torch.isfinite(
        torch.tensor([result["train_loss"], result["validation_loss"]])
    ).all()
    assert sampled.shape == (2, 9, 43)
    assert torch.isfinite(sampled).all()
    assert model.last_sample_nfe == 2


def test_sampling_is_deterministic_finite_and_counts_exact_nfe(model):
    task_ids, goals = conditions()
    first = model.sample(2, seed=23, task_ids=task_ids, goals=goals)
    assert model.last_sample_nfe == 3
    second = model.sample(2, seed=23, task_ids=task_ids, goals=goals)
    assert model.last_sample_nfe == 3
    different = model.sample(2, seed=24, task_ids=task_ids, goals=goals)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, different)
    assert first.shape == (2, 4, 43)
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
    assert first.shape == (2, 4, 43)
    assert torch.isfinite(first).all()
    assert model.last_sample_nfe == first_nfe
    assert first_nfe > 0
    if expected_nfe is not None:
        assert first_nfe == expected_nfe


def test_euler_matches_previous_manual_integration(model):
    task_ids, goals = conditions()
    steps = 5
    expected = torch.randn((2, 4, 43), generator=torch.Generator().manual_seed(31))
    with torch.no_grad():
        for step in range(steps):
            time = torch.full((2,), step / steps, dtype=torch.float32)
            expected = expected + model.velocity_model(expected, time, task_ids, goals) / steps
    sampled = model.sample(2, seed=31, task_ids=task_ids, goals=goals, integration_steps=steps)
    torch.testing.assert_close(sampled, expected, rtol=0, atol=2e-6)
    assert model.last_sample_nfe == 5


def test_checkpoint_round_trip_includes_normalizers_and_metadata(model, tmp_path):
    dynamic = FeatureNormalizer(torch.arange(43.0), torch.ones(43))
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
    assert checkpoint["config"]["backbone"] == "temporal_cnn"
    assert checkpoint["config"]["backbone_spec_version"] == BACKBONE_SPEC_VERSION


def test_v2_checkpoint_loads_only_as_original_cnn(model, tmp_path):
    dynamic = FeatureNormalizer(torch.zeros(43), torch.ones(43))
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3))
    current = tmp_path / "current.pt"
    legacy = tmp_path / "legacy.pt"
    model.save(current, dynamic, goal, strict_metadata())
    payload = torch.load(current, map_location="cpu", weights_only=True)
    payload["kind"] = LEGACY_CHECKPOINT_KIND
    payload["config"].pop("backbone")
    payload["config"].pop("backbone_spec_version")
    torch.save(payload, legacy)

    loaded, _, _, _ = TrajectoryFlowMatching.load(
        legacy, expected_config=model.config
    )

    assert loaded.config.backbone == "temporal_cnn"
    assert loaded.checkpoint_kind == LEGACY_CHECKPOINT_KIND
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def test_v2_checkpoint_cannot_claim_unet_backbone(model, tmp_path):
    dynamic = FeatureNormalizer(torch.zeros(43), torch.ones(43))
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3))
    source = tmp_path / "source.pt"
    changed = tmp_path / "invalid-v2.pt"
    model.save(source, dynamic, goal, strict_metadata())
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["kind"] = LEGACY_CHECKPOINT_KIND
    payload["config"]["backbone"] = "temporal_unet1d"
    torch.save(payload, changed)

    with pytest.raises(ValueError, match="v2 checkpoints support temporal_cnn only"):
        TrajectoryFlowMatching.load(changed)


def test_v3_unet_round_trip_and_backbone_mismatch_rejection(tmp_path):
    config = FlowMatchingConfig(
        horizon=9,
        width=8,
        num_tasks=2,
        integration_steps=3,
        backbone="temporal_unet1d",
    )
    model = TrajectoryFlowMatching(config)
    dynamic = FeatureNormalizer(torch.zeros(43), torch.ones(43))
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3))
    path = tmp_path / "unet.pt"
    model.save(path, dynamic, goal, strict_metadata())

    loaded, _, _, _ = TrajectoryFlowMatching.load(path, expected_config=config)
    assert isinstance(loaded.velocity_model, TemporalUNetVelocityModel)
    assert loaded.checkpoint_kind == CHECKPOINT_KIND
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)

    with pytest.raises(ValueError, match="model configuration differs"):
        TrajectoryFlowMatching.load(
            path,
            expected_config=FlowMatchingConfig(
                horizon=9, width=8, num_tasks=2, integration_steps=3
            ),
        )


def test_expected_config_rejects_same_shape_different_time_scale(model, tmp_path):
    path = tmp_path / "flow.pt"
    model.save(
        path,
        FeatureNormalizer(torch.zeros(43), torch.ones(43)),
        FeatureNormalizer(torch.zeros(3), torch.ones(3)),
        strict_metadata(),
    )
    with pytest.raises(ValueError, match="model configuration differs"):
        TrajectoryFlowMatching.load(
            path,
            expected_config=FlowMatchingConfig(
                horizon=model.config.horizon,
                width=model.config.width,
                num_tasks=model.config.num_tasks,
                integration_steps=model.config.integration_steps,
                time_embedding_scale=model.config.time_embedding_scale + 1.0,
            ),
        )


@pytest.mark.parametrize("missing", ["backbone", "backbone_spec_version"])
def test_v3_checkpoint_requires_explicit_backbone_identity(model, tmp_path, missing):
    source = tmp_path / "source.pt"
    changed = tmp_path / f"missing-{missing}.pt"
    model.save(
        source,
        FeatureNormalizer(torch.zeros(43), torch.ones(43)),
        FeatureNormalizer(torch.zeros(3), torch.ones(3)),
        strict_metadata(),
    )
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["config"].pop(missing)
    torch.save(payload, changed)

    with pytest.raises(ValueError, match="lacks explicit backbone identity"):
        TrajectoryFlowMatching.load(changed)


@pytest.mark.parametrize("backbone", ["unknown", "temporal_unet1d"])
def test_checkpoint_rejects_unknown_or_state_incompatible_backbone(model, tmp_path, backbone):
    dynamic = FeatureNormalizer(torch.zeros(43), torch.ones(43))
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3))
    source = tmp_path / "source.pt"
    changed = tmp_path / f"{backbone}.pt"
    model.save(source, dynamic, goal, {})
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["config"]["backbone"] = backbone
    torch.save(payload, changed)

    message = "backbone configuration" if backbone == "unknown" else "model state"
    with pytest.raises(ValueError, match=message):
        TrajectoryFlowMatching.load(changed)


def test_incompatible_checkpoint_kind_is_rejected(tmp_path):
    path = Path(tmp_path) / "wrong.pt"
    torch.save({"kind": "trajectory_diffusion_22_v1"}, path)
    with pytest.raises(ValueError, match="legacy 22D checkpoints"):
        TrajectoryFlowMatching.load(path)
