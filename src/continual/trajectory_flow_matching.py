"""Conditional Flow Matching for normalized 22D full-trajectory generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from src.support.io import atomic_torch_save
from torch import nn
from torchdiffeq import odeint

from .diffusion_data import DIFFUSION_FEATURES
from .generator_normalization import FeatureNormalizer
from .generator_training import (
    fit_generator,
    generator_checkpoint_payload,
    run_cpu_inputs_on_module_device,
)

CHECKPOINT_KIND = "trajectory_flow_matching_22_v1"


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Serialized dimensions and numerical settings for trajectory flow matching."""

    horizon: int = 200
    features: int = DIFFUSION_FEATURES
    width: int = 64
    num_tasks: int = 0
    integration_steps: int = 100
    time_embedding_scale: float = 100.0


def _finite_trajectory_batch(x: torch.Tensor) -> None:
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 3
        or min(x.shape) < 1
        or not x.is_floating_point()
        or x.device.type != "cpu"
        or not torch.isfinite(x).all()
    ):
        raise ValueError("Expected a nonempty finite floating CPU tensor (N,T,F).")


def _continuous_time(time: torch.Tensor, count: int) -> None:
    if (
        not isinstance(time, torch.Tensor)
        or time.shape != (count,)
        or time.dtype != torch.float32
        or time.device.type != "cpu"
        or not torch.isfinite(time).all()
        or (time < 0).any()
        or (time > 1).any()
    ):
        raise ValueError("Expected finite float32 CPU times in [0,1].")


def linear_conditional_flow(
    clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return linear-path points and their constant conditional velocities."""
    _finite_trajectory_batch(clean)
    _finite_trajectory_batch(noise)
    if clean.shape != noise.shape or clean.dtype != noise.dtype:
        raise ValueError(
            "Clean trajectories and noise must have matching shape and dtype."
        )
    _continuous_time(time, len(clean))
    weight = time[:, None, None]
    return (1 - weight) * noise + weight * clean, clean - noise


class ContinuousTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar continuous time on the closed unit interval."""

    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        if width < 4 or width % 4 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Time embedding width and scale are invalid.")
        half = width // 2
        self.scale = float(scale)
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(10000) * torch.arange(half) / (half - 1)),
        )
        self.projection = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if (
            time.ndim != 1
            or not time.is_floating_point()
            or not torch.isfinite(time).all()
            or (time < 0).any()
            or (time > 1).any()
        ):
            raise ValueError("Continuous time must be a finite vector in [0,1].")
        angles = self.scale * time[:, None] * self.frequencies[None]
        return self.projection(torch.cat((angles.sin(), angles.cos()), dim=-1))


class TemporalVelocityModel(nn.Module):
    """Residual dilated temporal ConvNet predicting conditional trajectory velocity."""

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__()
        width = config.width
        self.task_embedding = (
            nn.Embedding(config.num_tasks, width) if config.num_tasks else None
        )
        self.input = nn.Conv1d(config.features + 1, width, 1)
        self.time_embedding = ContinuousTimeEmbedding(
            width, config.time_embedding_scale
        )
        self.goal_projection = nn.Linear(3, width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, width),
                    nn.SiLU(),
                    nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
                    nn.GroupNorm(4, width),
                    nn.SiLU(),
                    nn.Conv1d(width, width, 1),
                )
                for dilation in (1, 2, 4, 8, 16, 32)
            ]
        )
        self.output = nn.Conv1d(width, config.features, 1)

    def forward(self, x, time, task_ids, goals):
        condition = self.time_embedding(time)
        if self.task_embedding is not None:
            condition = condition + self.task_embedding(task_ids)
        condition = (condition + self.goal_projection(goals))[:, :, None]
        position = torch.linspace(-1, 1, x.shape[1], device=x.device)
        position = position[None, None].expand(x.shape[0], 1, -1)
        hidden = self.input(torch.cat((x.transpose(1, 2), position), dim=1))
        for block in self.blocks:
            hidden = (hidden + block(hidden + condition)) / math.sqrt(2)
        return self.output(hidden).transpose(1, 2)


class TrajectoryFlowMatching(nn.Module):
    """Conditional velocity model for normalized current18/action4 trajectories."""

    def __init__(self, config: FlowMatchingConfig = FlowMatchingConfig()) -> None:
        super().__init__()
        if (
            config.horizon < 1
            or config.features != DIFFUSION_FEATURES
            or config.width < 4
            or config.width % 4
            or config.num_tasks < 0
            or config.integration_steps < 1
            or not math.isfinite(config.time_embedding_scale)
            or config.time_embedding_scale <= 0
        ):
            raise ValueError("Invalid flow matching configuration.")
        self.config = config
        self.velocity_model = TemporalVelocityModel(config)
        self.last_sample_nfe = 0

    def _validate(self, x: torch.Tensor) -> None:
        _finite_trajectory_batch(x)
        if x.shape[1:] != (self.config.horizon, self.config.features):
            raise ValueError("Tensor shape differs from flow matching configuration.")

    def validate_tasks(self, task_ids, count: int) -> None:
        if self.config.num_tasks == 0:
            if task_ids is not None:
                raise ValueError("Unconditional model does not accept task IDs.")
        elif (
            not isinstance(task_ids, torch.Tensor)
            or task_ids.shape != (count,)
            or task_ids.dtype != torch.long
            or task_ids.device.type != "cpu"
            or (task_ids < 0).any()
            or (task_ids >= self.config.num_tasks).any()
        ):
            raise ValueError("Require one valid CPU integer task index per trajectory.")

    @staticmethod
    def _validate_goals(goals, count: int) -> None:
        if (
            not isinstance(goals, torch.Tensor)
            or goals.shape != (count, 3)
            or goals.dtype != torch.float32
            or goals.device.type != "cpu"
            or not torch.isfinite(goals).all()
        ):
            raise ValueError("Expected finite float32 CPU goals (N,3).")

    def forward(self, x_t, time, task_ids=None, *, goals=None):
        """Predict normalized velocity; only the temporal network uses its device."""
        self._validate(x_t)
        _continuous_time(time, len(x_t))
        self.validate_tasks(task_ids, len(x_t))
        self._validate_goals(goals, len(x_t))
        # The seeded path and integration state intentionally remain on CPU.
        # This boundary moves only the neural computation and preserves gradients.
        return run_cpu_inputs_on_module_device(
            self.velocity_model, x_t, time, task_ids, goals
        )

    def loss(self, clean, *, generator, task_ids=None, goals=None):
        """Regress the conditional velocity of the independent linear path."""
        self._validate(clean)
        self.validate_tasks(task_ids, len(clean))
        self._validate_goals(goals, len(clean))
        time = torch.rand((len(clean),), generator=generator)
        noise = torch.randn(clean.shape, generator=generator)
        x_t, target_velocity = linear_conditional_flow(clean, noise, time)
        return nn.functional.mse_loss(
            self(x_t, time, task_ids, goals=goals), target_velocity
        )

    @torch.no_grad()
    def sample(
        self,
        count,
        *,
        seed=0,
        task_ids=None,
        goals=None,
        integration_steps=None,
        solver="euler",
        rtol=1e-5,
        atol=1e-5,
    ):
        """Integrate the learned velocity field through ``torchdiffeq.odeint``."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Sample count must be a positive integer.")
        self.validate_tasks(task_ids, count)
        self._validate_goals(goals, count)
        steps = (
            self.config.integration_steps
            if integration_steps is None
            else integration_steps
        )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("integration_steps must be a positive integer.")
        fixed_step_solvers = {"euler", "midpoint", "heun2", "rk4"}
        if solver not in fixed_step_solvers | {"dopri5"}:
            raise ValueError("Unsupported ODE solver.")
        if (
            isinstance(rtol, bool)
            or isinstance(atol, bool)
            or not math.isfinite(rtol)
            or not math.isfinite(atol)
            or rtol <= 0
            or atol <= 0
        ):
            raise ValueError("rtol and atol must be positive and finite.")
        self.eval()
        generator = torch.Generator().manual_seed(seed)
        device = next(self.velocity_model.parameters()).device
        initial = torch.randn(
            (count, self.config.horizon, self.config.features), generator=generator
        ).to(device)
        conditioned_tasks = task_ids.to(device) if task_ids is not None else None
        conditioned_goals = goals.to(device)
        self.last_sample_nfe = 0

        def velocity(time, state):
            self.last_sample_nfe += 1
            time = time.clamp(0, 1)
            time = torch.full(
                (count,), time, dtype=state.dtype, device=state.device
            )
            return self.velocity_model(
                state, time, conditioned_tasks, conditioned_goals
            )

        times = torch.tensor([0.0, 1.0], device=device)
        options = {"step_size": 1.0 / steps} if solver in fixed_step_solvers else None
        result = odeint(
            velocity,
            initial,
            times,
            method=solver,
            rtol=rtol,
            atol=atol,
            options=options,
        )[-1].cpu()
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite flow matching sample.")
        self._validate(result)
        return result

    def save(self, path, dynamic_normalizer, goal_normalizer, metadata):
        """Persist Flow weights, normalization, and provenance without overwrite."""
        atomic_torch_save(
            path,
            generator_checkpoint_payload(
                kind=CHECKPOINT_KIND,
                config=self.config,
                state_dict=self.state_dict(),
                dynamic_normalizer=dynamic_normalizer,
                goal_normalizer=goal_normalizer,
                metadata=metadata,
            ),
        )

    @classmethod
    def load(
        cls,
        path,
        *,
        expected_config=None,
        expected_projection=None,
        expected_action_space=None,
        expected_clamp_epsilon=None,
        expected_task_order=None,
        expected_stage=None,
        expected_task=None,
    ):
        """Restore and validate a CPU Flow checkpoint and fitted normalizers."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("kind") != CHECKPOINT_KIND:
            raise ValueError(f"Checkpoint is not {CHECKPOINT_KIND}.")
        try:
            config = FlowMatchingConfig(**data["config"])
            metadata = data["metadata"]
            dynamic = FeatureNormalizer(data["dynamic_mean"], data["dynamic_scale"])
            goal = FeatureNormalizer(data["goal_mean"], data["goal_scale"])
        except (KeyError, TypeError) as error:
            raise ValueError("Flow Matching checkpoint is incomplete.") from error
        required_metadata = {
            "task_order",
            "training_banks",
            "train_configurations",
            "generated_action_projection",
            "action_space",
            "action_clamp_epsilon",
        }
        strict_metadata = any(
            value is not None
            for value in (
                expected_config,
                expected_projection,
                expected_action_space,
                expected_clamp_epsilon,
                expected_task_order,
                expected_stage,
                expected_task,
            )
        )
        if not isinstance(metadata, dict) or (
            strict_metadata and not required_metadata.issubset(metadata)
        ):
            raise ValueError("Flow Matching checkpoint metadata is incomplete.")
        for name, normalizer, features in (
            ("dynamic", dynamic, config.features),
            ("goal", goal, 3),
        ):
            if (
                normalizer.mean.shape != (features,)
                or normalizer.scale.shape != (features,)
                or not torch.isfinite(normalizer.mean).all()
                or not torch.isfinite(normalizer.scale).all()
                or (normalizer.scale <= 0).any()
            ):
                raise ValueError(
                    f"Flow Matching checkpoint {name} normalization is invalid."
                )
        if expected_config is not None and config != expected_config:
            raise ValueError(
                "Flow Matching checkpoint model configuration differs from run."
            )
        if (
            expected_projection is not None
            and metadata["generated_action_projection"] != expected_projection
            or expected_action_space is not None
            and metadata["action_space"] != expected_action_space
            or expected_clamp_epsilon is not None
            and metadata["action_clamp_epsilon"] != expected_clamp_epsilon
            or expected_task_order is not None
            and metadata["task_order"] != list(expected_task_order)
            or expected_stage is not None
            and metadata.get("stage") != expected_stage
            or expected_task is not None
            and metadata.get("task") != expected_task
        ):
            raise ValueError("Flow Matching checkpoint metadata differs from run.")
        model = cls(config)
        model.load_state_dict(data["state_dict"])
        model.eval()
        return (
            model,
            dynamic,
            goal,
            metadata,
        )


def fit_flow_matching(
    model,
    train,
    validation,
    train_goals,
    validation_goals,
    *,
    train_task_ids,
    validation_task_ids,
    epochs=300,
    batch_size=8,
    learning_rate=0.001,
    seed=0,
    progress=False,
):
    """Fit seeded velocity regression and restore the best validation epoch."""
    model._validate(train)
    model._validate(validation)
    model._validate_goals(train_goals, len(train))
    model._validate_goals(validation_goals, len(validation))
    model.validate_tasks(train_task_ids, len(train))
    model.validate_tasks(validation_task_ids, len(validation))
    return fit_generator(
        model,
        train,
        validation,
        train_goals,
        validation_goals,
        train_task_ids,
        validation_task_ids,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        progress=progress,
        progress_label="Flow Matching training",
        nonfinite_error="Nonfinite flow matching loss.",
        report_epoch_metrics=True,
    )
