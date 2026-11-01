"""22D trajectory DDPM conditioned on task IDs and normalized 3D goals."""

from __future__ import annotations

import math

import torch
from src.config import DiffusionModelConfig as DiffusionConfig
from src.support.io import atomic_torch_save
from torch import nn

from .diffusion_data import DIFFUSION_FEATURES
from .generator_normalization import FeatureNormalizer, validate_cpu_feature_batch
from .generator_training import (
    fit_generator,
    generator_checkpoint_payload,
    run_cpu_inputs_on_module_device,
)


def _finite_batch(x: torch.Tensor) -> None:
    validate_cpu_feature_batch(x)


class TemporalDenoiser(nn.Module):
    """Six residual dilated Conv1d blocks, width 64 by default; x0 prediction."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        width = config.width
        self.task_embedding = (
            nn.Embedding(config.num_tasks, width) if config.num_tasks else None
        )
        self.input = nn.Conv1d(config.features + 1, width, 1)
        self.time = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, width),
                    nn.SiLU(),
                    nn.Conv1d(width, width, 5, padding=2 * d, dilation=d),
                    nn.GroupNorm(4, width),
                    nn.SiLU(),
                    nn.Conv1d(width, width, 1),
                )
                for d in (1, 2, 4, 8, 16, 32)
            ]
        )
        self.output = nn.Conv1d(width, config.features, 1)
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2 - 1)),
        )

        # Preserve seeded weights and RNG state from the pre-goal-conditioning
        # 22D implementation before constructing the new goal projection.
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Conv1d, nn.Linear, nn.GroupNorm)):
                module.reset_parameters()
        self.goal_projection = nn.Linear(3, width)

    def forward(self, x, timestep, task_ids, goals):
        """Predict clean (N,T,22) dynamics with time, task, and goal conditions."""
        angles = timestep[:, None] * self.frequencies[None]
        condition = self.time(torch.cat((angles.sin(), angles.cos()), dim=-1))
        if self.task_embedding is not None:
            condition = condition + self.task_embedding(task_ids)
        condition = (condition + self.goal_projection(goals))[:, :, None]
        position = torch.linspace(-1, 1, x.shape[1], device=x.device)
        position = position[None, None].expand(x.shape[0], 1, -1)
        h = self.input(torch.cat((x.transpose(1, 2), position), dim=1))
        for block in self.blocks:
            h = (h + block(h + condition)) / math.sqrt(2)
        return self.output(h).transpose(1, 2)


class TrajectoryDiffusion(nn.Module):
    """Predict current18/action4 trajectories with clean goal conditioning."""

    def __init__(self, config: DiffusionConfig = DiffusionConfig()) -> None:
        super().__init__()
        if (
            config.horizon < 1
            or config.features != DIFFUSION_FEATURES
            or config.steps < 2
            or config.width < 4
            or config.width % 4
            or config.num_tasks < 0
            or config.diffusion_action_space not in ("raw", "pre-tanh")
            or not math.isfinite(config.diffusion_action_clamp_epsilon)
            or not 0 < config.diffusion_action_clamp_epsilon < 1
        ):
            raise ValueError("Invalid diffusion dimensions, steps, or width.")
        self.config = config
        self.denoiser = TemporalDenoiser(config)
        # Adapted cosine schedule: beta cap prevents singular posterior arithmetic.
        grid = torch.linspace(0, config.steps, config.steps + 1, dtype=torch.float64)
        cumulative = torch.cos((grid / config.steps + 0.008) / 1.008 * math.pi / 2) ** 2
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999)
        alpha = 1 - beta
        abar = alpha.cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), abar[:-1]))
        for name, values in {
            "abar": abar,
            "posterior_variance": beta * (1 - previous) / (1 - abar),
            "posterior_x0": beta * previous.sqrt() / (1 - abar),
            "posterior_xt": (1 - previous) * alpha.sqrt() / (1 - abar),
        }.items():
            self.register_buffer(name, values.float())

    def _validate(self, x: torch.Tensor) -> None:
        _finite_batch(x)
        if x.shape[1:] != (self.config.horizon, self.config.features):
            raise ValueError("Tensor shape differs from diffusion configuration.")

    def validate_tasks(self, task_ids, count):
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

    def _goals(self, goals, count):
        if (
            not isinstance(goals, torch.Tensor)
            or goals.shape != (count, 3)
            or goals.device.type != "cpu"
            or goals.dtype != self.abar.dtype
            or not torch.isfinite(goals).all()
        ):
            raise ValueError("Expected finite float32 CPU goals (N,3).")

    def forward(self, noisy, timestep, task_ids=None, *, goals=None):
        """Predict normalized x0; only the denoiser runs on its selected device."""
        self._validate(noisy)
        self.validate_tasks(task_ids, len(noisy))
        self._goals(goals, len(noisy))
        if (
            timestep.shape != (len(noisy),)
            or timestep.dtype != torch.long
            or (timestep < 0).any()
            or (timestep >= self.config.steps).any()
        ):
            raise ValueError("Invalid diffusion timesteps.")
        # Normalization, schedules, and seeded noise intentionally remain on CPU.
        # This boundary moves only the neural computation and preserves gradients.
        return run_cpu_inputs_on_module_device(
            self.denoiser, noisy, timestep, task_ids, goals
        )

    def loss(self, clean, *, generator, task_ids=None, goals=None):
        """Train x0 prediction at one randomly sampled timestep per trajectory."""
        self._validate(clean)
        self._goals(goals, len(clean))
        timestep = torch.randint(self.config.steps, (len(clean),), generator=generator)
        noise = torch.randn(clean.shape, generator=generator)
        abar = self.abar[timestep, None, None]
        return nn.functional.mse_loss(
            self(
                abar.sqrt() * clean + (1 - abar).sqrt() * noise,
                timestep,
                task_ids,
                goals=goals,
            ),
            clean,
        )

    @torch.no_grad()
    def sample(
        self,
        count,
        *,
        seed=0,
        task_ids=None,
        goals=None,
        execution_mode="legacy",
    ):
        """Sample normalized trajectories with legacy or device-native execution."""
        self._goals(goals, count)
        self.validate_tasks(task_ids, count)
        if execution_mode not in ("legacy", "device-native"):
            raise ValueError("execution_mode must be legacy or device-native.")
        self.eval()
        if execution_mode == "device-native":
            return self._sample_device_native(count, seed, task_ids, goals)

        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(
            (count, self.config.horizon, DIFFUSION_FEATURES), generator=generator
        )
        for step in reversed(range(self.config.steps)):
            timestep = torch.full((count,), step, dtype=torch.long)
            x0 = self(x, timestep, task_ids, goals=goals)
            x = self.posterior_x0[step] * x0 + self.posterior_xt[step] * x
            if step:
                x += self.posterior_variance[step].sqrt() * torch.randn(
                    x.shape, generator=generator
                )
        self._validate(x)
        return x

    def _sample_device_native(self, count, seed, task_ids, goals):
        """Run the full reverse process on the denoiser device, returning CPU data."""
        device = next(self.denoiser.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        device_tasks = task_ids.to(device) if task_ids is not None else None
        device_goals = goals.to(device)
        posterior_x0 = self.posterior_x0.to(device)
        posterior_xt = self.posterior_xt.to(device)
        posterior_variance = self.posterior_variance.to(device)
        x = torch.randn(
            (count, self.config.horizon, DIFFUSION_FEATURES),
            generator=generator,
            device=device,
        )
        for step in reversed(range(self.config.steps)):
            timestep = torch.full((count,), step, dtype=torch.long, device=device)
            x0 = self.denoiser(x, timestep, device_tasks, device_goals)
            x = posterior_x0[step] * x0 + posterior_xt[step] * x
            if step:
                x += posterior_variance[step].sqrt() * torch.randn(
                    x.shape, generator=generator, device=device
                )
        result = x.cpu()
        self._validate(result)
        return result

    def save(self, path, dynamic_normalizer, goal_normalizer, metadata):
        """Persist model, both normalizers, and replay provenance without overwrite."""
        atomic_torch_save(
            path,
            generator_checkpoint_payload(
                kind="trajectory_diffusion_22_v1",
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
        expected_action_space=None,
        expected_clamp_epsilon=None,
        expected_projection=None,
        expected_task_order=None,
        expected_config=None,
    ):
        """Restore a CPU replay checkpoint after validating its run contract."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("kind") != "trajectory_diffusion_22_v1":
            raise ValueError("Checkpoint is not trajectory_diffusion_22_v1.")
        config = DiffusionConfig(**data["config"])
        metadata = data["metadata"]
        if expected_config is not None and config != expected_config:
            raise ValueError("Diffusion checkpoint model configuration differs from run.")
        if (
            expected_action_space is not None
            and config.diffusion_action_space != expected_action_space
            or expected_clamp_epsilon is not None
            and config.diffusion_action_clamp_epsilon != expected_clamp_epsilon
            or expected_projection is not None
            and metadata["generated_action_projection"] != expected_projection
            or expected_task_order is not None
            and metadata["task_order"] != list(expected_task_order)
        ):
            raise ValueError("Diffusion checkpoint configuration differs from run.")
        model = cls(config)
        dynamic = FeatureNormalizer(data["dynamic_mean"], data["dynamic_scale"])
        goal = FeatureNormalizer(data["goal_mean"], data["goal_scale"])
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
                raise ValueError(f"Diffusion checkpoint {name} normalization is invalid.")

        state_dict = data["state_dict"]
        expected_state = model.state_dict()
        schedule_names = (
            "abar",
            "posterior_variance",
            "posterior_x0",
            "posterior_xt",
        )
        if any(
            name not in state_dict
            or not torch.equal(state_dict[name], expected_state[name])
            for name in schedule_names
        ):
            raise ValueError("Diffusion checkpoint schedule differs from its configuration.")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError("Diffusion checkpoint model state is incompatible.") from error
        model.eval()
        return (
            model,
            dynamic,
            goal,
            metadata,
        )


def fit_diffusion(
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
    """Fit seeded x0 diffusion and restore the lowest validation-loss epoch."""
    model._validate(train)
    model._validate(validation)
    model._goals(train_goals, len(train))
    model._goals(validation_goals, len(validation))
    model.validate_tasks(train_task_ids, len(train))
    model.validate_tasks(validation_task_ids, len(validation))
    # The shared progress iterator retains the explicit disable=not progress policy.
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
        progress_label="Diffusion training",
        nonfinite_error="Nonfinite diffusion loss.",
    )
