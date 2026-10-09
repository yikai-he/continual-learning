"""22D trajectory DDPM conditioned on task IDs and normalized 3D goals."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from src.config import DiffusionModelConfig as DiffusionConfig
from src.support.io import atomic_torch_save

from .diffusion_data import DIFFUSION_FEATURES


def _finite_batch(x: torch.Tensor) -> None:
    if (
        x.ndim != 3
        or min(x.shape) < 1
        or not x.is_floating_point()
        or x.device.type != "cpu"
        or not torch.isfinite(x).all()
    ):
        raise ValueError("Expected a nonempty finite floating CPU tensor (N,T,F).")


@dataclass
class FeatureNormalizer:
    """Per-feature z-score fitted only on supplied training episodes; no clipping."""

    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(
        cls, train: torch.Tensor, *, minimum_scale: float = 1e-3
    ) -> FeatureNormalizer:
        """Fit per-feature moments over batch and time with a scale floor."""
        _finite_batch(train)
        if not math.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive and finite.")
        return cls(
            train.mean(dim=(0, 1)),
            train.std(dim=(0, 1), unbiased=False).clamp_min(minimum_scale),
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        _finite_batch(x)
        if x.shape[-1] != self.mean.numel():
            raise ValueError("Feature count differs from fitted normalization.")
        return (x - self.mean) / self.scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        _finite_batch(x)
        if x.shape[-1] != self.mean.numel():
            raise ValueError("Feature count differs from fitted normalization.")
        return x * self.scale + self.mean


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
        # Keep normalization, schedules and seeded noise on CPU. Only the
        # denoiser executes on the selected device; transfers preserve gradients.
        device = next(self.denoiser.parameters()).device
        return self.denoiser(
            noisy.to(device),
            timestep.to(device),
            task_ids.to(device) if task_ids is not None else None,
            goals.to(device),
        ).cpu()

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
    def sample(self, count, *, seed=0, task_ids=None, goals=None):
        """Sample normalized (N,horizon,22) dynamics under task and goal conditions."""
        self._goals(goals, count)
        self.validate_tasks(task_ids, count)
        self.eval()
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

    def save(self, path, dynamic_normalizer, goal_normalizer, metadata):
        """Persist model, both normalizers, and replay provenance without overwrite."""
        atomic_torch_save(
            path,
            {
                "kind": "trajectory_diffusion_22_v1",
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "dynamic_mean": dynamic_normalizer.mean,
                "dynamic_scale": dynamic_normalizer.scale,
                "goal_mean": goal_normalizer.mean,
                "goal_scale": goal_normalizer.scale,
                "metadata": metadata,
            },
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
    ):
        """Restore a CPU replay checkpoint after validating its run contract."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("kind") != "trajectory_diffusion_22_v1":
            raise ValueError("Checkpoint is not trajectory_diffusion_22_v1.")
        config = DiffusionConfig(**data["config"])
        metadata = data["metadata"]
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
        model.load_state_dict(data["state_dict"])
        model.eval()
        return (
            model,
            FeatureNormalizer(data["dynamic_mean"], data["dynamic_scale"]),
            FeatureNormalizer(data["goal_mean"], data["goal_scale"]),
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
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid fit settings.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def assess(data, goals, ids, noise_seed):
        local = torch.Generator().manual_seed(noise_seed)
        return (
            sum(
                model.loss(data, generator=local, task_ids=ids, goals=goals).item()
                for _ in range(4)
            )
            / 4
        )

    def metrics():
        return {
            "train_loss": assess(train, train_goals, train_task_ids, seed + 1000),
            "validation_loss": assess(
                validation, validation_goals, validation_task_ids, seed + 2000
            ),
        }

    initial = metrics()
    history = []
    best = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in tqdm(
        range(1, epochs + 1),
        desc="Diffusion training",
        unit="epoch",
        disable=not progress,
    ):
        model.train()
        for batch in torch.randperm(len(train), generator=rng).split(batch_size):
            optimizer.zero_grad()
            loss = model.loss(
                train[batch],
                generator=rng,
                task_ids=train_task_ids[batch],
                goals=train_goals[batch],
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite diffusion loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        record = {"epoch": epoch, **metrics()}
        history.append(record)
        if record["validation_loss"] < best:
            best = record["validation_loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        "initial": initial,
        "best_epoch": best_epoch,
        "history": history,
        **metrics(),
    }
