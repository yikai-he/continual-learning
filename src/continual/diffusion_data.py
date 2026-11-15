"""Packing, normalization preparation, and diagnostics for 43D trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from continual_manipulation.environments import (
    ACTION_SHAPE,
    GOAL_SLICE,
    OBSERVATION_SHAPE,
)

from .generator_normalization import FeatureNormalizer
from continual_manipulation.evaluation.rollout import Trajectory, validate_trajectory

PHYSICAL_TRAJECTORY_FEATURES = OBSERVATION_SHAPE[0] + ACTION_SHAPE[0]
PHYSICAL_OBSERVATION_SLICE = slice(0, OBSERVATION_SHAPE[0])
PHYSICAL_ACTION_SLICE = slice(OBSERVATION_SHAPE[0], PHYSICAL_TRAJECTORY_FEATURES)
GENERATOR_FEATURES = PHYSICAL_TRAJECTORY_FEATURES
GENERATOR_ACTION_SLICE = PHYSICAL_ACTION_SLICE


@dataclass(frozen=True)
class PreparedGeneratorData:
    """Prepared 43D inputs and fitted statistics awaiting timed normalization."""

    train: torch.Tensor
    validation: torch.Tensor
    train_goals: torch.Tensor
    validation_goals: torch.Tensor
    normalizer: FeatureNormalizer
    goal_normalizer: FeatureNormalizer


@dataclass(frozen=True)
class NormalizedGeneratorData:
    """Normalized trajectories and goal conditions passed to a generator fit."""

    train: torch.Tensor
    validation: torch.Tensor
    train_goals: torch.Tensor
    validation_goals: torch.Tensor


def validate_stage_trajectory_group(values, *, stage, task, source):
    """Require a finite physical ``(N,T,43)`` stage trajectory group."""
    if values.ndim != 3 or values.shape[-1] != PHYSICAL_TRAJECTORY_FEATURES:
        raise ValueError(
            f"Invalid trajectory group shape at stage={stage}, task={task}: "
            f"{tuple(values.shape)}."
        )
    observations = values[..., PHYSICAL_OBSERVATION_SLICE]
    actions = values[..., PHYSICAL_ACTION_SLICE]
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError(
            f"Nonfinite trajectory data at stage={stage}, task={task}."
        )


def pack_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    horizon: int,
    task_names: Sequence[str] = ("reach-v3",),
) -> torch.Tensor:
    """Pack full episodes as (N, horizon, obs+action); reject padding/short episodes."""
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    for trajectory in trajectories:
        validate_trajectory(trajectory, horizon=horizon)
        if trajectory.task_id not in task_names or trajectory.length != horizon:
            raise ValueError(
                "Require complete episodes from the explicitly allowed tasks."
            )
    packed = np.stack(
        [np.concatenate((t.observations, t.actions), axis=-1) for t in trajectories]
    )
    result = torch.tensor(packed, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("Packed values must remain finite in float32.")
    return result


def unpack_trajectories(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return observation/action tensors without fabricating reward/success metadata."""
    if (
        packed.ndim != 3
        or packed.shape[-1] != PHYSICAL_TRAJECTORY_FEATURES
        or min(packed.shape) < 1
        or not torch.isfinite(packed).all()
    ):
        raise ValueError("Expected finite (N, T, observation_dim + action_dim) data.")
    return (
        packed[..., PHYSICAL_OBSERVATION_SLICE].clone(),
        packed[..., PHYSICAL_ACTION_SLICE].clone(),
    )

def transform_actions(trajectories, mode, epsilon=1e-4, *, decode=False):
    """Convert actions between bounded physical and pre-tanh latent coordinates."""
    _batch(trajectories, GENERATOR_FEATURES)
    if mode not in ("raw", "pre-tanh") or not 0 < epsilon < 1:
        raise ValueError("Invalid action representation.")
    if mode == "raw":
        return trajectories.clone()
    actions = trajectories[..., GENERATOR_ACTION_SLICE]
    if not decode and ((actions < -1).any() or (actions > 1).any()):
        raise ValueError("Pre-tanh encoding requires bounded actions.")
    changed = (
        torch.tanh(actions)
        if decode
        else torch.atanh(actions.clamp(-1 + epsilon, 1 - epsilon))
    )
    return torch.cat(
        (trajectories[..., : GENERATOR_ACTION_SLICE.start], changed), dim=-1
    )


def project_actions(trajectories, projection):
    """Optionally clip generated actions to [-1,1] without changing state."""
    _batch(trajectories, GENERATOR_FEATURES)
    if projection == "none":
        return trajectories
    if projection != "clip":
        raise ValueError("Unknown generated-action projection.")
    return torch.cat(
        (
            trajectories[..., :GENERATOR_ACTION_SLICE.start],
            trajectories[..., GENERATOR_ACTION_SLICE].clamp(-1, 1),
        ),
        dim=-1,
    )


def _batch(x, features):
    if (
        x.ndim != 3
        or min(x.shape) < 1
        or x.shape[-1] != features
        or not x.is_floating_point()
        or not torch.isfinite(x).all()
    ):
        raise ValueError(f"Expected finite floating (N,T,{features}) tensor.")


def prepare_generator_data(
    train: torch.Tensor,
    validation: torch.Tensor,
    *,
    action_space: str,
    action_clamp_epsilon: float,
    train_reference_goals: torch.Tensor | None = None,
    validation_reference_goals: torch.Tensor | None = None,
) -> PreparedGeneratorData:
    """Fit full-trajectory and external goal statistics on the training split."""
    _batch(train, GENERATOR_FEATURES)
    _batch(validation, GENERATOR_FEATURES)
    train_goals = (
        train[:, 0, GOAL_SLICE]
        if train_reference_goals is None
        else train_reference_goals
    )
    validation_goals = (
        validation[:, 0, GOAL_SLICE]
        if validation_reference_goals is None
        else validation_reference_goals
    )
    for name, goals, trajectories in (
        ("training", train_goals, train),
        ("validation", validation_goals, validation),
    ):
        if (
            goals.shape != (len(trajectories), 3)
            or goals.dtype != trajectories.dtype
            or goals.device != trajectories.device
            or not torch.isfinite(goals).all()
        ):
            raise ValueError(f"Invalid {name} goal conditions.")
    transformed_train = transform_actions(
        train,
        action_space,
        action_clamp_epsilon,
    )
    transformed_validation = transform_actions(
        validation,
        action_space,
        action_clamp_epsilon,
    )
    goal_normalizer = FeatureNormalizer.fit(train_goals[:, None, :])
    normalizer = FeatureNormalizer.fit(transformed_train)
    return PreparedGeneratorData(
        train=transformed_train,
        validation=transformed_validation,
        train_goals=train_goals,
        validation_goals=validation_goals,
        normalizer=normalizer,
        goal_normalizer=goal_normalizer,
    )


def normalize_generator_data(
    prepared: PreparedGeneratorData,
) -> NormalizedGeneratorData:
    """Normalize prepared inputs in the historical timed argument order."""
    return NormalizedGeneratorData(
        train=prepared.normalizer.normalize(prepared.train),
        validation=prepared.normalizer.normalize(prepared.validation),
        train_goals=prepared.goal_normalizer.normalize(
            prepared.train_goals[:, None, :]
        )[:, 0],
        validation_goals=prepared.goal_normalizer.normalize(
            prepared.validation_goals[:, None, :]
        )[:, 0],
    )
