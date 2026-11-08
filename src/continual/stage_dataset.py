"""Deterministic preparation of grouped DiffCRL stage trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from src.continual.diffusion_data import (
    PHYSICAL_TRAJECTORY_FEATURES,
    unpack_trajectories,
    validate_stage_trajectory_group,
)
from src.envs import EPISODE_HORIZON
from torch.utils.data import TensorDataset


@dataclass
class TrajectoryGroupPreparation:
    """Deterministic split and BC diagnostic for one task trajectory group."""

    train_ids: list[int]
    validation_ids: list[int]
    bc_target_clip_fraction: float


@dataclass
class PreparedGroupedTrajectoryDataset:
    """Stage ``(N,T,43)`` splits with one diffusion task label per episode."""

    train: torch.Tensor
    validation: torch.Tensor
    train_labels: torch.Tensor
    validation_labels: torch.Tensor


@dataclass
class PreparedStageGroups:
    """Stateless products of ordered per-task stage preparation."""

    dataset: PreparedGroupedTrajectoryDataset
    preparations: dict[int, TrajectoryGroupPreparation]
    splits: dict[str, dict[str, object]]


def validate_stage_group(
    values: torch.Tensor,
    source: dict[str, object],
    *,
    stage: int,
    task: str,
    horizon: int,
) -> int:
    """Validate one ordered physical group and return its episode count."""
    group_count = len(values)
    validate_trajectory_group(values, count=group_count, horizon=horizon)
    validate_stage_trajectory_group(
        values,
        stage=stage,
        task=task,
        source=source,
    )
    return group_count


def prepare_stage_group_split(
    values: torch.Tensor,
    *,
    count: int,
    split_seed: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    clamp_epsilon: float,
    training_only: bool,
) -> TrajectoryGroupPreparation:
    """Create one deterministic split and optional training-only override."""
    prepared = prepare_trajectory_group(
        values,
        count=count,
        seed=split_seed,
        action_low=action_low,
        action_high=action_high,
        clamp_epsilon=clamp_epsilon,
    )
    if training_only:
        prepared.train_ids = list(range(count))
        prepared.validation_ids = []
    return prepared


def finalize_stage_groups(
    groups: Mapping[int, torch.Tensor],
    preparations: Mapping[int, TrajectoryGroupPreparation],
    task_names: list[str] | tuple[str, ...],
    *,
    horizon: int,
) -> tuple[PreparedGroupedTrajectoryDataset, dict[str, dict[str, object]]]:
    """Concatenate ordered groups and construct their established split metadata."""
    splits: dict[str, dict[str, object]] = {}
    for index, values in groups.items():
        prepared = preparations[index]
        name = task_names[index]
        splits[name] = {
            "train_ids": prepared.train_ids,
            "validation_ids": prepared.validation_ids,
            "trajectories": len(values),
            "train_samples": len(prepared.train_ids) * horizon,
            "validation_samples": len(prepared.validation_ids) * horizon,
        }
    return concatenate_trajectory_groups(groups, preparations), splits


def split_indices(count: int, seed: int) -> tuple[list[int], list[int]]:
    """Whole-trajectory permutation with a rounded 20% holdout, per task."""
    if count < 2:
        raise ValueError("Need >=2 trajectories per task for a disjoint holdout.")
    indices = np.random.default_rng(seed).permutation(count)
    nval = max(1, min(count - 1, round(count * 0.2)))
    return indices[nval:].tolist(), indices[:nval].tolist()


def bc_dataset(
    packed: torch.Tensor, task_labels: torch.Tensor | None = None
) -> TensorDataset:
    """Flatten episode/time axes into (N*T,39) observation-action BC samples."""
    observations, actions = unpack_trajectories(packed)
    tensors = [observations.flatten(0, 1), actions.flatten(0, 1)]
    if task_labels is not None:
        if task_labels.shape != (len(packed),) or task_labels.dtype != torch.long:
            raise ValueError("BC task labels must be one integer label per episode.")
        tensors.append(
            task_labels[:, None].expand(-1, packed.shape[1]).reshape(-1)
        )
    return TensorDataset(*tensors)


def validate_trajectory_group(
    values: torch.Tensor,
    *,
    count: int,
    horizon: int = EPISODE_HORIZON,
    features: int = PHYSICAL_TRAJECTORY_FEATURES,
) -> None:
    """Require one finite, fixed-size packed trajectory group."""
    if values.shape != (count, horizon, features) or not torch.isfinite(values).all():
        raise ValueError("Invalid physical stage data.")


def prepare_trajectory_group(
    values: torch.Tensor,
    *,
    count: int,
    seed: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    clamp_epsilon: float,
) -> TrajectoryGroupPreparation:
    """Return the original whole-episode split and BC clipping diagnostic."""
    actions = bc_dataset(values).tensors[1]
    scaled = 2 * (actions - action_low) / (action_high - action_low) - 1
    clip_fraction = float(
        ((scaled < -1 + clamp_epsilon) | (scaled > 1 - clamp_epsilon)).float().mean()
    )
    # A fixed total replay budget can assign one historical episode to a task.
    # Keep it in training; the current-task group still provides validation.
    train_ids, validation_ids = ([0], []) if count == 1 else split_indices(count, seed)
    return TrajectoryGroupPreparation(
        train_ids=train_ids,
        validation_ids=validation_ids,
        bc_target_clip_fraction=clip_fraction,
    )


def concatenate_trajectory_groups(
    groups: Mapping[int, torch.Tensor],
    preparations: Mapping[int, TrajectoryGroupPreparation],
) -> PreparedGroupedTrajectoryDataset:
    """Concatenate task groups in mapping order and construct task-index labels."""
    trains, validations, train_labels, validation_labels = [], [], [], []
    for index, values in groups.items():
        prepared = preparations[index]
        trains.append(values[prepared.train_ids])
        validations.append(values[prepared.validation_ids])
        train_labels.append(
            torch.full((len(prepared.train_ids),), index, dtype=torch.long)
        )
        validation_labels.append(
            torch.full((len(prepared.validation_ids),), index, dtype=torch.long)
        )
    return PreparedGroupedTrajectoryDataset(
        train=torch.cat(trains),
        validation=torch.cat(validations),
        train_labels=torch.cat(train_labels),
        validation_labels=torch.cat(validation_labels),
    )


def prepare_stage_groups(
    groups: Mapping[int, torch.Tensor],
    sources: Mapping[str, dict[str, object]],
    task_names: list[str] | tuple[str, ...],
    *,
    stage: int,
    horizon: int,
    runtime_seed: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    clamp_epsilon: float,
    historical_replay_training_only: bool,
) -> PreparedStageGroups:
    """Validate and split ordered physical groups without I/O or mutation."""
    preparations: dict[int, TrajectoryGroupPreparation] = {}
    for index, values in groups.items():
        name = task_names[index]
        group_count = validate_stage_group(
            values,
            sources[name],
            stage=stage,
            task=name,
            horizon=horizon,
        )
        preparations[index] = prepare_stage_group_split(
            values,
            count=group_count,
            split_seed=runtime_seed + stage * 1000 + index,
            action_low=action_low,
            action_high=action_high,
            clamp_epsilon=clamp_epsilon,
            training_only=historical_replay_training_only and index < stage,
        )
    dataset, splits = finalize_stage_groups(
        groups, preparations, task_names, horizon=horizon
    )
    return PreparedStageGroups(
        dataset=dataset,
        preparations=preparations,
        splits=splits,
    )
