from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.continual.diffusion_data import PHYSICAL_TRAJECTORY_FEATURES, unpack_trajectories
from src.envs import EPISODE_HORIZON


@dataclass
class TrajectoryGroupPreparation:
    """Deterministic split and BC diagnostic for one task trajectory group."""

    train_ids: list[int]
    validation_ids: list[int]
    bc_target_clip_fraction: float


@dataclass
class PreparedGroupedTrajectoryDataset:
    """Stage (N,200,43) splits with one diffusion task label per episode."""

    train: torch.Tensor
    validation: torch.Tensor
    train_labels: torch.Tensor
    validation_labels: torch.Tensor


def split_indices(count: int, seed: int) -> tuple[list[int], list[int]]:
    """Whole-trajectory permutation with a rounded 20% holdout, per task."""
    if count < 2:
        raise ValueError("Need >=2 trajectories per task for a disjoint holdout.")
    indices = np.random.default_rng(seed).permutation(count)
    nval = max(1, min(count - 1, round(count * 0.2)))
    return indices[nval:].tolist(), indices[:nval].tolist()


def bc_dataset(packed: torch.Tensor) -> TensorDataset:
    """Flatten episode/time axes into (N*T,39) observation-action BC samples."""
    observations, actions = unpack_trajectories(packed)
    return TensorDataset(observations.flatten(0, 1), actions.flatten(0, 1))


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
    train_ids, validation_ids = split_indices(count, seed)
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
