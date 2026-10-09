"""Physical BC trajectory packing and goal-conditioned 22D diffusion encoding."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.envs import (
    ACTION_SHAPE,
    CURRENT_STATE_SLICE,
    GOAL_SLICE,
    OBSERVATION_SHAPE,
    PREVIOUS_STATE_SLICE,
)

from .schema import Trajectory, validate_trajectory

PHYSICAL_TRAJECTORY_FEATURES = OBSERVATION_SHAPE[0] + ACTION_SHAPE[0]
PHYSICAL_OBSERVATION_SLICE = slice(0, OBSERVATION_SHAPE[0])
PHYSICAL_ACTION_SLICE = slice(OBSERVATION_SHAPE[0], PHYSICAL_TRAJECTORY_FEATURES)
DIFFUSION_FEATURES = CURRENT_STATE_SLICE.stop + ACTION_SHAPE[0]
DIFFUSION_ACTION_SLICE = slice(CURRENT_STATE_SLICE.stop, DIFFUSION_FEATURES)


def validate_stage_trajectory_group(values, *, stage, task, source):
    """Validate physical episode structure with actionable stage provenance."""
    source_type = {
        "real_expert": "current",
        "generated": "replay",
    }.get(source.get("kind"), source.get("kind", "unknown"))
    if values.ndim != 3 or values.shape[-1] != PHYSICAL_TRAJECTORY_FEATURES:
        raise ValueError(
            f"Invalid trajectory group shape at stage={stage}, task={task}: "
            f"{tuple(values.shape)}."
        )
    observations = values[..., PHYSICAL_OBSERVATION_SLICE]
    actions = values[..., PHYSICAL_ACTION_SLICE]
    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError(
            f"Nonfinite trajectory data at stage={stage}, task={task}, "
            f"source={source_type}."
        )
    for trajectory_index, episode in enumerate(observations):
        current = episode[..., CURRENT_STATE_SLICE]
        previous = episode[..., PREVIOUS_STATE_SLICE]
        goals = episode[..., GOAL_SLICE]
        if not torch.equal(previous[0], current[0]) or not torch.equal(
            previous[1:], current[:-1]
        ):
            raise ValueError(
                f"Previous-state contract failed at stage={stage}, task={task}, "
                f"trajectory={trajectory_index}, source={source_type}."
            )
        conflicts = torch.nonzero(
            torch.any(goals != goals[0], dim=1), as_tuple=False
        ).flatten()
        if len(conflicts):
            metadata = source.get("collection_configurations", [])
            record = (
                metadata[trajectory_index]
                if trajectory_index < len(metadata)
                else {}
            )
            if not record:
                indices = source.get("training_configuration_indices", [])
                hashes = source.get("training_configuration_hashes", [])
                record = {
                    "task_bank_index": indices[trajectory_index]
                    if trajectory_index < len(indices)
                    else None,
                    "task_hash": hashes[trajectory_index]
                    if trajectory_index < len(hashes)
                    else None,
                }
            drift = float(torch.max(torch.abs(goals - goals[0])))
            raise ValueError(
                "Goal must be exactly constant within each episode: "
                f"stage={stage}, task={task}, trajectory={trajectory_index}, "
                f"source={source_type}, length={len(episode)}, "
                f"task_bank_index={record.get('task_bank_index', record.get('configuration_index'))}, "
                f"task_hash={record.get('task_hash', record.get('task_data_sha256'))}, "
                f"reset_seed={record.get('reset_seed')}, max_goal_drift={drift}, "
                f"first_conflicting_timestep={int(conflicts[0])}."
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


def transform_actions(dynamic, mode, epsilon=1e-4, *, decode=False):
    """Convert actions between bounded physical and pre-tanh latent coordinates."""
    _batch(dynamic, DIFFUSION_FEATURES)
    if mode not in ("raw", "pre-tanh") or not 0 < epsilon < 1:
        raise ValueError("Invalid action representation.")
    if mode == "raw":
        return dynamic.clone()
    actions = dynamic[..., DIFFUSION_ACTION_SLICE]
    if not decode and ((actions < -1).any() or (actions > 1).any()):
        raise ValueError("Pre-tanh encoding requires bounded actions.")
    changed = (
        torch.tanh(actions)
        if decode
        else torch.atanh(actions.clamp(-1 + epsilon, 1 - epsilon))
    )
    return torch.cat((dynamic[..., CURRENT_STATE_SLICE], changed), dim=-1)


def project_actions(dynamic, projection):
    """Optionally clip generated actions to [-1,1] without changing state."""
    _batch(dynamic, DIFFUSION_FEATURES)
    if projection == "none":
        return dynamic
    if projection != "clip":
        raise ValueError("Unknown generated-action projection.")
    return torch.cat(
        (
            dynamic[..., CURRENT_STATE_SLICE],
            dynamic[..., DIFFUSION_ACTION_SLICE].clamp(-1, 1),
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


def encode_trajectory(observations, actions):
    """(N,T,39), (N,T,4) -> (N,T,22), (N,3), with exact constraints."""
    _batch(observations, OBSERVATION_SHAPE[0])
    _batch(actions, ACTION_SHAPE[0])
    if (
        observations.shape[:2] != actions.shape[:2]
        or observations.dtype != actions.dtype
        or observations.device != actions.device
    ):
        raise ValueError("Observation/action batch, time, dtype and device must match.")
    current, previous, goals = (
        observations[..., CURRENT_STATE_SLICE],
        observations[..., PREVIOUS_STATE_SLICE],
        observations[..., GOAL_SLICE],
    )
    if not torch.equal(previous[:, 0], current[:, 0]):
        raise ValueError("Reset previous state must equal reset current state.")
    if not torch.equal(previous[:, 1:], current[:, :-1]):
        raise ValueError("Previous state does not equal preceding current state.")
    if not torch.equal(goals, goals[:, :1].expand_as(goals)):
        raise ValueError("Goal must be exactly constant within each episode.")
    return torch.cat((current, actions), dim=-1), goals[:, 0].clone()


def decode_trajectory(dynamic, goals):
    """Rebuild physical (N,T,43) trajectories without changing actions.

    Previous state equals current at reset and the preceding current thereafter;
    the supplied goal is repeated exactly at every timestep.
    """
    _batch(dynamic, DIFFUSION_FEATURES)
    if (
        goals.shape != (len(dynamic), GOAL_SLICE.stop - GOAL_SLICE.start)
        or goals.dtype != dynamic.dtype
        or goals.device != dynamic.device
        or not torch.isfinite(goals).all()
    ):
        raise ValueError("Expected matching finite (N,3) episode goals.")
    current = dynamic[..., CURRENT_STATE_SLICE]
    previous = torch.cat((current[:, :1], current[:, :-1]), dim=1)
    return torch.cat(
        (
            current,
            previous,
            goals[:, None].expand(-1, dynamic.shape[1], -1),
            dynamic[..., DIFFUSION_ACTION_SLICE],
        ),
        dim=-1,
    )
