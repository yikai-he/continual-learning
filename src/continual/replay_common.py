"""Private behavior-preserving helpers shared by generated replay providers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .diffusion_data import (
    DIFFUSION_ACTION_SLICE,
    decode_trajectory,
    generated_action_diagnostics,
    project_actions,
    transform_actions,
)


def valid_replay_counts(counts: Mapping[int, int], stage: int) -> bool:
    """Return whether counts exactly cover prior task indices with integers."""
    return set(counts) == set(range(stage)) and all(
        not isinstance(count, bool) and isinstance(count, int) and count >= 0
        for count in counts.values()
    )


@dataclass(frozen=True)
class ReplayTaskContext:
    """Deterministic conditioning inputs for one historical replay task."""

    index: int
    name: str
    trajectory_count: int
    seed: int
    labels: torch.Tensor
    bank: dict
    selected_configuration_indices: list[int]
    goals: torch.Tensor


@dataclass(frozen=True)
class DecodedReplay:
    """Shared physical replay products after action decoding and projection."""

    before_projection: torch.Tensor
    projected: torch.Tensor
    latent_cap_fraction: float | None
    action_diagnostics: dict[str, object]


def prepare_replay_task(
    request: Any,
    metadata: Mapping[str, Any],
    old_index: int,
    trajectory_count: int,
    *,
    sample_configurations: Callable[..., list[int]],
    reconstruct: Callable[..., Any],
) -> ReplayTaskContext:
    """Reconstruct the exact historical task labels, selection, and goals."""
    name = request.task_names[old_index]
    seed = request.runtime_seed + 100000 + request.stage * 1000 + old_index
    labels = torch.full((trajectory_count,), old_index, dtype=torch.long)
    bank = metadata["training_banks"][name]
    selected = sample_configurations(
        bank,
        metadata["train_configurations"][name]["indices"],
        trajectory_count,
        seed,
    )
    environment_kwargs = {
        "backend": request.backend,
        "reward_function_version": request.reward_function_version,
    }
    if request.backend == "kuka-v3":
        environment_kwargs.update(
            hammer_reward_variant=request.hammer_reward_variant,
            hammer_nail_progress_weight=request.hammer_nail_progress_weight,
        )
    goals = torch.as_tensor(
        reconstruct(bank, selected, **environment_kwargs), dtype=torch.float32
    )
    return ReplayTaskContext(
        index=old_index,
        name=name,
        trajectory_count=trajectory_count,
        seed=seed,
        labels=labels,
        bank=bank,
        selected_configuration_indices=selected,
        goals=goals,
    )


def normalize_replay_goals(context: ReplayTaskContext, normalizer: Any) -> torch.Tensor:
    """Normalize one task's physical goals using checkpoint statistics."""
    return normalizer.normalize(context.goals[:, None, :])[:, 0]


def decode_generated_replay(
    sampled: torch.Tensor,
    goals: torch.Tensor,
    *,
    action_space: str,
    action_projection: str,
    bc_clamp_epsilon: float,
) -> DecodedReplay:
    """Decode sampled dynamics, project actions, and preserve diagnostics."""
    sampled_actions = sampled[..., DIFFUSION_ACTION_SLICE]
    latent_cap = float(np.arctanh(1 - bc_clamp_epsilon))
    latent_cap_fraction = (
        float((sampled_actions.abs() > latent_cap).float().mean())
        if action_space == "pre-tanh"
        else None
    )
    decoded_dynamic = transform_actions(sampled, action_space, decode=True)
    before_projection = decode_trajectory(decoded_dynamic, goals)
    projected_dynamic = project_actions(decoded_dynamic, action_projection)
    projected = decode_trajectory(projected_dynamic, goals)
    action_diagnostics = generated_action_diagnostics(
        before_projection, projected, action_projection
    )
    return DecodedReplay(
        before_projection=before_projection,
        projected=projected,
        latent_cap_fraction=latent_cap_fraction,
        action_diagnostics=action_diagnostics,
    )


def training_configuration_hashes(context: ReplayTaskContext) -> list[str]:
    """Resolve selected configuration indices to their ordered task hashes."""
    return [
        context.bank["ordered_task_hashes"][index]
        for index in context.selected_configuration_indices
    ]
