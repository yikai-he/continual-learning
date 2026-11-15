"""Private behavior-preserving helpers shared by generated replay providers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from tqdm.auto import tqdm

from .diffusion_data import (
    project_actions,
    transform_actions,
    validate_stage_trajectory_group,
)
from .task_bank import reconstruct_goals, sample_training_configurations


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


@dataclass
class ReplayResult:
    """Physical trajectory groups and provenance keyed by historical task."""

    groups: dict[int, torch.Tensor]
    sources: dict[str, dict[str, object]]


def prepare_replay_task(
    request: Any,
    metadata: Mapping[str, Any],
    old_index: int,
    trajectory_count: int,
) -> ReplayTaskContext:
    """Reconstruct the exact historical task labels, selection, and goals."""
    name = request.task_names[old_index]
    seed = request.runtime_seed + 100000 + request.stage * 1000 + old_index
    labels = torch.full((trajectory_count,), old_index, dtype=torch.long)
    bank = metadata["training_banks"][name]
    selected = sample_training_configurations(
        bank,
        metadata["train_configurations"][name]["indices"],
        trajectory_count,
        seed,
    )
    environment_kwargs = {
        "backend": request.backend,
        "reward_function_version": request.reward_function_version,
    }
    goals = torch.as_tensor(
        reconstruct_goals(bank, selected, **environment_kwargs), dtype=torch.float32
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


def prepare_generated_replay(
    sampled: torch.Tensor,
    *,
    action_space: str,
    action_projection: str,
) -> torch.Tensor:
    """Decode action coordinates without modifying generated state or goal values."""
    before_projection = transform_actions(sampled, action_space, decode=True)
    return project_actions(before_projection, action_projection)


def training_configuration_hashes(context: ReplayTaskContext) -> list[str]:
    """Resolve selected configuration indices to their ordered task hashes."""
    return [
        context.bank["ordered_task_hashes"][index]
        for index in context.selected_configuration_indices
    ]


def generate_replay_groups(
    request: Any,
    *,
    counts: Mapping[int, int],
    metadata: Mapping[str, Any],
    dynamic_normalizer: Any,
    goal_normalizer: Any,
    sample_normalized: Callable[
        [ReplayTaskContext, torch.Tensor], tuple[torch.Tensor, Any]
    ],
    build_source: Callable[[ReplayTaskContext, Any], dict[str, object]],
    validate_projected: Callable[[torch.Tensor, int], None],
    progress_message: Callable[..., None] | None,
    progress_heading: str,
    progress_description: Callable[[str], str],
) -> ReplayResult:
    """Run the shared deterministic per-task generated-replay workflow."""
    groups: dict[int, torch.Tensor] = {}
    sources: dict[str, dict[str, object]] = {}
    for old_index in range(request.stage):
        trajectory_count = counts[old_index]
        if trajectory_count == 0:
            continue
        name = request.task_names[old_index]
        if progress_message is not None:
            progress_message(
                progress_heading,
                Task=name,
                Trajectories=trajectory_count,
            )
        task = prepare_replay_task(
            request,
            metadata,
            old_index,
            trajectory_count,
        )
        normalized_goals = normalize_replay_goals(task, goal_normalizer)
        with tqdm(
            total=trajectory_count,
            desc=progress_description(name),
            unit="trajectory",
            disable=not request.progress,
        ) as replay_progress:
            normalized, sampling_details = sample_normalized(task, normalized_goals)
            sampled = dynamic_normalizer.denormalize(normalized)
            replay_progress.update(trajectory_count)

        projected = prepare_generated_replay(
            sampled,
            action_space=request.action_space,
            action_projection=request.action_projection,
        )
        validate_projected(projected, trajectory_count)
        source = build_source(task, sampling_details)
        source["reference_goals"] = task.goals.tolist()
        validate_stage_trajectory_group(
            projected, stage=request.stage, task=name, source=source
        )
        groups[old_index] = projected
        sources[name] = source
    return ReplayResult(groups=groups, sources=sources)
