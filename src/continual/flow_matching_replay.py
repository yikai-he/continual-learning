"""Previous-task Flow Matching replay generation for one continual stage."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from src.envs import CURRENT_STATE_SLICE, GOAL_SLICE, PREVIOUS_STATE_SLICE
from src.support.reproducibility import file_hash, state_hash
from tqdm.auto import tqdm

from .diffusion_data import (
    DIFFUSION_FEATURES,
    PHYSICAL_ACTION_SLICE,
    PHYSICAL_TRAJECTORY_FEATURES,
    validate_stage_trajectory_group,
)
from .replay_common import (
    decode_generated_replay,
    normalize_replay_goals,
    prepare_replay_task,
    training_configuration_hashes,
    valid_replay_counts,
)
from .task_bank import reconstruct_goals, sample_training_configurations
from .trajectory_flow_matching import CHECKPOINT_KIND, TrajectoryFlowMatching

ReplaySource = dict[str, object]
ProgressMessage = Callable[..., None]


@dataclass(frozen=True)
class FlowMatchingReplayRequest:
    """Generate old-task physical replay from a prior Flow Matching checkpoint."""

    stage: int
    trajectory_counts: dict[int, int]
    checkpoint: Path
    expected_state_hash: str | None
    task_names: tuple[str, ...]
    backend: str
    horizon: int
    runtime_seed: int
    device: str
    reward_function_version: str
    action_space: str
    action_clamp_epsilon: float
    action_projection: str
    bc_clamp_epsilon: float
    progress: bool
    integration_steps: int | None = None
    hammer_reward_variant: str = "original"
    hammer_nail_progress_weight: float = 0.0
    execution_mode: str = "legacy"


@dataclass
class FlowMatchingReplayResult:
    """Generated physical trajectory groups and provenance by previous task."""

    groups: dict[int, torch.Tensor]
    sources: dict[str, ReplaySource]


def _validate_request(request: FlowMatchingReplayRequest) -> None:
    if (
        isinstance(request.stage, bool)
        or not isinstance(request.stage, int)
        or not 0 < request.stage <= len(request.task_names)
        or not valid_replay_counts(request.trajectory_counts, request.stage)
        or isinstance(request.horizon, bool)
        or not isinstance(request.horizon, int)
        or request.horizon < 1
        or request.action_space not in ("raw", "pre-tanh")
        or request.action_projection not in ("none", "clip")
        or not 0 < request.action_clamp_epsilon < 1
        or not 0 < request.bc_clamp_epsilon < 1
        or request.execution_mode not in ("legacy", "device-native")
        or request.integration_steps is not None
        and (
            isinstance(request.integration_steps, bool)
            or not isinstance(request.integration_steps, int)
            or request.integration_steps < 1
        )
    ):
        raise ValueError("Invalid Flow Matching replay request.")


def _validate_checkpoint_metadata(metadata, request: FlowMatchingReplayRequest) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("Flow Matching checkpoint metadata must be a mapping.")
    required = {
        "task_order",
        "training_banks",
        "train_configurations",
        "generated_action_projection",
        "action_space",
        "action_clamp_epsilon",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(
            f"Flow Matching checkpoint metadata is incomplete: {sorted(missing)[0]}."
        )
    if (
        metadata["task_order"] != list(request.task_names)
        or metadata["generated_action_projection"] != request.action_projection
        or metadata["action_space"] != request.action_space
        or metadata["action_clamp_epsilon"] != request.action_clamp_epsilon
    ):
        raise ValueError(
            "Flow Matching checkpoint metadata differs from replay request."
        )
    for old_index in range(request.stage):
        name = request.task_names[old_index]
        trajectory_count = request.trajectory_counts[old_index]
        if trajectory_count == 0:
            continue
        bank = metadata["training_banks"].get(name)
        selection = metadata["train_configurations"].get(name)
        if (
            not isinstance(bank, dict)
            or not isinstance(selection, dict)
            or not isinstance(selection.get("indices"), list)
            or not selection["indices"]
            or not isinstance(bank.get("ordered_task_hashes"), list)
        ):
            raise ValueError(
                f"Flow Matching replay metadata is invalid for task {name}."
            )


def generate_previous_task_flow_matching_replay(
    request: FlowMatchingReplayRequest,
    *,
    progress_message: ProgressMessage | None = None,
) -> FlowMatchingReplayResult:
    """Generate, denormalize, decode, project, and validate old-task replay."""
    _validate_request(request)
    old, old_norm, old_goal_norm, old_meta = TrajectoryFlowMatching.load(
        request.checkpoint
    )
    _validate_checkpoint_metadata(old_meta, request)
    if (
        old.config.horizon != request.horizon
        or old.config.features != DIFFUSION_FEATURES
    ):
        raise ValueError(
            "Flow Matching checkpoint trajectory shape differs from replay request."
        )
    old.velocity_model.to(request.device)
    if state_hash(old) != request.expected_state_hash:
        raise RuntimeError(
            "Previous checkpoint differs from current Flow Matching memory."
        )

    groups: dict[int, torch.Tensor] = {}
    sources: dict[str, ReplaySource] = {}
    for old_index in range(request.stage):
        trajectory_count = request.trajectory_counts[old_index]
        if trajectory_count == 0:
            continue
        name = request.task_names[old_index]
        if progress_message is not None:
            progress_message(
                "Generating Flow Matching replay",
                Task=name,
                Trajectories=trajectory_count,
            )
        task = prepare_replay_task(
            request,
            old_meta,
            old_index,
            trajectory_count,
            sample_configurations=sample_training_configurations,
            reconstruct=reconstruct_goals,
        )
        normalized_goals = normalize_replay_goals(task, old_goal_norm)
        with tqdm(
            total=trajectory_count,
            desc=f"Flow replay: {name}",
            unit="trajectory",
            disable=not request.progress,
        ) as replay_progress:
            generation_started = time.perf_counter()
            normalized = old.sample(
                trajectory_count,
                seed=task.seed,
                task_ids=task.labels,
                goals=normalized_goals,
                integration_steps=request.integration_steps,
                execution_mode=request.execution_mode,
            )
            generation_seconds = time.perf_counter() - generation_started
            if normalized.shape != (
                trajectory_count,
                request.horizon,
                DIFFUSION_FEATURES,
            ):
                raise ValueError("Flow Matching generated an invalid trajectory shape.")
            sampled = old_norm.denormalize(normalized)
            replay_progress.update(trajectory_count)

        replay = decode_generated_replay(
            sampled,
            task.goals,
            action_space=request.action_space,
            action_projection=request.action_projection,
            bc_clamp_epsilon=request.bc_clamp_epsilon,
        )
        projected = replay.projected
        if projected.shape != (
            trajectory_count,
            request.horizon,
            PHYSICAL_TRAJECTORY_FEATURES,
        ):
            raise ValueError("Decoded Flow Matching replay has an invalid shape.")
        if not torch.equal(
            projected[..., PREVIOUS_STATE_SLICE],
            torch.cat(
                (
                    projected[:, :1, CURRENT_STATE_SLICE],
                    projected[:, :-1, CURRENT_STATE_SLICE],
                ),
                dim=1,
            ),
        ):
            raise ValueError(
                "Decoded Flow Matching replay violates previous-state alignment."
            )
        if not torch.equal(
            projected[..., GOAL_SLICE],
            task.goals[:, None].expand(-1, request.horizon, -1),
        ):
            raise ValueError(
                "Decoded Flow Matching replay violates constant-goal alignment."
            )
        if request.action_space == "pre-tanh" and torch.any(
            projected[..., PHYSICAL_ACTION_SLICE].abs() > 1
        ):
            raise ValueError("Decoded pre-tanh Flow Matching actions exceed bounds.")

        source: ReplaySource = {
            "kind": "generated",
            "generator": "flow_matching",
            "checkpoint_kind": CHECKPOINT_KIND,
            "checkpoint": str(request.checkpoint),
            "checkpoint_sha256": file_hash(request.checkpoint),
            "sampling_seed": task.seed,
            "condition": old_index,
            "trajectory_count": trajectory_count,
            "integration_steps": (
                old.config.integration_steps
                if request.integration_steps is None
                else request.integration_steps
            ),
            "neural_function_evaluations": old.last_sample_nfe,
            "generation_seconds": generation_seconds,
            "success": None,
            "return": None,
            "generated_latent_abs_gt_bc_cap_fraction": replay.latent_cap_fraction,
            "training_configuration_indices": task.selected_configuration_indices,
            "training_configuration_hashes": training_configuration_hashes(task),
            **replay.action_diagnostics,
        }
        validate_stage_trajectory_group(
            projected, stage=request.stage, task=name, source=source
        )
        groups[old_index] = projected
        sources[name] = source
    del old
    return FlowMatchingReplayResult(groups=groups, sources=sources)
