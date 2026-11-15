"""Previous-task Flow Matching replay generation for one continual stage."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from src.support.reproducibility import file_hash, state_hash

from .diffusion_data import (
    GENERATOR_FEATURES,
    PHYSICAL_ACTION_SLICE,
    PHYSICAL_TRAJECTORY_FEATURES,
)
from .replay_common import (
    ReplayResult,
    ReplayTaskContext,
    generate_replay_groups,
    training_configuration_hashes,
    valid_replay_counts,
)
from .trajectory_flow_matching import TrajectoryFlowMatching

ReplaySource = dict[str, object]
ProgressMessage = Callable[..., None]


@dataclass(frozen=True)
class FlowMatchingReplayRequest:
    """Generate old-task physical replay from a prior Flow Matching checkpoint."""

    stage: int
    trajectory_counts: dict[int, int]
    checkpoint: Path
    expected_state_hash: str | None
    expected_config: object | None
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
    solver: str = "euler"
    rtol: float = 1e-5
    atol: float = 1e-5


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
        or request.solver not in ("euler", "midpoint", "heun2", "rk4", "dopri5")
        or not 0 < request.rtol < float("inf")
        or not 0 < request.atol < float("inf")
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
        # Invalid serialized checkpoint contents use the established ValueError API.
        raise ValueError(  # noqa: TRY004
            "Flow Matching checkpoint metadata must be a mapping."
        )
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
            f"Flow Matching checkpoint metadata is incomplete: {min(missing)}."
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
) -> ReplayResult:
    """Generate, denormalize, project actions, and validate old-task replay."""
    _validate_request(request)
    old, old_norm, old_goal_norm, old_meta = TrajectoryFlowMatching.load(
        request.checkpoint,
        expected_config=request.expected_config,
    )
    _validate_checkpoint_metadata(old_meta, request)
    if (
        old.config.horizon != request.horizon
        or old.config.features != GENERATOR_FEATURES
    ):
        raise ValueError(
            "Flow Matching checkpoint trajectory shape differs from replay request."
        )
    old.velocity_model.to(request.device)
    if state_hash(old) != request.expected_state_hash:
        raise RuntimeError(
            "Previous checkpoint differs from current Flow Matching memory."
        )
    checkpoint_sha256 = file_hash(request.checkpoint)

    def sample_normalized(task: ReplayTaskContext, normalized_goals: torch.Tensor):
        generation_started = time.perf_counter()
        normalized = old.sample(
            task.trajectory_count,
            seed=task.seed,
            task_ids=task.labels,
            goals=normalized_goals,
            integration_steps=request.integration_steps,
            solver=request.solver,
            rtol=request.rtol,
            atol=request.atol,
        )
        generation_seconds = time.perf_counter() - generation_started
        if normalized.shape != (
            task.trajectory_count,
            request.horizon,
            GENERATOR_FEATURES,
        ):
            raise ValueError("Flow Matching generated an invalid trajectory shape.")
        return normalized, generation_seconds

    def validate_projected(projected: torch.Tensor, trajectory_count: int) -> None:
        if projected.shape != (
            trajectory_count,
            request.horizon,
            PHYSICAL_TRAJECTORY_FEATURES,
        ):
            raise ValueError("Flow Matching replay has an invalid physical shape.")
        if request.action_space == "pre-tanh" and torch.any(
            projected[..., PHYSICAL_ACTION_SLICE].abs() > 1
        ):
            raise ValueError("Flow Matching pre-tanh replay actions exceed bounds.")

    def build_source(
        task: ReplayTaskContext,
        generation_seconds: float,
    ) -> ReplaySource:
        return {
            "kind": "generated",
            "generator": "flow_matching",
            "checkpoint_kind": old.checkpoint_kind,
            "checkpoint": str(request.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "sampling_seed": task.seed,
            "condition": task.index,
            "trajectory_count": task.trajectory_count,
            "integration_steps": (
                old.config.integration_steps
                if request.integration_steps is None
                else request.integration_steps
            ),
            "neural_function_evaluations": old.last_sample_nfe,
            "ode_solver": request.solver,
            "rtol": request.rtol,
            "atol": request.atol,
            "generation_seconds": generation_seconds,
            "success": None,
            "return": None,
            "training_configuration_indices": task.selected_configuration_indices,
            "training_configuration_hashes": training_configuration_hashes(task),
        }

    result = generate_replay_groups(
        request,
        counts=request.trajectory_counts,
        metadata=old_meta,
        dynamic_normalizer=old_norm,
        goal_normalizer=old_goal_norm,
        sample_normalized=sample_normalized,
        build_source=build_source,
        validate_projected=validate_projected,
        progress_message=progress_message,
        progress_heading="Generating Flow Matching replay",
        progress_description=lambda name: f"Flow replay: {name}",
    )
    return result
