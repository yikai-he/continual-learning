"""Previous-task diffusion replay generation for one DiffCRL stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from src.support.reproducibility import file_hash, state_hash

from .diffusion_data import PHYSICAL_ACTION_SLICE
from .replay_common import (
    ReplayResult,
    ReplayTaskContext,
    generate_replay_groups,
    training_configuration_hashes,
    valid_replay_counts,
)
from .trajectory_diffusion import TrajectoryDiffusion

ReplaySource = dict[str, object]
ProgressMessage = Callable[..., None]


@dataclass(frozen=True)
class ReplayRequest:
    """Generate per-task replay for tasks [0, stage) from the prior checkpoint.

    The request fixes the expected memory hash and action decoding contract.
    """

    stage: int
    trajectory_count: int
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
    progress: bool
    expected_config: object | None = None
    trajectory_counts: dict[int, int] | None = None
    execution_mode: str = "legacy"


def generate_previous_task_replay(
    request: ReplayRequest,
    *,
    progress_message: ProgressMessage | None = None,
) -> ReplayResult:
    """Generate direct 43D replay conditioned on reconstructed training goals.

    ``trajectory_counts`` selects shared fixed-total allocation when present;
    otherwise each prior task receives legacy ``trajectory_count`` replay.
    """
    if request.execution_mode not in ("legacy", "device-native"):
        raise ValueError("Invalid DDPM replay execution mode.")
    old, old_norm, old_goal_norm, old_meta = TrajectoryDiffusion.load(
        request.checkpoint,
        expected_action_space=request.action_space,
        expected_clamp_epsilon=request.action_clamp_epsilon,
        expected_projection=request.action_projection,
        expected_task_order=request.task_names,
        expected_config=request.expected_config,
    )
    old.denoiser.to(request.device)
    if state_hash(old) != request.expected_state_hash:
        raise RuntimeError("Previous checkpoint differs from current diffusion memory.")

    counts = request.trajectory_counts or {
        index: request.trajectory_count for index in range(request.stage)
    }
    if not valid_replay_counts(counts, request.stage):
        raise ValueError("Invalid DDPM replay trajectory allocation.")
    checkpoint_sha256 = file_hash(request.checkpoint)

    def sample_normalized(task: ReplayTaskContext, normalized_goals: torch.Tensor):
        return (
            old.sample(
                task.trajectory_count,
                seed=task.seed,
                task_ids=task.labels,
                goals=normalized_goals,
                execution_mode=request.execution_mode,
            ),
            None,
        )

    def validate_projected(projected: torch.Tensor, _trajectory_count: int) -> None:
        if projected.shape[1] != request.horizon:
            raise ValueError("Replay checkpoint horizon differs from selected backend.")
        if request.action_space == "pre-tanh" and torch.any(
            projected[..., PHYSICAL_ACTION_SLICE].abs() > 1
        ):
            raise ValueError("Generated replay actions exceed environment bounds.")

    def build_source(task: ReplayTaskContext, _sampling_details) -> ReplaySource:
        return {
            "kind": "generated",
            "checkpoint": str(request.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "sampling_seed": task.seed,
            "condition": task.index,
            "trajectory_count": task.trajectory_count,
            "diffusion_steps": old.config.steps,
            # One batched denoiser call per reverse-diffusion step.
            "neural_function_evaluations": old.config.steps,
            "success": None,
            "return": None,
            "training_configuration_indices": task.selected_configuration_indices,
            "training_configuration_hashes": training_configuration_hashes(task),
        }

    result = generate_replay_groups(
        request,
        counts=counts,
        metadata=old_meta,
        dynamic_normalizer=old_norm,
        goal_normalizer=old_goal_norm,
        sample_normalized=sample_normalized,
        build_source=build_source,
        validate_projected=validate_projected,
        progress_message=progress_message,
        progress_heading="Generating replay",
        progress_description=lambda name: f"Replay: {name}",
    )
    return result
