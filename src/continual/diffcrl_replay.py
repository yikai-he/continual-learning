"""Previous-task diffusion replay generation for one DiffCRL stage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from src.envs import (
    CURRENT_STATE_SLICE,
    GOAL_SLICE,
    PREVIOUS_STATE_SLICE,
)
from src.support.reproducibility import file_hash, state_hash
from tqdm.auto import tqdm

from .diffusion_data import (
    PHYSICAL_ACTION_SLICE,
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
    bc_clamp_epsilon: float
    progress: bool
    trajectory_counts: dict[int, int] | None = None
    execution_mode: str = "legacy"


@dataclass
class ReplayResult:
    """Generated physical trajectory groups and their provenance by old task."""

    groups: dict[int, torch.Tensor]
    sources: dict[str, ReplaySource]


def generate_previous_task_replay(
    request: ReplayRequest,
    *,
    progress_message: ProgressMessage | None = None,
) -> ReplayResult:
    """Reconstruct old training goals and decode sampled 22D dynamics to 43D replay.

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
    )
    old.denoiser.to(request.device)
    if state_hash(old) != request.expected_state_hash:
        raise RuntimeError("Previous checkpoint differs from current diffusion memory.")

    groups: dict[int, torch.Tensor] = {}
    sources: dict[str, ReplaySource] = {}
    counts = request.trajectory_counts or {
        index: request.trajectory_count for index in range(request.stage)
    }
    if not valid_replay_counts(counts, request.stage):
        raise ValueError("Invalid DDPM replay trajectory allocation.")
    for old_index in range(request.stage):
        trajectory_count = counts[old_index]
        if trajectory_count == 0:
            continue
        name = request.task_names[old_index]
        if progress_message is not None:
            progress_message(
                "Generating replay",
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
        with tqdm(
            total=trajectory_count,
            desc=f"Replay: {name}",
            unit="trajectory",
            disable=not request.progress,
        ) as replay_progress:
            sampled = old_norm.denormalize(
                old.sample(
                    trajectory_count,
                    seed=task.seed,
                    task_ids=task.labels,
                    goals=normalize_replay_goals(task, old_goal_norm),
                    execution_mode=request.execution_mode,
                )
            )
            replay_progress.update(trajectory_count)

        replay = decode_generated_replay(
            sampled,
            task.goals,
            action_space=request.action_space,
            action_projection=request.action_projection,
            bc_clamp_epsilon=request.bc_clamp_epsilon,
        )
        projected = replay.projected
        assert torch.equal(
            projected[..., PREVIOUS_STATE_SLICE],
            torch.cat(
                (
                    projected[:, :1, CURRENT_STATE_SLICE],
                    projected[:, :-1, CURRENT_STATE_SLICE],
                ),
                dim=1,
            ),
        )
        assert torch.equal(
            projected[..., GOAL_SLICE],
            task.goals[:, None].expand(-1, request.horizon, -1),
        )
        if projected.shape[1] != request.horizon:
            raise ValueError("Replay checkpoint horizon differs from selected backend.")
        if request.action_space == "pre-tanh" and torch.any(
            projected[..., PHYSICAL_ACTION_SLICE].abs() > 1
        ):
            raise ValueError("Decoded replay actions exceed environment bounds.")
        sources[name] = {
            "kind": "generated",
            "checkpoint": str(request.checkpoint),
            "checkpoint_sha256": file_hash(request.checkpoint),
            "sampling_seed": task.seed,
            "condition": old_index,
            "trajectory_count": trajectory_count,
            "diffusion_steps": old.config.steps,
            # One batched denoiser call per reverse-diffusion step.
            "neural_function_evaluations": old.config.steps,
            "success": None,
            "return": None,
            "generated_latent_abs_gt_bc_cap_fraction": replay.latent_cap_fraction,
            **replay.action_diagnostics,
        }
        sources[name]["training_configuration_indices"] = (
            task.selected_configuration_indices
        )
        sources[name]["training_configuration_hashes"] = training_configuration_hashes(
            task
        )
        validate_stage_trajectory_group(
            projected, stage=request.stage, task=name, source=sources[name]
        )
        groups[old_index] = projected
    del old
    return ReplayResult(groups=groups, sources=sources)
