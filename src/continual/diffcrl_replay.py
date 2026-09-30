"""Previous-task diffusion replay generation for one DiffCRL stage."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.envs import (
    CURRENT_STATE_SLICE,
    EPISODE_HORIZON,
    GOAL_SLICE,
    PREVIOUS_STATE_SLICE,
)
from src.support.reproducibility import file_hash, state_hash

from .diagnostics import generated_action_diagnostics
from .diffusion_data import (
    DIFFUSION_ACTION_SLICE,
    PHYSICAL_ACTION_SLICE,
    decode_trajectory,
    project_actions,
    transform_actions,
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
    runtime_seed: int
    device: str
    reward_function_version: str
    action_space: str
    action_clamp_epsilon: float
    action_projection: str
    bc_clamp_epsilon: float
    progress: bool


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

    Each prior task receives ``trajectory_count`` physical 200-step trajectories.
    """
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
    for old_index in range(request.stage):
        name = request.task_names[old_index]
        if progress_message is not None:
            progress_message(
                "Generating replay",
                Task=name,
                Trajectories=request.trajectory_count,
            )
        seed = request.runtime_seed + 100000 + request.stage * 1000 + old_index
        labels = torch.full((request.trajectory_count,), old_index, dtype=torch.long)
        source = old_meta["training_banks"][name]
        selected = sample_training_configurations(
            source,
            old_meta["train_configurations"][name]["indices"],
            request.trajectory_count,
            seed,
        )
        goals = torch.as_tensor(
            reconstruct_goals(
                source,
                selected,
                reward_function_version=request.reward_function_version,
            ),
            dtype=torch.float32,
        )
        with tqdm(
            total=request.trajectory_count,
            desc=f"Replay: {name}",
            unit="trajectory",
            disable=not request.progress,
        ) as replay_progress:
            sampled = old_norm.denormalize(
                old.sample(
                    request.trajectory_count,
                    seed=seed,
                    task_ids=labels,
                    goals=old_goal_norm.normalize(goals[:, None, :])[:, 0],
                )
            )
            replay_progress.update(request.trajectory_count)

        sampled_actions = sampled[..., DIFFUSION_ACTION_SLICE]
        latent_cap = float(np.arctanh(1 - request.bc_clamp_epsilon))
        latent_cap_fraction = (
            float((sampled_actions.abs() > latent_cap).float().mean())
            if request.action_space == "pre-tanh"
            else None
        )
        decoded_dynamic = transform_actions(sampled, request.action_space, decode=True)
        decoded = decode_trajectory(decoded_dynamic, goals)
        projected_dynamic = project_actions(decoded_dynamic, request.action_projection)
        projected = decode_trajectory(projected_dynamic, goals)
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
            goals[:, None].expand(-1, EPISODE_HORIZON, -1),
        )
        groups[old_index] = projected
        action_diagnostics = generated_action_diagnostics(
            decoded, projected, request.action_projection
        )
        if request.action_space == "pre-tanh" and torch.any(
            groups[old_index][..., PHYSICAL_ACTION_SLICE].abs() > 1
        ):
            raise ValueError("Decoded replay actions exceed environment bounds.")
        sources[name] = {
            "kind": "generated",
            "checkpoint": str(request.checkpoint),
            "checkpoint_sha256": file_hash(request.checkpoint),
            "sampling_seed": seed,
            "condition": old_index,
            "success": None,
            "return": None,
            "generated_latent_abs_gt_bc_cap_fraction": latent_cap_fraction,
            **action_diagnostics,
        }
        sources[name]["training_configuration_indices"] = selected
        sources[name]["training_configuration_hashes"] = [
            source["ordered_task_hashes"][i] for i in selected
        ]
    del old
    return ReplayResult(groups=groups, sources=sources)
