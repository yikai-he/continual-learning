"""Single- and repeated-episode collection through the common policy boundary.

Use environments constructed through ``continual_manipulation``. Preserve
Gymnasium reset's (observation, info) and step's five returns:
    observation, reward, terminated, truncated, info
End on terminated or truncated; do not use the old Gym four-return API.
Success must accumulate info['success'] across the episode, consistent with
EpisodeSuccessWrapper's terminal info['is_success'] and
scripts/evaluation/evaluate_sac.py.
Success alone does not end an episode. Repeated collection optionally prefers
successful episodes; individual trajectories are never padded or modified.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from continual_manipulation.evaluation.rollout import (
    Trajectory as _Trajectory,
    collect_trajectory as _collect_trajectory,
)
from continual_manipulation.interfaces import Policy as _Policy
from tqdm.auto import tqdm


if TYPE_CHECKING:
    import gymnasium as gym


EpisodeResetCallback = Callable[[object, np.ndarray, int | None, int], None]


def collect_trajectories(
    env: gym.Env,
    policy: _Policy,
    task_id: str,
    *,
    episodes: int,
    initial_seed: int,
    deterministic: bool = True,
    on_reset: EpisodeResetCallback | None = None,
    progress: bool = False,
    description: str | None = None,
    collection_mode: str = "all",
    max_attempts: int | None = None,
    return_metadata: bool = False,
) -> list[_Trajectory] | tuple[list[_Trajectory], dict[str, object]]:
    """Collect ordered episodes using consecutive deterministic reset seeds.

    The caller owns the environment and policy. ``on_reset`` receives the
    zero-based episode index in addition to the single-rollout callback values.
    ``all`` preserves the original ordered fixed-attempt behavior.
    ``success_preferred`` tries up to ``max_attempts``, retaining successes
    first and filling any deficit with failures in attempt order. No trajectory
    is padded, packed, or persisted here.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer.")
    if collection_mode not in ("all", "success_preferred"):
        raise ValueError("collection_mode must be all or success_preferred.")
    attempt_limit = episodes if collection_mode == "all" else max_attempts
    if (
        isinstance(attempt_limit, bool)
        or not isinstance(attempt_limit, int)
        or attempt_limit < episodes
    ):
        raise ValueError("max_attempts must be an integer at least as large as episodes.")

    successful: list[tuple[int, _Trajectory]] = []
    unsuccessful: list[tuple[int, _Trajectory]] = []
    attempted: list[tuple[int, _Trajectory]] = []
    model_device = getattr(getattr(policy, "model", None), "device", None)
    resolved_model_device = torch.device(model_device) if model_device is not None else None
    cuda_devices = []
    if resolved_model_device is not None and resolved_model_device.type == "cuda":
        cuda_devices = [
            resolved_model_device.index
            if resolved_model_device.index is not None
            else torch.cuda.current_device()
        ]
    # SB3 SAC draws stochastic actions from PyTorch's global RNG. Forking it
    # makes a collection repeatable without advancing later training RNG state.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(initial_seed)
        for attempt in tqdm(
            range(attempt_limit),
            desc=description,
            unit="trajectory",
            disable=not progress,
        ):
            reset_callback = (
                None
                if on_reset is None
                else lambda current_env, observation, seed, index=attempt: on_reset(
                    current_env, observation, seed, index
                )
            )
            trajectory = _collect_trajectory(
                env,
                policy,
                task_id,
                seed=initial_seed + attempt,
                deterministic=deterministic,
                on_reset=reset_callback,
            )
            target = successful if bool(getattr(trajectory, "success", False)) else unsuccessful
            target.append((attempt, trajectory))
            attempted.append((attempt, trajectory))
            if collection_mode == "all" and attempt + 1 == episodes:
                break
            if collection_mode == "success_preferred" and len(successful) == episodes:
                break

    if collection_mode == "all":
        selected = attempted
        accepted_successful = len(successful)
        accepted_unsuccessful = len(unsuccessful)
        fallback_count = 0
    else:
        selected = successful[:episodes]
        fallback_count = max(0, episodes - len(selected))
        selected.extend(unsuccessful[:fallback_count])
        accepted_successful = len(selected) - fallback_count
        accepted_unsuccessful = fallback_count
    if len(selected) != episodes:
        raise RuntimeError(
            f"Collected only {len(selected)} valid trajectories; requested {episodes}."
        )
    selected_attempts = [attempt for attempt, _ in selected]
    trajectories = [trajectory for _, trajectory in selected]
    attempts = len(successful) + len(unsuccessful)
    selected_returns = []
    for trajectory in trajectories:
        try:
            selected_returns.append(float(trajectory.episode_return))
        except (AttributeError, TypeError, ValueError):
            selected_returns.append(0.0)
    metadata = {
        "mode": collection_mode,
        "expert_action_mode": "deterministic" if deterministic else "stochastic",
        "requested_trajectories": episodes,
        "maximum_attempts": attempt_limit,
        "actual_attempts": attempts,
        "successful_attempts": len(successful),
        "failed_attempts": len(unsuccessful),
        "accepted_successful": accepted_successful,
        "accepted_unsuccessful": accepted_unsuccessful,
        "accepted_fallback": fallback_count,
        "rejected_trajectories": attempts - len(selected),
        "selected_attempt_indices": selected_attempts,
        "selected_seeds": [initial_seed + attempt for attempt in selected_attempts],
        "selected_success_fraction": float(
            np.mean([bool(getattr(trajectory, "success", False)) for trajectory in trajectories])
        ),
        "selected_mean_return": float(np.mean(selected_returns)),
        "selected_return_std": float(np.std(selected_returns)),
    }
    return (trajectories, metadata) if return_metadata else trajectories
