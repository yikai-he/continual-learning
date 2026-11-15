"""Fresh historical SAC-expert rollouts for privileged replay comparison."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from stable_baselines3 import SAC

from continual_manipulation.environments import EnvironmentSettings, make_env
from src.support.expert_manifest import ExpertIdentity

from .collector import collect_trajectories
from .diffusion_data import PHYSICAL_ACTION_SLICE, pack_trajectories
from .expert_space import validate_expert_spaces
from .policy_adapter import SB3SACPolicy
from .task_bank import selected_configuration, training_bank


EXPERT_RECOLLECTION_SEED_OFFSET = 200_000


@dataclass(frozen=True)
class ExpertRecollectionRequest:
    """Inputs required to recollect historical trajectories for one stage."""

    stage: int
    trajectory_counts: dict[int, int]
    task_names: tuple[str, ...]
    experts: dict[str, Path]
    expert_hashes: dict[str, str]
    expert_identities: dict[str, ExpertIdentity]
    environment: EnvironmentSettings
    runtime_seed: int
    device: str
    horizon: int
    collection_mode: str
    max_attempt_multiplier: int
    progress: bool


def _trajectory_hash(group: torch.Tensor) -> str:
    values = group.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def collect_historical_expert_replay(
    request: ExpertRecollectionRequest,
) -> tuple[dict[int, torch.Tensor], dict[str, dict[str, object]]]:
    """Collect fresh deterministic rollouts for every historical task."""
    if set(request.trajectory_counts) != set(range(request.stage)):
        raise ValueError("Expert recollection allocation does not match historical tasks.")
    groups: dict[int, torch.Tensor] = {}
    sources: dict[str, dict[str, object]] = {}
    for task_index in range(request.stage):
        count = request.trajectory_counts[task_index]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid expert recollection trajectory count.")
        if not count:
            continue
        task_name = request.task_names[task_index]
        checkpoint = request.experts[task_name]
        started = time.perf_counter()
        model = SAC.load(checkpoint, device=request.device)
        env = make_env(
            request.environment.backend,
            task_name,
            request.runtime_seed,
            **request.environment.kwargs(),
        )
        attempted_configurations: list[dict[str, object]] = []
        bank = training_bank(
            task_name,
            request.runtime_seed,
            backend=request.environment.backend,
        )
        initial_seed = (
            request.runtime_seed
            + EXPERT_RECOLLECTION_SEED_OFFSET
            + request.stage * 10_000
            + task_index * 1000
        )
        try:
            validate_expert_spaces(
                model, env, request.expert_identities[task_name]
            )
            trajectories, collection = collect_trajectories(
                env,
                SB3SACPolicy(model),
                task_name,
                episodes=count,
                initial_seed=initial_seed,
                deterministic=True,
                on_reset=lambda e, obs, seed, attempt: attempted_configurations.append(
                    selected_configuration(
                        e, bank, obs, attempt, reset_seed=seed
                    )
                ),
                progress=request.progress,
                description=f"Expert recollection: {task_name}",
                collection_mode=request.collection_mode,
                max_attempts=count * request.max_attempt_multiplier,
                return_metadata=True,
            )
        finally:
            env.close()
        selected_attempts = collection["selected_attempt_indices"]
        configurations = [
            attempted_configurations[index] for index in selected_attempts
        ]
        group = pack_trajectories(
            trajectories,
            horizon=request.horizon,
            task_names=request.task_names,
        )
        actions = group[..., PHYSICAL_ACTION_SLICE]
        if torch.any(actions < -1) or torch.any(actions > 1):
            raise ValueError("Recollected expert actions exceed environment bounds.")
        hashes = [_trajectory_hash(values) for values in group]
        successes = [bool(trajectory.success) for trajectory in trajectories]
        final_successes = [
            bool(trajectory.final_success) for trajectory in trajectories
        ]
        returns = [float(trajectory.episode_return) for trajectory in trajectories]
        reference_goals = group[:, 0, 36:39].tolist()
        sources[task_name] = {
            "kind": "expert_recollection",
            "historical_task": task_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": request.expert_hashes[task_name],
            "requested_trajectory_count": count,
            "trajectory_count": len(trajectories),
            "collection_action_mode": "deterministic",
            "collection_mode": request.collection_mode,
            "collection_seed_namespace": {
                "offset": EXPERT_RECOLLECTION_SEED_OFFSET,
                "initial_seed": initial_seed,
            },
            "reset_seeds": collection["selected_seeds"],
            "source_attempt_ids": selected_attempts,
            "trajectory_sha256": hashes,
            "exact_duplicate_trajectory_count": len(hashes) - len(set(hashes)),
            "collection_configurations": configurations,
            "unique_configuration_count": len(
                {configuration["task_data_sha256"] for configuration in configurations}
            ),
            "success_count": sum(successes),
            "success_rate": collection["selected_success_fraction"],
            "final_success_count": sum(final_successes),
            "final_success_rate": sum(final_successes) / len(final_successes),
            "episode_lengths": [int(t.length) for t in trajectories],
            "episode_returns": returns,
            "mean_return": collection["selected_mean_return"],
            "std_return": collection["selected_return_std"],
            "collection_seconds": time.perf_counter() - started,
            "collection": {
                key: value
                for key, value in collection.items()
                if key not in ("selected_attempt_indices", "selected_seeds")
            },
            "reference_goals": reference_goals,
        }
        groups[task_index] = group
    return groups, sources
