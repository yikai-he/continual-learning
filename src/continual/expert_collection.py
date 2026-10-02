"""Standalone expert trajectory collection without continual training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import SAC
from tqdm.auto import tqdm

from src.continual.collector import collect_trajectories, collect_trajectory
from src.continual.diffusion_data import pack_trajectories, unpack_trajectories
from src.continual.policy_adapter import SB3SACPolicy
from src.envs import GOAL_SLICE, horizon_for_backend, make_env


@dataclass(frozen=True)
class CollectionConfig:
    task: str
    backend: str
    reward_function_version: str
    expert: Path
    trajectories: int
    initial_seed: int
    horizon: int
    deterministic: bool
    successful_only: bool
    device: str
    output: Path


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value


def load_collection_config(path: str | Path) -> CollectionConfig:
    """Load the small collection preset using established config section names."""
    source = Path(path)
    values = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "config")
    if values.get("experiment") != "trajectory_collection":
        raise ValueError("experiment must be trajectory_collection.")
    environment = _mapping(values.get("environment"), "environment")
    continual = _mapping(values.get("continual"), "continual")
    collection = _mapping(values.get("collection"), "collection")
    runtime = _mapping(values.get("runtime"), "runtime")
    tasks = continual.get("tasks")
    experts = continual.get("experts")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str):
        raise ValueError("continual.tasks must contain exactly one task.")
    if not isinstance(experts, dict) or not isinstance(experts.get(tasks[0]), str):
        raise ValueError("continual.experts must map the task to its expert path.")
    config = CollectionConfig(
        task=tasks[0],
        backend=environment.get("backend"),
        reward_function_version=environment.get("reward_function_version"),
        expert=Path(experts[tasks[0]]).expanduser().resolve(),
        trajectories=continual.get("trajectories_per_task"),
        initial_seed=runtime.get("seed"),
        horizon=collection.get("horizon"),
        deterministic=collection.get("deterministic"),
        successful_only=collection.get("successful_only", False),
        device=runtime.get("device"),
        output=Path(runtime.get("output", "")).expanduser().resolve(),
    )
    if config.backend not in ("metaworld-v3", "kuka-v2"):
        raise ValueError("environment.backend is unsupported.")
    if config.reward_function_version not in ("v1", "v2"):
        raise ValueError("environment.reward_function_version must be v1 or v2.")
    if not config.expert.is_file():
        raise FileNotFoundError(f"Expert model not found: {config.expert}")
    if isinstance(config.trajectories, bool) or not isinstance(config.trajectories, int) or config.trajectories < 1:
        raise ValueError("continual.trajectories_per_task must be positive.")
    if isinstance(config.initial_seed, bool) or not isinstance(config.initial_seed, int):
        raise ValueError("runtime.seed must be an integer.")
    if config.horizon != horizon_for_backend(config.backend):
        raise ValueError("collection.horizon must match the selected backend.")
    if config.deterministic is not True:
        raise ValueError("collection.deterministic must be true.")
    if not isinstance(config.successful_only, bool):
        raise ValueError("collection.successful_only must be boolean.")
    if not isinstance(config.device, str) or not config.device:
        raise ValueError("runtime.device must be a non-empty string.")
    if not str(runtime.get("output", "")).strip():
        raise ValueError("runtime.output must be set.")
    return config


def collect_successful_trajectories(
    env,
    policy,
    task_id: str,
    *,
    target_successes: int,
    initial_seed: int,
    deterministic: bool = True,
    progress: bool = False,
) -> tuple[list, list]:
    """Attempt consecutive seeds until exactly ``target_successes`` are accepted."""
    if (
        isinstance(target_successes, bool)
        or not isinstance(target_successes, int)
        or target_successes < 1
    ):
        raise ValueError("target_successes must be a positive integer.")
    accepted, attempts = [], []
    with tqdm(
        total=target_successes,
        desc=f"Accepted expert trajectories: {task_id}",
        unit="trajectory",
        disable=not progress,
    ) as progress_bar:
        while len(accepted) < target_successes:
            attempt_index = len(attempts)
            seed = initial_seed + attempt_index
            trajectory = collect_trajectory(
                env,
                policy,
                task_id,
                seed=seed,
                deterministic=deterministic,
            )
            is_accepted = bool(trajectory.success)
            attempts.append((seed, trajectory, is_accepted))
            if is_accepted:
                accepted.append(trajectory)
                progress_bar.update()
            progress_bar.set_postfix(Attempts=len(attempts))
    return accepted, attempts


def collect_expert_dataset(config: CollectionConfig) -> dict[str, Any]:
    """Collect, validate, and save one fixed-size expert dataset."""
    if config.output.exists():
        raise FileExistsError(f"Output directory already exists: {config.output}")
    model = SAC.load(config.expert, device=config.device)
    policy = SB3SACPolicy(model)
    env = make_env(
        config.backend,
        config.task,
        config.initial_seed,
        reward_function_version=config.reward_function_version,
    )
    try:
        if env.spec.max_episode_steps != config.horizon:
            raise ValueError("Environment horizon differs from collection config.")
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError("Expert/environment spaces differ.")
        if config.successful_only:
            trajectories, attempts = collect_successful_trajectories(
                env,
                policy,
                config.task,
                target_successes=config.trajectories,
                initial_seed=config.initial_seed,
                deterministic=config.deterministic,
                progress=True,
            )
        else:
            trajectories = collect_trajectories(
                env,
                policy,
                config.task,
                episodes=config.trajectories,
                initial_seed=config.initial_seed,
                deterministic=config.deterministic,
                progress=True,
                description=f"Expert collection: {config.task}",
            )
            attempts = [
                (config.initial_seed + index, trajectory, True)
                for index, trajectory in enumerate(trajectories)
            ]
        action_low = np.asarray(env.action_space.low)
        action_high = np.asarray(env.action_space.high)
    finally:
        env.close()

    packed = pack_trajectories(
        trajectories, horizon=config.horizon, task_names=(config.task,)
    )
    observations, actions = unpack_trajectories(packed)
    observation_values = observations.numpy()
    action_values = actions.numpy()
    rewards = np.stack([item.rewards for item in trajectories])
    goals = observation_values[:, 0, GOAL_SLICE]
    returns = np.asarray([item.episode_return for item in trajectories])
    lengths = np.asarray([item.length for item in trajectories])
    successes = np.asarray([item.success for item in trajectories])
    nonfinite_observations = int((~np.isfinite(observation_values)).sum())
    nonfinite_actions = int((~np.isfinite(action_values)).sum())
    action_oob = (action_values < action_low) | (action_values > action_high)
    accepted_seeds = [seed for seed, _, accepted in attempts if accepted]
    attempt_ledger = [
        {
            "attempt_index": attempt_index,
            "seed": seed,
            "accepted": accepted,
            "length": int(item.length),
            "success": bool(item.success),
            "return": float(item.episode_return),
            "observations_finite": bool(np.isfinite(item.observations).all()),
            "actions_finite": bool(np.isfinite(item.actions).all()),
            "actions_within_bounds": bool(
                np.all(item.actions >= action_low) and np.all(item.actions <= action_high)
            ),
            "action_bound_violation": bool(
                np.any(item.actions < action_low) or np.any(item.actions > action_high)
            ),
            "goal": item.observations[0, GOAL_SLICE].tolist(),
            "final_success": bool(item.final_success),
        }
        for attempt_index, (seed, item, accepted) in enumerate(attempts)
    ]
    episodes = [entry for entry in attempt_ledger if entry["accepted"]]
    report = {
        "task": config.task,
        "backend": config.backend,
        "expert": str(config.expert),
        "deterministic": config.deterministic,
        "collection_mode": (
            "successful-only" if config.successful_only else "fixed-attempts"
        ),
        "target_trajectories": config.trajectories,
        "initial_seed": config.initial_seed,
        "accepted_seeds": accepted_seeds,
        "attempt_count": len(attempts),
        "rejected_attempt_count": len(attempts) - len(trajectories),
        "trajectory_count": len(trajectories),
        "success_count": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "trajectory_lengths": sorted(np.unique(lengths).tolist()),
        "observation_shape": list(observation_values.shape),
        "action_shape": list(action_values.shape),
        "reward_shape": list(rewards.shape),
        "observation_dtype": str(observation_values.dtype),
        "action_dtype": str(action_values.dtype),
        "reward_dtype": str(rewards.dtype),
        "nonfinite_observation_count": nonfinite_observations,
        "nonfinite_action_count": nonfinite_actions,
        "action_out_of_bounds_count": int(action_oob.sum()),
        "distinct_goal_count": int(np.unique(goals, axis=0).shape[0]),
        "observation_contract": {
            "current_frame": "obs[0:18]",
            "previous_frame": "obs[18:36]",
            "goal": "obs[36:39]",
            "previous_frame_aligned": bool(
                np.array_equal(observation_values[:, 0, 18:36], observation_values[:, 0, 0:18])
                and np.array_equal(observation_values[:, 1:, 18:36], observation_values[:, :-1, 0:18])
            ),
            "goal_constant_per_trajectory": bool(
                np.array_equal(
                    observation_values[:, :, GOAL_SLICE],
                    np.broadcast_to(goals[:, None], observation_values[:, :, GOAL_SLICE].shape),
                )
            ),
        },
        "episodes": episodes,
        "attempts": attempt_ledger,
    }
    config.output.mkdir(parents=True)
    np.savez_compressed(
        config.output / "real_trajectories.npz",
        **{
            f"episode_{index}_{key}": value
            for index, trajectory in enumerate(trajectories)
            for key, value in asdict(trajectory).items()
        },
    )
    np.save(config.output / "observations.npy", observation_values)
    np.save(config.output / "actions.npy", action_values)
    np.save(config.output / "rewards.npy", rewards)
    (config.output / "metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (config.output / "attempt_ledger.json").write_text(
        json.dumps(attempt_ledger, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
