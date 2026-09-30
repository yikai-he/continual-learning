"""Single- and repeated-episode collection through the common policy boundary.

Use environments constructed by src.envs.make_metaworld_env. Preserve
Gymnasium reset's (observation, info) and step's five returns:
    observation, reward, terminated, truncated, info
End on terminated or truncated; do not use the old Gym four-return API.
Success must accumulate info['success'] across the episode, consistent with
EpisodeSuccessWrapper's terminal info['is_success'] and
scripts/evaluation/evaluate_sac.py.
Success alone does not end an episode. No filtering or padding is performed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from src.envs import ACTION_SHAPE, OBSERVATION_SHAPE

from .policy_adapter import Policy
from .schema import Trajectory, validate_trajectory

if TYPE_CHECKING:
    import gymnasium as gym


EpisodeResetCallback = Callable[[object, np.ndarray, int | None, int], None]


def collect_trajectory(
    env: gym.Env,
    policy: Policy,
    task_id: str,
    *,
    seed: int | None = None,
    deterministic: bool = True,
    on_reset=None,
) -> Trajectory:
    """Roll out one policy episode and return aligned, validated trajectory data.

    Observations are captured immediately before their corresponding actions.
    ``on_reset`` is intended for read-only provenance capture after reset; the
    callback can access the mutable environment, so callers must not change it.
    The caller owns ``env``. A missing end signal by its declared horizon is
    treated as an invalid environment rather than fabricated truncation.
    """
    horizon = getattr(env.spec, "max_episode_steps", None)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Environment must declare a positive max_episode_steps.")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty scalar task-name string.")
    if (
        env.observation_space.shape != OBSERVATION_SHAPE
        or env.action_space.shape != ACTION_SHAPE
    ):
        raise ValueError("Environment spaces must match MetaWorld v3.")
    observation, _ = env.reset(seed=seed)
    if on_reset is not None:
        on_reset(env, np.asarray(observation).copy(), seed)
    observations, actions, rewards = [], [], []
    episode_success = False
    for _ in range(horizon):
        # Snapshot before act/step: policies and environments may reuse arrays.
        observation = np.asarray(observation).copy()
        if observation.shape != OBSERVATION_SHAPE or not np.isfinite(observation).all():
            raise ValueError("Expected one finite MetaWorld observation.")
        observations.append(observation.copy())
        action = np.asarray(policy.act(observation, deterministic=deterministic)).copy()
        if action.shape != ACTION_SHAPE or not np.isfinite(action).all():
            raise ValueError("Expected one finite MetaWorld action.")
        actions.append(action.copy())
        observation, reward, terminated, truncated, info = env.step(action)
        if np.ndim(reward) != 0 or not np.isfinite(reward):
            raise ValueError("Expected a finite scalar reward.")
        rewards.append(float(reward))
        episode_success |= bool(info.get("success", False))
        if terminated or truncated:
            trajectory = Trajectory(
                observations=np.stack(observations),
                actions=np.stack(actions),
                rewards=np.asarray(rewards, dtype=np.float64),
                task_id=task_id,
                episode_return=sum(rewards),
                success=episode_success,
                length=len(rewards),
            )
            validate_trajectory(trajectory, horizon=horizon)
            return trajectory
    raise ValueError("Environment did not terminate or truncate within its horizon.")


def collect_trajectories(
    env: gym.Env,
    policy: Policy,
    task_id: str,
    *,
    episodes: int,
    initial_seed: int,
    deterministic: bool = True,
    on_reset: EpisodeResetCallback | None = None,
    progress: bool = False,
    description: str | None = None,
) -> list[Trajectory]:
    """Collect ordered episodes using consecutive deterministic reset seeds.

    The caller owns the environment and policy. ``on_reset`` receives the
    zero-based episode index in addition to the single-rollout callback values.
    No trajectory is filtered, padded, packed, or persisted here.
    """
    trajectories = []
    for episode in tqdm(
        range(episodes),
        desc=description,
        unit="trajectory",
        disable=not progress,
    ):
        reset_callback = (
            None
            if on_reset is None
            else lambda current_env, observation, seed, index=episode: on_reset(
                current_env, observation, seed, index
            )
        )
        trajectories.append(
            collect_trajectory(
                env,
                policy,
                task_id,
                seed=initial_seed + episode,
                deterministic=deterministic,
                on_reset=reset_callback,
            )
        )
    return trajectories
