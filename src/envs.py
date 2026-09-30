"""MetaWorld MT1 environment construction and validation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import metaworld  # noqa: F401  # Registers the Meta-World Gymnasium environments.
import numpy as np
from numpy.typing import NDArray

CURRENT_STATE_DIM = 18
GOAL_DIM = 3
OBSERVATION_SHAPE = (2 * CURRENT_STATE_DIM + GOAL_DIM,)
ACTION_SHAPE = (4,)
CURRENT_STATE_SLICE = slice(0, CURRENT_STATE_DIM)
PREVIOUS_STATE_SLICE = slice(CURRENT_STATE_SLICE.stop, 2 * CURRENT_STATE_DIM)
GOAL_SLICE = slice(PREVIOUS_STATE_SLICE.stop, OBSERVATION_SHAPE[0])
EPISODE_HORIZON = 200


class MetaWorldObservationWrapper(gym.ObservationWrapper):
    """Expose the physical 39D MetaWorld v3 environment-policy contract.

    MetaWorld reports finite bounds for its final 3D goal even though sampled
    goals may exceed them. The wrapper relaxes only those bounds and converts
    observations to float32 without clipping their physical values. Diffusion
    uses a separate encoding derived from stored physical trajectories.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(
                "Expected a Box observation space from MetaWorld, "
                f"got {type(space).__name__}."
            )
        if space.shape != OBSERVATION_SHAPE:
            raise ValueError(
                f"Expected MetaWorld observation shape {OBSERVATION_SHAPE}, "
                f"got {space.shape}."
            )

        low = np.asarray(space.low, dtype=np.float32).copy()
        high = np.asarray(space.high, dtype=np.float32).copy()
        low[GOAL_SLICE] = -np.inf
        high[GOAL_SLICE] = np.inf
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=OBSERVATION_SHAPE, dtype=np.float32
        )

    def observation(self, observation: Any) -> NDArray[np.float32]:
        """Validate an observation without clipping its goal coordinates."""
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != OBSERVATION_SHAPE:
            raise ValueError(
                f"Expected MetaWorld observation shape {OBSERVATION_SHAPE}, "
                f"got {array.shape}."
            )
        return array


class EpisodeSuccessWrapper(gym.Wrapper):
    """Accumulate MetaWorld's per-step success until the episode ends."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._episode_success = False

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._episode_success = False
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        self._episode_success |= bool(info.get("success", False))
        if terminated or truncated:
            # Stable-Baselines3's EvalCallback reads is_success on the final step.
            info["is_success"] = self._episode_success
        return observation, reward, terminated, truncated, info


def make_metaworld_env(
    task: str,
    seed: int,
    render_mode: str | None = None,
    *,
    reward_function_version: str = "v2",
) -> gym.Env:
    """Create one validated MetaWorld v3 MT1 task environment.

    The returned Gymnasium environment has a 200-step horizon, 39D float32
    observations, 4D continuous actions, and any-step episode success. The
    caller owns the environment and must close it.
    """
    if reward_function_version not in ("v1", "v2"):
        raise ValueError("reward_function_version must be v1 or v2.")
    env = gym.make(
        "Meta-World/MT1",
        env_name=task,
        seed=seed,
        use_one_hot=False,
        max_episode_steps=EPISODE_HORIZON,
        terminate_on_success=False,
        reward_function_version=reward_function_version,
        render_mode=render_mode,
        disable_env_checker=True,
    )
    env = MetaWorldObservationWrapper(env)
    env = EpisodeSuccessWrapper(env)

    if env.observation_space.shape != OBSERVATION_SHAPE:
        env.close()
        raise ValueError(
            f"Expected observation shape {OBSERVATION_SHAPE}, "
            f"got {env.observation_space.shape}."
        )
    if not isinstance(env.action_space, gym.spaces.Box):
        env.close()
        raise TypeError(
            f"Expected a Box action space, got {type(env.action_space).__name__}."
        )
    if env.action_space.shape != ACTION_SHAPE:
        env.close()
        raise ValueError(
            f"Expected action shape {ACTION_SHAPE}, got {env.action_space.shape}."
        )
    return env
