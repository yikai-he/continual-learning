"""MetaWorld-v3 construction and observation compatibility."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from . import ACTION_SHAPE, GOAL_SLICE, OBSERVATION_SHAPE, EpisodeSuccessWrapper

METAWORLD_V3_HORIZON = 200


class MetaWorldObservationWrapper(gym.ObservationWrapper):
    """Expose the physical 39D MetaWorld-v3 observation contract."""

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
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != OBSERVATION_SHAPE:
            raise ValueError(
                f"Expected MetaWorld observation shape {OBSERVATION_SHAPE}, "
                f"got {array.shape}."
            )
        return array


def make_metaworld_v3_env(
    task: str,
    seed: int,
    render_mode: str | None = None,
    *,
    reward_function_version: str = "v2",
) -> gym.Env:
    """Create one validated MetaWorld-v3 MT1 environment."""
    if reward_function_version not in ("v1", "v2"):
        raise ValueError("reward_function_version must be v1 or v2.")
    import metaworld  # noqa: F401

    env = gym.make(
        "Meta-World/MT1",
        env_name=task,
        seed=seed,
        use_one_hot=False,
        max_episode_steps=METAWORLD_V3_HORIZON,
        terminate_on_success=False,
        reward_function_version=reward_function_version,
        render_mode=render_mode,
        disable_env_checker=True,
    )
    env = EpisodeSuccessWrapper(MetaWorldObservationWrapper(env))
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
