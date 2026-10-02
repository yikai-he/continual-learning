"""Backend adapters and the shared policy-facing environment contract."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

CURRENT_STATE_DIM = 18
GOAL_DIM = 3
OBSERVATION_SHAPE = (2 * CURRENT_STATE_DIM + GOAL_DIM,)
ACTION_SHAPE = (4,)
CURRENT_STATE_SLICE = slice(0, CURRENT_STATE_DIM)
PREVIOUS_STATE_SLICE = slice(CURRENT_STATE_SLICE.stop, 2 * CURRENT_STATE_DIM)
GOAL_SLICE = slice(PREVIOUS_STATE_SLICE.stop, OBSERVATION_SHAPE[0])


class EpisodeSuccessWrapper(gym.Wrapper):
    """Expose any-step episode success in terminal ``is_success`` metadata."""

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
            info["is_success"] = self._episode_success
        return observation, reward, terminated, truncated, info
