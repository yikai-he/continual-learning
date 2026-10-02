"""Gymnasium compatibility and construction for legacy KUKA-v2 tasks."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from . import ACTION_SHAPE, GOAL_SLICE, OBSERVATION_SHAPE, EpisodeSuccessWrapper

KUKA_TASKS = (
    "kuka-reach-v2",
    "kuka-push-v2",
    "kuka-hammer-v2",
    "kuka-handle-press-side-v2",
    "kuka-button-press-v2",
)
KUKA_HORIZON = 300


class KukaV2GymnasiumAdapter(gym.Env):
    """Expose one legacy Gym KUKA environment through the Gymnasium API."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, raw_env: Any, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported KUKA render mode: {render_mode!r}.")
        self.raw_env = raw_env
        self.render_mode = render_mode
        self.observation_space = self._convert_box(raw_env.observation_space)
        self.action_space = self._convert_box(raw_env.action_space)
        self.spec = gym.envs.registration.EnvSpec(
            id=type(raw_env).__name__, max_episode_steps=KUKA_HORIZON
        )
        self._elapsed_steps = 0
        self._needs_reset = True

    @staticmethod
    def _convert_box(space: Any) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=np.asarray(space.low, dtype=space.dtype),
            high=np.asarray(space.high, dtype=space.dtype),
            shape=space.shape,
            dtype=space.dtype,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("KUKA-v2 environments do not support reset options.")
        if seed is not None:
            self.raw_env.seed(seed)
        observation = np.asarray(self.raw_env.reset(), dtype=np.float32)
        self._elapsed_steps = 0
        self._needs_reset = False
        return observation, {}

    def step(
        self, action: Any
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("KUKA environment must be reset before stepping.")
        observation, reward, _legacy_done, info = self.raw_env.step(action)
        self._elapsed_steps += 1
        truncated = self._elapsed_steps == KUKA_HORIZON
        if truncated:
            self._needs_reset = True
        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            False,
            truncated,
            dict(info),
        )

    def render(self) -> Any:
        if self.render_mode == "human":
            return self.raw_env.render()
        return None

    def close(self) -> None:
        self.raw_env.close()


def make_kuka_v2_env(
    task: str,
    seed: int,
    render_mode: str | None = None,
) -> gym.Env:
    """Construct a registered KUKA-v2 task without importing MetaWorld-v3."""
    if task not in KUKA_TASKS:
        raise ValueError(f"KUKA-v2 backend supports {KUKA_TASKS!r}, got {task!r}.")

    from metaworld.envs.mujoco.env_dict import KUKA_V2_ENVIRONMENTS

    if task not in KUKA_V2_ENVIRONMENTS:
        raise ValueError(f"KUKA-v2 task is not registered: {task!r}.")
    raw_env = KUKA_V2_ENVIRONMENTS[task]()
    raw_env._set_task_called = True
    raw_env._freeze_rand_vec = False
    raw_env.seeded_rand_vec = True
    raw_env._partially_observable = False

    env = EpisodeSuccessWrapper(
        KukaV2GymnasiumAdapter(raw_env, render_mode=render_mode)
    )
    observation, _ = env.reset(seed=seed)
    if env.observation_space.shape != OBSERVATION_SHAPE:
        env.close()
        raise ValueError(
            f"Expected observation shape {OBSERVATION_SHAPE}, "
            f"got {env.observation_space.shape}."
        )
    if env.action_space.shape != ACTION_SHAPE:
        env.close()
        raise ValueError(
            f"Expected action shape {ACTION_SHAPE}, got {env.action_space.shape}."
        )
    if observation.dtype != np.float32:
        env.close()
        raise TypeError(f"Expected float32 observations, got {observation.dtype}.")
    if not np.any(observation[GOAL_SLICE]):
        env.close()
        raise ValueError("KUKA goal is not visible in observation[36:39].")
    return env
