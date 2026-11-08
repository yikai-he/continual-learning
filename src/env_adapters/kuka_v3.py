"""Public Gymnasium construction for isolated MetaWorld KUKA-v3 tasks."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from . import ACTION_SHAPE, GOAL_SLICE, OBSERVATION_SHAPE, EpisodeSuccessWrapper

KUKA_V3_HORIZON = 200
KUKA_V3_REGISTRATION_ID = "Meta-World-KUKA/MT1"
KUKA_V3_TASKS = (
    "kuka-reach-v3", "kuka-push-v3", "kuka-push-cube-v3", "kuka-handle-press-side-v3",
    "kuka-button-press-v3", "kuka-button-press-topdown-v3", "kuka-peg-unplug-side-v3",
    "kuka-hammer-v3", "kuka-drawer-close-v3", "kuka-faucet-open-v3",
    "kuka-faucet-close-v3", "kuka-window-open-v3", "kuka-window-close-v3",
    "kuka-door-open-v3", "kuka-pick-place-v3",
)

def kuka_v3_tasks(task: str, seed: int):
    if task not in KUKA_V3_TASKS: raise ValueError(f"KUKA-v3 backend does not support {task!r}.")
    import metaworld_kuka  # noqa: F401
    from metaworld import _make_tasks
    from metaworld_kuka.registration import KUKA_ENVIRONMENTS
    return _make_tasks(KUKA_ENVIRONMENTS, {task: {"args": [], "kwargs": {"task_id": 0}}},
                       {"partially_observable": False}, seed=seed)

def make_kuka_v3_env(task: str, seed: int, render_mode: str | None = None, *,
    reward_function_version: str = "v2") -> gym.Env:
    if task not in KUKA_V3_TASKS: raise ValueError(f"KUKA-v3 backend supports {KUKA_V3_TASKS!r}, got {task!r}.")
    if reward_function_version not in ("v1", "v2"): raise ValueError("reward_function_version must be v1 or v2.")
    import metaworld_kuka  # noqa: F401
    env = gym.make(KUKA_V3_REGISTRATION_ID, env_name=task, seed=seed,
        reward_function_version=reward_function_version,
        render_mode=render_mode, disable_env_checker=True)
    env = EpisodeSuccessWrapper(env); observation, _ = env.reset(seed=seed)
    if env.observation_space.shape != OBSERVATION_SHAPE or observation.shape != OBSERVATION_SHAPE: raise ValueError("KUKA-v3 observation contract must be 39D.")
    if observation.dtype != np.float32 or env.action_space.shape != ACTION_SHAPE: raise ValueError("KUKA-v3 requires float32 observations and 4D actions.")
    if not np.isfinite(observation).all() or not np.allclose(observation[GOAL_SLICE], env.unwrapped._target_pos): raise ValueError("KUKA-v3 reset produced an invalid goal-observable state.")
    return env
