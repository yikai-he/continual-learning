"""Public environment API and backend dispatcher."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

from src.env_adapters import (
    ACTION_SHAPE,
    CURRENT_STATE_DIM,
    CURRENT_STATE_SLICE,
    GOAL_DIM,
    GOAL_SLICE,
    OBSERVATION_SHAPE,
    PREVIOUS_STATE_SLICE,
    EpisodeSuccessWrapper,
)
from src.env_adapters.kuka_v2 import KUKA_HORIZON, make_kuka_v2_env
from src.env_adapters.kuka_v3 import KUKA_V3_HORIZON, make_kuka_v3_env
from src.env_adapters.metaworld_v3 import (
    METAWORLD_V3_HORIZON,
    MetaWorldObservationWrapper,
    make_metaworld_v3_env,
)

EPISODE_HORIZON = METAWORLD_V3_HORIZON
ENV_BACKEND_HORIZONS = {
    "metaworld-v3": METAWORLD_V3_HORIZON,
    "kuka-v2": KUKA_HORIZON,
    "kuka-v3": KUKA_V3_HORIZON,
}


@dataclass(frozen=True)
class EnvironmentSettings:
    """Immutable environment semantics shared by every construction path."""

    backend: str = "metaworld-v3"
    reward_function_version: str = "v2"
    hammer_reward_variant: str = "original"
    hammer_nail_progress_weight: float = 0.0
    peg_unplug_reward_variant: str = "original"
    peg_unplug_progress_weight: float = 0.0
    peg_unplug_progress_gamma: float = 0.99
    cube_push_curriculum_level: int = 3
    cube_push_initialization: str = "default"
    cube_y_min: float = 0.490
    near_contact_gap_m: float = 0.005

    @classmethod
    def from_config(cls, config) -> "EnvironmentSettings":
        return cls(
            backend=config.backend,
            reward_function_version=config.reward_function_version,
            hammer_reward_variant=getattr(config, "hammer_reward_variant", "original"),
            hammer_nail_progress_weight=getattr(
                config, "hammer_nail_progress_weight", 0.0
            ),
            peg_unplug_reward_variant=getattr(
                config, "peg_unplug_reward_variant", "original"
            ),
            peg_unplug_progress_weight=getattr(
                config, "peg_unplug_progress_weight", 0.0
            ),
            peg_unplug_progress_gamma=getattr(
                config, "peg_unplug_progress_gamma", 0.99
            ),
            cube_push_curriculum_level=getattr(
                config, "cube_push_curriculum_level", 3
            ),
            cube_push_initialization=getattr(config, "cube_push_initialization", "default"),
            cube_y_min=getattr(config, "cube_y_min", 0.490),
            near_contact_gap_m=getattr(config, "near_contact_gap_m", 0.005),
        )

    def kwargs(self) -> dict:
        return {
            "reward_function_version": self.reward_function_version,
            "hammer_reward_variant": self.hammer_reward_variant,
            "hammer_nail_progress_weight": self.hammer_nail_progress_weight,
            "peg_unplug_reward_variant": self.peg_unplug_reward_variant,
            "peg_unplug_progress_weight": self.peg_unplug_progress_weight,
            "peg_unplug_progress_gamma": self.peg_unplug_progress_gamma,
            "cube_push_curriculum_level": self.cube_push_curriculum_level,
            "cube_push_initialization": self.cube_push_initialization,
            "cube_y_min": self.cube_y_min,
            "near_contact_gap_m": self.near_contact_gap_m,
        }

    def make(self, task: str, seed: int, render_mode: str | None = None) -> gym.Env:
        return make_env(self.backend, task, seed, render_mode=render_mode, **self.kwargs())


def make_metaworld_env(
    task: str,
    seed: int,
    render_mode: str | None = None,
    *,
    reward_function_version: str = "v2",
) -> gym.Env:
    """Backward-compatible public alias for the MetaWorld-v3 adapter."""
    return make_metaworld_v3_env(
        task,
        seed,
        render_mode=render_mode,
        reward_function_version=reward_function_version,
    )


def make_env(
    backend: str,
    task: str,
    seed: int,
    render_mode: str | None = None,
    *,
    reward_function_version: str = "v2",
    hammer_reward_variant: str = "original",
    hammer_nail_progress_weight: float = 0.0,
    peg_unplug_reward_variant: str = "original",
    peg_unplug_progress_weight: float = 0.0,
    peg_unplug_progress_gamma: float = 0.99,
    cube_push_curriculum_level: int = 3,
    cube_push_initialization: str = "default",
    cube_y_min: float = 0.490,
    near_contact_gap_m: float = 0.005,
) -> gym.Env:
    """Create an environment through the selected backend adapter."""
    if backend == "metaworld-v3":
        return make_metaworld_v3_env(
            task,
            seed,
            render_mode=render_mode,
            reward_function_version=reward_function_version,
        )
    if backend == "kuka-v2":
        return make_kuka_v2_env(task, seed, render_mode=render_mode)
    if backend == "kuka-v3":
        return make_kuka_v3_env(
            task,
            seed,
            render_mode=render_mode,
            reward_function_version=reward_function_version,
            hammer_reward_variant=hammer_reward_variant,
            hammer_nail_progress_weight=hammer_nail_progress_weight,
            peg_unplug_reward_variant=peg_unplug_reward_variant,
            peg_unplug_progress_weight=peg_unplug_progress_weight,
            peg_unplug_progress_gamma=peg_unplug_progress_gamma,
            cube_push_curriculum_level=cube_push_curriculum_level,
            cube_push_initialization=cube_push_initialization,
            cube_y_min=cube_y_min,
            near_contact_gap_m=near_contact_gap_m,
        )
    raise ValueError(f"Unsupported environment backend: {backend!r}.")


def horizon_for_backend(backend: str) -> int:
    """Return the task horizon associated with an environment backend."""
    try:
        return ENV_BACKEND_HORIZONS[backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported environment backend: {backend!r}.") from exc


def base_task_env(env: gym.Env, backend: str):
    """Return the backend's actual task environment, hiding adapter topology."""
    if backend == "kuka-v2":
        base = getattr(env.unwrapped, "raw_env", None)
    elif backend in ("metaworld-v3", "kuka-v3"):
        base = env.unwrapped
    else:
        raise ValueError(f"Unsupported environment backend: {backend!r}.")
    if base is None:
        raise ValueError(f"Cannot locate base task environment for {backend!r}.")
    return base


def target_position(env: gym.Env, backend: str):
    """Copy the active goal from the backend's base task environment."""
    import numpy as np

    base = base_task_env(env, backend)
    if not hasattr(base, "_target_pos"):
        raise ValueError(f"Base task environment for {backend!r} has no target position.")
    target = np.asarray(base._target_pos, dtype=np.float32)
    if target.shape != (GOAL_DIM,) or not np.isfinite(target).all():
        raise ValueError("Environment target position must be a finite 3-vector.")
    return target.copy()


def task_configuration(env: gym.Env, backend: str):
    """Copy the active randomized task vector when that backend exposes one."""
    import numpy as np

    base = base_task_env(env, backend)
    value = getattr(base, "_last_rand_vec", None)
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("Environment task configuration must be a finite vector.")
    return result.copy()
