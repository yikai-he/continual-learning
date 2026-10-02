"""Public environment API and backend dispatcher."""

from __future__ import annotations

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
from src.env_adapters.metaworld_v3 import (
    METAWORLD_V3_HORIZON,
    MetaWorldObservationWrapper,
    make_metaworld_v3_env,
)

EPISODE_HORIZON = METAWORLD_V3_HORIZON
ENV_BACKEND_HORIZONS = {
    "metaworld-v3": METAWORLD_V3_HORIZON,
    "kuka-v2": KUKA_HORIZON,
}


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
    raise ValueError(f"Unsupported environment backend: {backend!r}.")


def horizon_for_backend(backend: str) -> int:
    """Return the task horizon associated with an environment backend."""
    try:
        return ENV_BACKEND_HORIZONS[backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported environment backend: {backend!r}.") from exc
