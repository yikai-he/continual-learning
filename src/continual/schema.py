"""Trajectory contract for collected environment episodes.

Current environment contract (owned by src.envs): observations (T, 39),
actions (T, 4), rewards (T,), with T bounded by the effective environment
horizon. Each observation is the state
before its corresponding action; each reward follows that action.
This schema is unpadded: T equals length. Any padded representation requires an
explicit valid-length mask so padding cannot enter learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.envs import ACTION_SHAPE, OBSERVATION_SHAPE


@dataclass
class Trajectory:
    """An unpadded episode aligned as pre-action observations and actions.

    ``task_id`` is a task name, not a one-hot vector. ``episode_return`` is the
    sum of valid rewards, ``success`` means success at any step, and ``length``
    is the common leading dimension of all three arrays.
    """

    observations: NDArray[np.float32]
    actions: NDArray[np.floating]
    rewards: NDArray[np.floating]
    task_id: str
    episode_return: float
    success: bool
    final_success: bool
    length: int


def validate_trajectory(trajectory: Trajectory, *, horizon: int) -> None:
    """Validate shapes, finite values, metadata, and return consistency.

    ``horizon`` should come from ``env.spec.max_episode_steps``. This function
    accepts shorter terminated episodes but does not accept or infer padding.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if not isinstance(trajectory.task_id, str) or not trajectory.task_id.strip():
        raise ValueError("task_id must be a non-empty scalar task-name string.")
    if isinstance(trajectory.length, bool) or not isinstance(trajectory.length, int):
        raise ValueError("length must be an integer.")
    t = trajectory.length
    if not 1 <= t <= horizon:
        raise ValueError(f"length must be between 1 and horizon ({horizon}).")
    for name, shape in (
        ("observations", (t, *OBSERVATION_SHAPE)),
        ("actions", (t, *ACTION_SHAPE)),
        ("rewards", (t,)),
    ):
        array = getattr(trajectory, name)
        if not isinstance(array, np.ndarray) or array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite real numbers.")
    value = trajectory.episode_return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError("episode_return must be a finite real scalar.")
    if not np.isclose(
        value, trajectory.rewards.sum(dtype=np.float64), rtol=1e-7, atol=1e-8
    ):
        raise ValueError("episode_return must equal the sum of rewards.")
    if not isinstance(trajectory.success, (bool, np.bool_)):
        raise ValueError("success must be boolean.")
    if not isinstance(trajectory.final_success, (bool, np.bool_)):
        raise ValueError("final_success must be boolean.")
