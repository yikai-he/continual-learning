"""Strict space validation, including one authenticated KUKA-v3 legacy repair."""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np

from src.env_adapters import GOAL_SLICE
from src.support.expert_manifest import ExpertIdentity


def validate_expert_spaces(model, env: gym.Env, identity: ExpertIdentity) -> bool:
    """Validate spaces and repair only historical KUKA-v3 goal-bound metadata."""
    model_obs = model.observation_space
    env_obs = env.observation_space
    if model_obs == env_obs and model.action_space == env.action_space:
        return False

    if model.action_space != env.action_space:
        raise ValueError("Expert/environment action spaces differ.")
    if not isinstance(model_obs, gym.spaces.Box) or not isinstance(
        env_obs, gym.spaces.Box
    ):
        raise ValueError("Expert/environment observation spaces differ.")
    if model_obs.shape != env_obs.shape:
        raise ValueError("Expert/environment observation shapes differ.")
    if model_obs.dtype != env_obs.dtype:
        raise ValueError("Expert/environment observation dtypes differ.")

    # A hash-validated KUKA-v3 sidecar is required. An unmanifested checkpoint
    # admitted elsewhere with allow_legacy=True is insufficient provenance.
    env_spec = getattr(env, "spec", None)
    env_id = getattr(env_spec, "id", None)
    authenticated_kuka_v3 = (
        identity.backend == "kuka-v3"
        and identity.algorithm == "SAC"
        and identity.manifest_path is not None
        and not identity.legacy
        and identity.observation_shape == model_obs.shape
        and identity.action_shape == model.action_space.shape
        and env_id in (None, "Meta-World-KUKA/MT1")
    )
    goal = np.arange(model_obs.shape[0])[GOAL_SLICE]
    non_goal = np.ones(model_obs.shape[0], dtype=bool)
    non_goal[goal] = False
    known_legacy_signature = (
        np.array_equal(model_obs.low[non_goal], env_obs.low[non_goal])
        and np.array_equal(model_obs.high[non_goal], env_obs.high[non_goal])
        and np.array_equal(
            model_obs.low[goal], np.zeros(goal.size, dtype=model_obs.dtype)
        )
        and np.array_equal(
            model_obs.high[goal], np.zeros(goal.size, dtype=model_obs.dtype)
        )
        and (
            not np.array_equal(env_obs.low[goal], model_obs.low[goal])
            or not np.array_equal(env_obs.high[goal], model_obs.high[goal])
        )
    )
    if not authenticated_kuka_v3 or not known_legacy_signature:
        raise ValueError("Expert/environment observation spaces differ.")

    model.observation_space = deepcopy(env_obs)
    return True
