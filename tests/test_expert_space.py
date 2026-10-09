from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from src.continual.expert_space import validate_expert_spaces
from src.support.expert_manifest import ExpertIdentity


class Model:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class Env:
    def __init__(self, observation_space, action_space, spec=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.spec = spec


def box(low, high, dtype=np.float32):
    return gym.spaces.Box(
        np.asarray(low, dtype=dtype), np.asarray(high, dtype=dtype), dtype=dtype
    )


def identity(**updates):
    value = ExpertIdentity(
        task_name="kuka-reach-v3",
        checkpoint_path="/checkpoints/expert.zip",
        checkpoint_sha256="a" * 64,
        backend="kuka-v3",
        algorithm="SAC",
        reward_function_version="v2",
        training_seed=0,
        training_config_reference="configs/sac/kuka_reach_v3_1m.yaml",
        training_config_sha256="b" * 64,
        observation_shape=(39,),
        action_shape=(4,),
        horizon=200,
        hammer_reward_variant=None,
        hammer_nail_progress_weight=None,
        qualification={"protocol": "fixed-mt1-task-bank-v1"},
        manifest_path="/checkpoints/expert_manifest.json",
        legacy=False,
    )
    return replace(value, **updates)


def spaces():
    low = np.full(39, -np.inf, dtype=np.float32)
    high = np.full(39, np.inf, dtype=np.float32)
    low[36:39] = [-0.2, 0.4, 0.1]
    high[36:39] = [0.2, 0.8, 0.4]
    legacy_low, legacy_high = low.copy(), high.copy()
    legacy_low[36:39] = 0
    legacy_high[36:39] = 0
    action = box(np.full(4, -1), np.full(4, 1))
    return box(legacy_low, legacy_high), box(low, high), action


def test_authenticated_legacy_kuka_goal_bounds_are_repaired_in_memory():
    legacy, current, action = spaces()
    model = Model(legacy, action)
    env = Env(current, action)

    assert validate_expert_spaces(model, env, identity()) is True
    assert model.observation_space == current
    assert model.observation_space is not env.observation_space
    assert legacy.low[36:39].tolist() == [0.0, 0.0, 0.0]


def test_equal_spaces_remain_strict_noop():
    _, current, action = spaces()
    model = Model(current, action)
    assert validate_expert_spaces(model, Env(current, action), identity()) is False
    assert model.observation_space is current


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("other_bound", "observation spaces"),
        ("goal_not_zero", "observation spaces"),
        ("dtype", "dtypes"),
        ("shape", "shapes"),
        ("action", "action spaces"),
        ("backend", "observation spaces"),
        ("environment", "observation spaces"),
        ("unmanifested", "observation spaces"),
    ],
)
def test_non_signature_mismatches_are_rejected(mutation, match):
    legacy, current, action = spaces()
    expert_identity = identity()
    env_spec = None
    if mutation == "other_bound":
        legacy.low[10] = -7
    elif mutation == "goal_not_zero":
        legacy.low[36] = -0.1
    elif mutation == "dtype":
        legacy = box(legacy.low, legacy.high, np.float64)
    elif mutation == "shape":
        legacy = box(np.zeros(38), np.zeros(38))
    elif mutation == "action":
        action = box(np.full(4, -2), np.full(4, 1))
    elif mutation == "backend":
        expert_identity = identity(backend="metaworld-v3")
    elif mutation == "environment":
        env_spec = type("Spec", (), {"id": "Other/Environment"})()
    elif mutation == "unmanifested":
        expert_identity = identity(manifest_path=None, legacy=True)

    model = Model(legacy, action)
    _, _, env_action = spaces()
    with pytest.raises(ValueError, match=match):
        validate_expert_spaces(
            model, Env(current, env_action, spec=env_spec), expert_identity
        )
