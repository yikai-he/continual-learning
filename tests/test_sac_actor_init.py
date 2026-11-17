"""Regression coverage for strict, actor-only SAC warm starts."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import SAC

from src.config import ActorInitConfig, config_from_dict, validate_config
from src.continual.bc_policy import DETERMINISTIC_MSE, GeneralPolicy
from src.sac_actor_init import initialize_sac_actor
from src.sac_policy import LegacySACPolicy


class _Env(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(39,), dtype=np.float32)
    action_space = spaces.Box(-2.0, 3.0, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(39, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(39, dtype=np.float32), 0.0, False, True, {}


def _model(seed=0):
    return SAC(
        LegacySACPolicy,
        _Env(),
        learning_starts=1,
        buffer_size=16,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        seed=seed,
        device="cpu",
    )


def _state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)


def test_scratch_is_default_and_does_not_change_actor():
    model = _model()
    before = _state(model.actor)
    record = initialize_sac_actor(model, ActorInitConfig())
    _assert_state_equal(before, _state(model.actor))
    assert record["mode"] == "scratch"
    assert record["transferred_actor_parameters"] == []
    assert all(record["fresh_training_components"].values())


def test_actor_init_config_defaults_and_validation():
    config = config_from_dict(
        {"experiment": "sac", "runtime": {"output": "/tmp/sac-init-test"}}
    )
    assert config.actor_init == ActorInitConfig()
    configured = config_from_dict(
        {
            "experiment": "sac",
            "actor_init": {"mode": "sac_actor", "checkpoint": "expert.zip"},
            "runtime": {"output": "/tmp/sac-init-test"},
        }
    )
    validate_config(configured)
    with pytest.raises(ValueError, match="checkpoint is required"):
        validate_config(
            config_from_dict(
                {
                    "experiment": "sac",
                    "actor_init": {"mode": "general_actor"},
                    "runtime": {"output": "/tmp/sac-init-test"},
                }
            )
        )


def test_sac_actor_transfer_is_exact_and_leaves_training_state_fresh(tmp_path: Path):
    source, target = _model(1), _model(2)
    with torch.no_grad():
        for index, parameter in enumerate(source.actor.parameters()):
            parameter.fill_(0.001 * (index + 1))
    checkpoint = tmp_path / "source.zip"
    source.save(checkpoint)
    critic_before = _state(target.critic)
    critic_target_before = _state(target.critic_target)

    record = initialize_sac_actor(
        target, ActorInitConfig("sac_actor", str(checkpoint))
    )

    _assert_state_equal(_state(source.actor), _state(target.actor))
    _assert_state_equal(critic_before, _state(target.critic))
    _assert_state_equal(critic_target_before, _state(target.critic_target))
    assert all(record["fresh_training_components"].values())
    assert all(parameter.requires_grad for parameter in target.actor.parameters())
    observation = np.linspace(-1, 1, 39, dtype=np.float32)
    np.testing.assert_array_equal(
        source.predict(observation, deterministic=True)[0],
        target.predict(observation, deterministic=True)[0],
    )


@pytest.mark.parametrize("objective", [DETERMINISTIC_MSE, "squashed_gaussian_nll"])
def test_compatible_general_actor_transfer_matches_mean_policy(tmp_path: Path, objective: str):
    target = _model(3)
    source = GeneralPolicy(
        target.action_space.low,
        target.action_space.high,
        activation="leaky_relu",
        hidden_dim=256,
        objective=objective,
    ).eval()
    checkpoint = tmp_path / f"general-{objective}.pt"
    source.save(checkpoint)
    original_log_std = _state(target.actor)["log_std.weight"].clone()

    record = initialize_sac_actor(
        target, ActorInitConfig("general_actor", str(checkpoint))
    )
    observation = np.linspace(-1, 1, 39, dtype=np.float32)
    with torch.no_grad():
        expected = source(torch.as_tensor(observation).unsqueeze(0)).squeeze(0).numpy()
    actual = target.predict(observation, deterministic=True)[0]
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
    if objective == DETERMINISTIC_MSE:
        assert torch.equal(target.actor.state_dict()["log_std.weight"], original_log_std)
        assert "log_std.weight" in record["fresh_actor_parameters"]
    else:
        assert "log_std.weight" in record["transferred_actor_parameters"]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"hidden_dim": 512, "activation": "leaky_relu"}, "hidden sizes"),
        ({"hidden_dim": 256, "activation": "relu"}, "activation"),
        ({"hidden_dim": 256, "activation": "leaky_relu", "task_conditioning": "one_hot", "task_order": ["a"]}, "Task-conditioned"),
    ],
)
def test_incompatible_general_actor_is_rejected(tmp_path: Path, kwargs, error):
    source = GeneralPolicy(-2 * np.ones(4), 3 * np.ones(4), **kwargs)
    checkpoint = tmp_path / "incompatible.pt"
    source.save(checkpoint)
    with pytest.raises(ValueError, match=error):
        initialize_sac_actor(
            _model(), ActorInitConfig("general_actor", str(checkpoint))
        )


def test_missing_checkpoint_and_incompatible_standard_sac_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        initialize_sac_actor(
            _model(), ActorInitConfig("sac_actor", str(tmp_path / "missing.zip"))
        )
    standard = SAC("MlpPolicy", _Env(), policy_kwargs={"net_arch": [64, 64]}, device="cpu")
    checkpoint = tmp_path / "standard.zip"
    standard.save(checkpoint)
    with pytest.raises(ValueError, match="architecture is incompatible"):
        initialize_sac_actor(
            _model(), ActorInitConfig("sac_actor", str(checkpoint))
        )


def test_standard_four_layer_sac_maps_all_learned_actor_parameters(tmp_path: Path):
    source = SAC("MlpPolicy", _Env(), policy_kwargs={"net_arch": [256] * 4}, device="cpu")
    checkpoint = tmp_path / "standard-four-layer.zip"
    source.save(checkpoint)
    target = _model()
    record = initialize_sac_actor(
        target, ActorInitConfig("sac_actor", str(checkpoint))
    )
    assert record["source_architecture"]["policy_class"].endswith("SACPolicy")
    assert record["source_architecture"]["fresh_target_modules"] == ["layer_norm"]
    assert set(record["fresh_actor_parameters"]) == {
        "latent_pi.1.weight", "latent_pi.1.bias"
    }


def test_general_actor_action_bounds_mismatch_is_rejected(tmp_path: Path):
    source = GeneralPolicy(-np.ones(4), np.ones(4), activation="leaky_relu")
    checkpoint = tmp_path / "wrong-bounds.pt"
    source.save(checkpoint)
    with pytest.raises(ValueError, match="action bounds"):
        initialize_sac_actor(
            _model(), ActorInitConfig("general_actor", str(checkpoint))
        )


def test_warm_started_actor_can_complete_an_update(tmp_path: Path):
    source, target = _model(4), _model(5)
    checkpoint = tmp_path / "source.zip"
    source.save(checkpoint)
    initialize_sac_actor(target, ActorInitConfig("sac_actor", str(checkpoint)))
    target.learn(total_timesteps=3)
    assert target._n_updates >= 1
