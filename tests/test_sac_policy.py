"""Legacy-inspired network architecture within Stable-Baselines3 SAC."""

from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from src.continual.policy_adapter import SB3SACPolicy
from src.sac_policy import (
    LEAKY_RELU_NEGATIVE_SLOPE,
    LegacySACPolicy,
    SAC_HIDDEN_SIZES,
)
from stable_baselines3 import SAC
from train_sac import LegacyScheduleSAC, target_entropy_from_output_std


class _SACFixtureEnv(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(39,), dtype=np.float32)
    action_space = spaces.Box(-2.0, 3.0, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(39, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(39, dtype=np.float32), 0.0, False, True, {}


def _model() -> SAC:
    return SAC(
        LegacySACPolicy,
        _SACFixtureEnv(),
        learning_starts=1,
        buffer_size=16,
        batch_size=2,
        seed=7,
        device="cpu",
    )


def _assert_legacy_core(modules, input_dim: int, *, output_dim=None) -> None:
    expected = [
        nn.Linear,
        nn.LayerNorm,
        nn.Tanh,
        nn.Linear,
        nn.LeakyReLU,
        nn.Linear,
        nn.LeakyReLU,
        nn.Linear,
        nn.LeakyReLU,
    ]
    if output_dim is not None:
        expected.append(nn.Linear)
    assert [type(module) for module in modules] == expected
    linear = [module for module in modules if isinstance(module, nn.Linear)]
    assert linear[0].in_features == input_dim
    assert [layer.out_features for layer in linear[:4]] == list(SAC_HIDDEN_SIZES)
    if output_dim is not None:
        assert linear[-1].out_features == output_dim
    for module in modules:
        if isinstance(module, nn.LeakyReLU):
            assert module.negative_slope == LEAKY_RELU_NEGATIVE_SLOPE


def test_actor_and_twin_critics_match_legacy_architecture() -> None:
    model = _model()
    actor = model.policy.actor
    _assert_legacy_core(list(actor.latent_pi), 39)
    assert actor.mu.in_features == actor.log_std.in_features == 256
    assert actor.mu.out_features == actor.log_std.out_features == 4

    assert len(model.policy.critic.q_networks) == 2
    assert len(model.policy.critic_target.q_networks) == 2
    for critic in (*model.policy.critic.q_networks, *model.policy.critic_target.q_networks):
        _assert_legacy_core(list(critic), 43, output_dim=1)


def test_sb3_actions_gradients_adapter_and_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _model()
    observation = np.linspace(-1.0, 1.0, 39, dtype=np.float32)
    deterministic, _ = model.predict(observation, deterministic=True)
    stochastic, _ = model.predict(observation, deterministic=False)
    assert deterministic.shape == stochastic.shape == (4,)
    assert np.all(deterministic >= model.action_space.low)
    assert np.all(deterministic <= model.action_space.high)
    assert np.all(stochastic >= model.action_space.low)
    assert np.all(stochastic <= model.action_space.high)
    np.testing.assert_allclose(
        SB3SACPolicy(model).act(observation), deterministic, rtol=0, atol=0
    )

    observations = torch.zeros((2, 39))
    actions, log_prob = model.policy.actor.action_log_prob(observations)
    actor_loss = (actions.square().mean() + log_prob.mean())
    actor_loss.backward()
    assert any(parameter.grad is not None for parameter in model.policy.actor.parameters())

    q_values = model.policy.critic(observations, actions.detach())
    assert len(q_values) == 2
    sum(value.mean() for value in q_values).backward()
    assert all(value.shape == (2, 1) for value in q_values)
    assert any(parameter.grad is not None for parameter in model.policy.critic.parameters())
    target_values = model.policy.critic_target(observations, actions.detach())
    assert len(target_values) == 2
    assert all(value.shape == (2, 1) for value in target_values)

    checkpoint = tmp_path / "legacy_sac"
    model.save(checkpoint)
    loaded = SAC.load(checkpoint, env=_SACFixtureEnv(), device="cpu")
    assert isinstance(loaded.policy, LegacySACPolicy)
    reloaded, _ = loaded.predict(observation, deterministic=True)
    np.testing.assert_allclose(reloaded, deterministic, rtol=0, atol=1e-7)


def test_standard_sb3_checkpoint_keeps_its_original_policy(tmp_path: Path) -> None:
    model = SAC(
        "MlpPolicy",
        _SACFixtureEnv(),
        policy_kwargs={"net_arch": [256, 256, 256, 256]},
        seed=11,
        device="cpu",
    )
    checkpoint = tmp_path / "standard_sb3_sac"
    model.save(checkpoint)

    loaded = SAC.load(checkpoint, env=_SACFixtureEnv(), device="cpu")
    assert type(loaded.policy).__name__ == "SACPolicy"
    assert not isinstance(loaded.policy, LegacySACPolicy)


def _scheduled_model() -> LegacyScheduleSAC:
    return LegacyScheduleSAC(
        LegacySACPolicy,
        _SACFixtureEnv(),
        learning_starts=1_000,
        random_exploration_steps=10_000,
        update_interval=50,
        gradient_steps=50,
        buffer_size=2_000,
        batch_size=2,
        seed=7,
        device="cpu",
    )


def test_legacy_schedule_keeps_random_actions_through_timestep_10000() -> None:
    model = _scheduled_model()
    model._last_obs = np.zeros((1, 39), dtype=np.float32)
    with patch.object(model.action_space, "sample", wraps=model.action_space.sample) as sample:
        with patch.object(model, "predict", wraps=model.predict) as predict:
            model.num_timesteps = 10_000
            model._sample_action(model.learning_starts)
            assert sample.call_count == 1
            assert predict.call_count == 0

            model.num_timesteps = 10_001
            model._sample_action(model.learning_starts)
            assert sample.call_count == 1
            assert predict.call_count == 1


def test_legacy_schedule_updates_at_zero_based_1000_then_every_50_steps() -> None:
    model = _scheduled_model()
    with patch.object(SAC, "train") as sb3_train:
        for num_timesteps in (1_000, 1_001, 1_050, 1_051):
            model.num_timesteps = num_timesteps
            model.train(gradient_steps=50, batch_size=128)
        assert sb3_train.call_count == 2
        sb3_train.assert_called_with(gradient_steps=50, batch_size=128)


def test_legacy_target_entropy_matches_source_formula() -> None:
    assert target_entropy_from_output_std(0.089, 4) == -4.000721504181298


def test_short_legacy_schedule_run_performs_a_real_gradient_update() -> None:
    model = LegacyScheduleSAC(
        LegacySACPolicy,
        _SACFixtureEnv(),
        learning_starts=1,
        random_exploration_steps=2,
        update_interval=2,
        gradient_steps=1,
        buffer_size=16,
        batch_size=2,
        seed=7,
        device="cpu",
    )
    before = [parameter.detach().clone() for parameter in model.actor.parameters()]
    model.learn(total_timesteps=3)
    assert model._n_updates == 1
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.actor.parameters(), strict=True)
    )
