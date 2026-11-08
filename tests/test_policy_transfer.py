"""GeneralPolicy-to-SB3-SAC Immediate Policy transfer contract."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from src.continual.bc_policy import GeneralPolicy
from src.sac.policy_transfer import (
    PARAMETER_MAPPING,
    deterministic_equivalence,
    transfer_general_policy_to_sac,
)
from src.sac.transferable_actor import TransferableActor, TransferableSACPolicy


class ContinuousTestEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(-5, 5, shape=(39,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.np_random.uniform(-1, 1, 39).astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        observation = self.np_random.uniform(-1, 1, 39).astype(np.float32)
        reward = -float(np.square(action).mean())
        return observation, reward, False, self.steps >= 8, {}


def make_model(env, *, log_std_init=-2.25):
    return SAC(
        TransferableSACPolicy,
        env,
        learning_rate=3e-4,
        buffer_size=100,
        learning_starts=2,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256, 256, 256], "log_std_init": log_std_init},
        seed=7,
        device="cpu",
        verbose=0,
    )


def make_general_policy(env):
    torch.manual_seed(11)
    return GeneralPolicy(
        env.action_space.low,
        env.action_space.high,
    )


def test_actor_architecture_transfer_scaling_and_exact_equivalence():
    env = ContinuousTestEnv()
    model = make_model(env)
    general = make_general_policy(env)
    actor = model.policy.actor
    assert isinstance(actor, TransferableActor)
    assert [type(module) for module in actor.latent_pi] == [
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.ReLU,
    ]
    log_std_before = {
        name: value.detach().clone() for name, value in actor.log_std.named_parameters()
    }
    critic_before = {
        name: value.detach().clone() for name, value in model.policy.critic.state_dict().items()
    }
    target_before = {
        name: value.detach().clone()
        for name, value in model.policy.critic_target.state_dict().items()
    }
    report = transfer_general_policy_to_sac(model, general)
    assert report.parameter_mapping == PARAMETER_MAPPING
    for name, value in actor.log_std.named_parameters():
        torch.testing.assert_close(value, log_std_before[name], rtol=0, atol=0)
    for name, value in model.policy.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[name], rtol=0, atol=0)
    for name, value in model.policy.critic_target.state_dict().items():
        torch.testing.assert_close(value, target_before[name], rtol=0, atol=0)
    assert model.replay_buffer.size() == 0
    assert model.ent_coef_optimizer is not None
    assert model.ent_coef_optimizer.state == {}
    assert torch.count_nonzero(actor.log_std.weight) == 0
    torch.testing.assert_close(
        actor.log_std.bias,
        torch.full_like(actor.log_std.bias, -2.25),
        rtol=0,
        atol=0,
    )
    observations = np.random.default_rng(13).normal(size=(1000, 39)).astype(np.float32)
    equivalence = deterministic_equivalence(general, model, observations)
    assert equivalence.observations == 1000
    assert equivalence.failed_observations == 0
    assert equivalence.maximum_absolute_error < 1e-5
    deterministic, _ = model.predict(observations[:16], deterministic=True)
    stochastic, _ = model.predict(observations[:16], deterministic=False)
    assert np.isfinite(stochastic).all()
    assert not np.array_equal(deterministic, stochastic)
    assert np.all(stochastic >= env.action_space.low)
    assert np.all(stochastic <= env.action_space.high)


def test_transfer_keeps_fresh_optimizers_and_sac_training_updates_actor_and_critic():
    env = ContinuousTestEnv()
    model = make_model(env)
    general = make_general_policy(env)
    transfer_general_policy_to_sac(model, general)
    assert model.policy.actor.optimizer.state == {}
    assert model.policy.critic.optimizer.state == {}
    actor_ids = {id(parameter) for parameter in model.policy.actor.parameters()}
    critic_ids = {id(parameter) for parameter in model.policy.critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)
    actor_before = {
        name: value.detach().clone()
        for name, value in model.policy.actor.named_parameters()
    }
    critic_before = {
        name: value.detach().clone()
        for name, value in model.policy.critic.named_parameters()
    }
    model.learn(total_timesteps=12)
    assert np.isfinite(model.logger.name_to_value["train/actor_loss"])
    assert np.isfinite(model.logger.name_to_value["train/critic_loss"])
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.policy.actor.parameters()
    )
    assert any(
        not torch.equal(actor_before[name], value)
        for name, value in model.policy.actor.named_parameters()
    )
    assert any(
        not torch.equal(critic_before[name], value)
        for name, value in model.policy.critic.named_parameters()
    )
    assert model.policy.actor.optimizer.state
    assert model.policy.critic.optimizer.state
    assert all(
        torch.isfinite(value).all()
        for module in (model.policy.actor, model.policy.critic)
        for value in module.state_dict().values()
    )


def test_sac_save_reload_preserves_deterministic_predictions():
    env = ContinuousTestEnv()
    model = make_model(env)
    transfer_general_policy_to_sac(model, make_general_policy(env))
    observations = np.random.default_rng(17).normal(size=(32, 39)).astype(np.float32)
    expected, _ = model.predict(observations, deterministic=True)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "model.zip"
        model.save(path)
        restored = SAC.load(path, env=env, device="cpu")
        actual, _ = restored.predict(observations, deterministic=True)
    np.testing.assert_array_equal(actual, expected)


def test_incompatible_general_policy_has_clear_error():
    env = ContinuousTestEnv()
    with pytest.raises(ValueError, match="only net_arch"):
        SAC(
            TransferableSACPolicy,
            env,
            policy_kwargs={"net_arch": [256, 256], "log_std_init": -3.0},
            device="cpu",
        )
