"""Unified four-layer policy, transfer, and benchmark regression tests."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

from src.continual.bc_policy import GeneralPolicy
from src.sac.architecture_benchmark import load_benchmark_config
from src.sac.policy_transfer import deterministic_equivalence, transfer_general_policy_to_sac
from src.sac.transferable_actor import TransferableSACPolicy


class Env(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (39,), dtype=np.float32)
    action_space = gym.spaces.Box(-np.ones(4, np.float32), np.ones(4, np.float32))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(39, np.float32), {}

    def step(self, action):
        return np.zeros(39, np.float32), 0.0, False, True, {}


def _model():
    return SAC(
        TransferableSACPolicy,
        Env(),
        policy_kwargs={"net_arch": [256] * 4, "log_std_init": -3.0},
        buffer_size=10,
        device="cpu",
    )


def test_four_layer_actor_critics_transfer_and_equivalence():
    model = _model()
    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    critic_linears = [m for m in model.policy.critic.q_networks[0] if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in critic_linears[:-1]] == [256] * 4
    critic_before = {k: v.clone() for k, v in model.policy.critic.state_dict().items()}
    report = transfer_general_policy_to_sac(model, policy)
    assert len(report.parameter_mapping) == 6  # four Linear layers, LayerNorm, mean head
    assert all(torch.equal(v, critic_before[k]) for k, v in model.policy.critic.state_dict().items())
    observations = np.random.default_rng(1).normal(size=(16, 39)).astype(np.float32)
    assert deterministic_equivalence(policy, model, observations).failed_observations == 0


def test_four_layer_checkpoint_and_benchmark(tmp_path):
    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    checkpoint = tmp_path / "policy.pt"
    policy.save(checkpoint)
    assert GeneralPolicy.load(checkpoint, expected_hidden_sizes=(256,) * 4).hidden_sizes == (256,) * 4
    config = load_benchmark_config(Path(__file__).parents[1] / "configs/sac_4layer_benchmark.yaml")
    assert config.policy.net_arch == [256] * 4
    assert config.output == "runs/sac_4layer_benchmark"
