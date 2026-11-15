"""Legacy-inspired neural networks for Stable-Baselines3 SAC experts."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    TanhBijector,
    sum_independent_dims,
)
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import Actor, SACPolicy


SAC_POLICY_ARCHITECTURE = "legacy-continual-diff-pytorch-4x256-stable-tanh-v2"
SAC_HIDDEN_SIZES = (256, 256, 256, 256)
LEAKY_RELU_NEGATIVE_SLOPE = 0.01


class LegacySquashedDiagGaussianDistribution(SquashedDiagGaussianDistribution):
    """Squashed Gaussian with the Legacy stable pre-tanh Jacobian."""

    def log_prob(
        self,
        actions: torch.Tensor,
        gaussian_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)
        gaussian_log_prob = sum_independent_dims(
            self.distribution.log_prob(gaussian_actions)
        )
        log_tanh_jacobian = 2.0 * (
            math.log(2.0)
            - gaussian_actions
            - F.softplus(-2.0 * gaussian_actions)
        )
        return gaussian_log_prob - log_tanh_jacobian.sum(dim=1)


def _legacy_network(input_dim: int, output_dim: int | None = None) -> nn.Sequential:
    """Build the legacy four-layer core and an optional linear output head."""
    layers: list[nn.Module] = [
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE),
        nn.Linear(256, 256),
        nn.LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE),
        nn.Linear(256, 256),
        nn.LeakyReLU(negative_slope=LEAKY_RELU_NEGATIVE_SLOPE),
    ]
    if output_dim is not None:
        layers.append(nn.Linear(256, output_dim))
    return nn.Sequential(*layers)


class LegacySACActor(Actor):
    """SB3 squashed-Gaussian actor with the legacy hidden network."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latent_pi = _legacy_network(self.features_dim)
        if not self.use_sde:
            self.action_dist = LegacySquashedDiagGaussianDistribution(
                int(self.action_space.shape[0])
            )


class LegacySACCritic(ContinuousCritic):
    """SB3 twin critic whose Q-networks use the legacy hidden network."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        action_dim = int(self.action_space.shape[0])
        features_dim = self.features_extractor.features_dim
        self.q_networks = []
        for index in range(self.n_critics):
            q_network = _legacy_network(
                features_dim + action_dim,
                output_dim=1,
            )
            self.add_module(f"qf{index}", q_network)
            self.q_networks.append(q_network)


class LegacySACPolicy(SACPolicy):
    """SB3 SAC policy using legacy-inspired actor and critic MLPs."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["net_arch"] = list(SAC_HIDDEN_SIZES)
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None) -> LegacySACActor:
        kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return LegacySACActor(**kwargs).to(self.device)

    def make_critic(self, features_extractor=None) -> LegacySACCritic:
        kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return LegacySACCritic(**kwargs).to(self.device)
