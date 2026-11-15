"""Legacy-inspired neural networks for Stable-Baselines3 SAC experts."""

from __future__ import annotations

from torch import nn

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import Actor, SACPolicy


SAC_POLICY_ARCHITECTURE = "legacy-continual-diff-pytorch-4x256-v1"
SAC_HIDDEN_SIZES = (256, 256, 256, 256)
LEAKY_RELU_NEGATIVE_SLOPE = 0.01


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
