"""SB3 SAC actor with the GeneralPolicy LayerNorm mean-action pathway."""

from __future__ import annotations

import numpy as np
from stable_baselines3.sac.policies import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    Actor,
    SACPolicy,
)
from torch import nn

GENERAL_POLICY_HIDDEN_SIZES = (256, 256, 256, 256)


class TransferableActor(Actor):
    """Standard squashed-Gaussian SAC actor with a LayerNorm policy trunk.

    The deterministic normalized action is ``tanh(mu(latent_pi(obs)))``.
    SB3 performs the subsequent affine mapping to the environment action bounds.
    The log-standard-deviation head is deliberately independent of transferred
    GeneralPolicy parameters.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_sde:
            raise ValueError("TransferableActor does not support gSDE.")
        if tuple(self.net_arch) != GENERAL_POLICY_HIDDEN_SIZES:
            raise ValueError(
                "TransferableActor supports only net_arch=[256, 256, 256, 256]."
            )
        if self.features_dim != 39:
            raise ValueError("TransferableActor requires 39D observations.")
        if not np.isfinite(self.log_std_init) or not (
            LOG_STD_MIN <= self.log_std_init <= LOG_STD_MAX
        ):
            raise ValueError(
                f"log_std_init must be finite and in [{LOG_STD_MIN}, {LOG_STD_MAX}]."
            )
        layers: list[nn.Module] = [nn.Linear(39, self.net_arch[0])]
        layers.extend((nn.LayerNorm(self.net_arch[0]), nn.ReLU()))
        for input_size, output_size in zip(self.net_arch, self.net_arch[1:]):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        self.latent_pi = nn.Sequential(*layers)
        if not isinstance(self.mu, nn.Linear) or not isinstance(
            self.log_std, nn.Linear
        ):
            raise ValueError("Expected ordinary SB3 SAC linear action heads.")
        # SB3's ordinary (non-gSDE) Actor does not otherwise use log_std_init.
        # Zero weights plus a constant bias give an observation-independent,
        # documented initial exploration scale without touching the mean head.
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, self.log_std_init)


class TransferableSACPolicy(SACPolicy):
    """SACPolicy that changes only actor construction; critics remain SB3 SAC."""

    def make_actor(self, features_extractor=None) -> TransferableActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return TransferableActor(**actor_kwargs).to(self.device)
