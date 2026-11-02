"""GeneralPolicy-to-SB3-SAC actor transfer and equivalence diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from torch import nn

from src.continual.bc_policy import GeneralPolicy
from src.envs import ACTION_SHAPE, OBSERVATION_SHAPE
from src.sac.transferable_actor import (
    GENERAL_POLICY_HIDDEN_SIZES,
    TransferableActor,
    TransferableSACPolicy,
)


@dataclass(frozen=True)
class TransferReport:
    """Names copied into the SAC actor; all other SAC state stays fresh."""

    source_checkpoint: str | None
    parameter_mapping: dict[str, str]


@dataclass(frozen=True)
class EquivalenceReport:
    """Deterministic physical-action comparison."""

    observations: int
    mean_absolute_error: float
    maximum_absolute_error: float
    failed_observations: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


PARAMETER_MAPPING = {
    "network.0": "actor.latent_pi.0",
    "network.1": "actor.latent_pi.1",
    "network.3": "actor.latent_pi.3",
    "network.5": "actor.mu",
}


def _validate_compatibility(policy: GeneralPolicy, actor: TransferableActor) -> None:
    if not policy.use_layer_norm:
        raise ValueError("Immediate Policy transfer requires a LayerNorm GeneralPolicy.")
    if policy.hidden_sizes != GENERAL_POLICY_HIDDEN_SIZES:
        raise ValueError(
            "Immediate Policy transfer requires GeneralPolicy hidden sizes (256, 256)."
        )
    if actor.observation_space.shape != OBSERVATION_SHAPE:
        raise ValueError("SAC actor observation space is not 39D.")
    if actor.action_space.shape != ACTION_SHAPE:
        raise ValueError("SAC actor action space is not 4D.")
    low_matches = np.allclose(
        policy.action_low.cpu().numpy(), actor.action_space.low
    )
    high_matches = np.allclose(
        policy.action_high.cpu().numpy(), actor.action_space.high
    )
    if not low_matches or not high_matches:
        raise ValueError("GeneralPolicy and SAC action bounds differ.")
    expected = (nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear, nn.ReLU)
    if tuple(type(module) for module in actor.latent_pi) != expected:
        raise ValueError("SAC actor mean pathway is incompatible with GeneralPolicy.")


def transfer_general_policy_to_sac(
    model: SAC,
    source: GeneralPolicy | str | Path,
) -> TransferReport:
    """Copy only the deterministic actor pathway into an existing fresh SAC.

    Log-standard-deviation, critics, target critics, replay, entropy state, and
    every optimizer state are deliberately untouched.
    """
    source_path = None
    if isinstance(source, (str, Path)):
        source_path = str(Path(source))
        policy = GeneralPolicy.load(source, expected_use_layer_norm=True)
    elif isinstance(source, GeneralPolicy):
        policy = source
    else:
        raise TypeError("source must be a GeneralPolicy or checkpoint path.")
    actor = model.policy.actor
    if not isinstance(model.policy, TransferableSACPolicy) or not isinstance(
        actor, TransferableActor
    ):
        raise ValueError("SAC model does not use TransferableSACPolicy.")
    _validate_compatibility(policy, actor)
    source_modules = (
        policy.network[0],
        policy.network[1],
        policy.network[3],
        policy.network[5],
    )
    target_modules = (
        actor.latent_pi[0],
        actor.latent_pi[1],
        actor.latent_pi[3],
        actor.mu,
    )
    with torch.no_grad():
        for source_module, target_module in zip(source_modules, target_modules):
            target_module.load_state_dict(source_module.state_dict(), strict=True)
    return TransferReport(source_path, dict(PARAMETER_MAPPING))


def deterministic_equivalence(
    policy: GeneralPolicy,
    model: SAC,
    observations: np.ndarray,
    *,
    tolerance: float = 1e-5,
) -> EquivalenceReport:
    """Compare bounded physical actions on corresponding observations."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1:] != OBSERVATION_SHAPE:
        raise ValueError("Expected an (N, 39) observation array.")
    if len(observations) == 0 or not np.isfinite(observations).all():
        raise ValueError("Observations must be nonempty and finite.")
    device = policy.action_low.device
    with torch.no_grad():
        expected = policy(torch.from_numpy(observations).to(device)).cpu().numpy()
    actual, _ = model.predict(observations, deterministic=True)
    errors = np.abs(expected - actual)
    failed = np.any(errors >= tolerance, axis=1)
    return EquivalenceReport(
        observations=len(observations),
        mean_absolute_error=float(errors.mean()),
        maximum_absolute_error=float(errors.max()),
        failed_observations=int(failed.sum()),
    )
