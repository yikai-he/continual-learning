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
from src.sac.transferable_actor import TransferableActor, TransferableSACPolicy


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


def parameter_mapping(policy: GeneralPolicy, actor: TransferableActor) -> dict[str, str]:
    """Derive the complete transferable pathway from module structure."""
    source = [(name, module) for name, module in policy.network.named_children()
              if isinstance(module, (nn.Linear, nn.LayerNorm))]
    target = [(name, module) for name, module in actor.latent_pi.named_children()
              if isinstance(module, (nn.Linear, nn.LayerNorm))]
    target.append(("mu", actor.mu))
    if len(source) != len(target) or any(type(a[1]) is not type(b[1]) for a, b in zip(source, target)):
        raise ValueError("GeneralPolicy and SAC actor parameter pathways are incompatible.")
    return {f"network.{a[0]}": f"actor.{('latent_pi.' + b[0]) if b[0] != 'mu' else 'mu'}"
            for a, b in zip(source, target)}


PARAMETER_MAPPING = {
    "network.0": "actor.latent_pi.0",
    "network.1": "actor.latent_pi.1",
    "network.3": "actor.latent_pi.3",
    "network.5": "actor.latent_pi.5",
    "network.7": "actor.latent_pi.7",
    "network.9": "actor.mu",
}


def _validate_compatibility(policy: GeneralPolicy, actor: TransferableActor) -> None:
    if not policy.use_layer_norm:
        raise ValueError("Immediate Policy transfer requires a LayerNorm GeneralPolicy.")
    if policy.hidden_sizes != tuple(actor.net_arch):
        raise ValueError(
            "Immediate Policy transfer requires matching GeneralPolicy and SAC "
            f"hidden sizes; checkpoint={policy.hidden_sizes}, actor={tuple(actor.net_arch)}."
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
    parameter_mapping(policy, actor)


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
    mapping = parameter_mapping(policy, actor)
    source_modules = dict(policy.network.named_modules())
    target_modules = dict(actor.named_modules())
    with torch.no_grad():
        for source_name, target_name in mapping.items():
            target_modules[target_name.removeprefix("actor.")].load_state_dict(
                source_modules[source_name.removeprefix("network.")].state_dict(), strict=True
            )
    return TransferReport(source_path, mapping)


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
