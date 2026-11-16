"""Strict actor-only warm starts for standalone SAC training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC

from src.config import ActorInitConfig
from src.continual.bc_policy import (
    DETERMINISTIC_MSE,
    SQUASHED_GAUSSIAN_NLL,
    GeneralPolicy,
)
from src.sac_policy import (
    SAC_HIDDEN_SIZES,
    SAC_POLICY_ARCHITECTURE,
    LegacySACPolicy,
)
from src.support.reproducibility import file_hash


_GENERAL_TO_SAC = {
    **{f"shared_network.{index}.{suffix}": f"latent_pi.{index}.{suffix}"
       for index in (0, 1, 3, 5, 7) for suffix in ("weight", "bias")},
    "mean_head.weight": "mu.weight",
    "mean_head.bias": "mu.bias",
    "log_std_head.weight": "log_std.weight",
    "log_std_head.bias": "log_std.bias",
}


def _checkpoint_path(checkpoint: str, base_dir: Path) -> Path:
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Actor initialization checkpoint does not exist: {path}")
    return path


def _require_target(model: SAC) -> None:
    if type(model.policy) is not LegacySACPolicy:
        raise ValueError("Actor warm-start requires the exact LegacySACPolicy target.")
    if tuple(model.observation_space.shape or ()) != (39,):
        raise ValueError("Actor warm-start requires observation shape (39,).")
    if tuple(model.action_space.shape or ()) != (4,):
        raise ValueError("Actor warm-start requires action shape (4,).")


def _require_equal_bounds(source: Any, target: Any) -> None:
    if not (
        np.array_equal(np.asarray(source.low), np.asarray(target.low))
        and np.array_equal(np.asarray(source.high), np.asarray(target.high))
    ):
        raise ValueError("Source and target action bounds are incompatible.")


def _strict_actor_state(target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> None:
    if target.keys() != source.keys():
        missing = sorted(target.keys() - source.keys())
        extra = sorted(source.keys() - target.keys())
        raise ValueError(f"SAC actor parameters are incompatible; missing={missing}, extra={extra}.")
    mismatched = [name for name in target if target[name].shape != source[name].shape]
    if mismatched:
        raise ValueError(f"SAC actor parameter shapes are incompatible: {mismatched}.")


def _fresh_components(model: SAC) -> dict[str, bool]:
    return {
        "critic": True,
        "critic_target": True,
        "actor_optimizer": not bool(model.actor.optimizer.state),
        "replay_buffer": model.replay_buffer.size() == 0,
        "entropy_coefficient": True,
        "entropy_optimizer": (
            model.ent_coef_optimizer is None
            or not bool(model.ent_coef_optimizer.state)
        ),
    }


def _require_fresh_training_state(model: SAC) -> None:
    checks = _fresh_components(model)
    if model.num_timesteps != 0:
        raise ValueError("Actor warm-start requires a new SAC model at timestep zero.")
    stale = [name for name, is_fresh in checks.items() if not is_fresh]
    if stale:
        raise ValueError(
            "Actor warm-start requires fresh target training state; stale components: "
            + ", ".join(stale)
            + "."
        )


def initialize_sac_actor(
    model: SAC,
    config: ActorInitConfig,
    *,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Initialize only ``model.actor`` and return auditable provenance."""
    _require_target(model)
    _require_fresh_training_state(model)
    target_state = model.actor.state_dict()
    transferred: list[str] = []
    source_architecture: dict[str, Any] | None = None
    source_path: Path | None = None

    if config.mode == "scratch":
        if config.checkpoint is not None:
            raise ValueError("Scratch actor initialization cannot specify a checkpoint.")
    elif config.mode == "sac_actor":
        if not config.checkpoint:
            raise ValueError("sac_actor initialization requires a checkpoint.")
        source_path = _checkpoint_path(config.checkpoint, Path(base_dir))
        try:
            source = SAC.load(
                source_path,
                device="cpu",
                custom_objects={"buffer_size": 1},
            )
        except Exception as error:
            raise ValueError(f"Could not load SAC actor checkpoint {source_path}: {error}") from error
        if type(source.policy) is not LegacySACPolicy:
            raise ValueError(
                "SAC actor checkpoint does not use the exact LegacySACPolicy class."
            )
        if source.observation_space != model.observation_space:
            raise ValueError("SAC source and target observation spaces are incompatible.")
        if tuple(source.action_space.shape or ()) != (4,):
            raise ValueError("SAC source action shape must be (4,).")
        _require_equal_bounds(source.action_space, model.action_space)
        source_state = source.actor.state_dict()
        _strict_actor_state(target_state, source_state)
        model.actor.load_state_dict(source_state, strict=True)
        transferred = list(source_state)
        source_architecture = {
            "kind": "stable_baselines3_sac",
            "policy": SAC_POLICY_ARCHITECTURE,
            "hidden_sizes": list(SAC_HIDDEN_SIZES),
        }
    elif config.mode == "general_actor":
        if not config.checkpoint:
            raise ValueError("general_actor initialization requires a checkpoint.")
        source_path = _checkpoint_path(config.checkpoint, Path(base_dir))
        try:
            source_policy = GeneralPolicy.load(source_path)
        except Exception as error:
            raise ValueError(f"Could not load GeneralPolicy checkpoint {source_path}: {error}") from error
        if source_policy.task_conditioning != "none":
            raise ValueError("Task-conditioned GeneralPolicy actors cannot initialize SAC.")
        if source_policy.hidden_sizes != SAC_HIDDEN_SIZES:
            raise ValueError(
                f"GeneralPolicy hidden sizes {source_policy.hidden_sizes} do not match {SAC_HIDDEN_SIZES}."
            )
        if source_policy.activation != "leaky_relu":
            raise ValueError("GeneralPolicy activation must be leaky_relu for exact SAC compatibility.")
        _require_equal_bounds(
            type("Bounds", (), {"low": source_policy.action_low.cpu().numpy(),
                                 "high": source_policy.action_high.cpu().numpy()})(),
            model.action_space,
        )
        general_state = source_policy.state_dict()
        mapped = dict(target_state)
        mapping = dict(_GENERAL_TO_SAC)
        if source_policy.objective == DETERMINISTIC_MSE:
            mapping.pop("log_std_head.weight")
            mapping.pop("log_std_head.bias")
        elif source_policy.objective != SQUASHED_GAUSSIAN_NLL:
            raise ValueError(f"Unsupported GeneralPolicy objective: {source_policy.objective}")
        for source_name, target_name in mapping.items():
            if source_name not in general_state or target_name not in target_state:
                raise ValueError(f"Missing compatible actor parameter: {source_name} -> {target_name}.")
            if general_state[source_name].shape != target_state[target_name].shape:
                raise ValueError(f"Actor parameter shape mismatch: {source_name} -> {target_name}.")
            mapped[target_name] = general_state[source_name]
            transferred.append(target_name)
        model.actor.load_state_dict(mapped, strict=True)
        source_architecture = {
            "kind": "general_policy",
            "objective": source_policy.objective,
            "hidden_sizes": list(source_policy.hidden_sizes),
            "activation": source_policy.activation,
            "layer_norm": source_policy.use_layer_norm,
            "task_conditioning": source_policy.task_conditioning,
        }
    else:
        raise ValueError(f"Unsupported actor initialization mode: {config.mode}")

    transferred_set = set(transferred)
    return {
        "schema_version": 1,
        "mode": config.mode,
        "source_checkpoint": str(source_path) if source_path else None,
        "source_sha256": file_hash(source_path) if source_path else None,
        "source_architecture": source_architecture,
        "target_architecture": {
            "policy": SAC_POLICY_ARCHITECTURE,
            "hidden_sizes": list(SAC_HIDDEN_SIZES),
            "observation_shape": [39],
            "action_shape": [4],
        },
        "transferred_actor_parameters": transferred,
        "fresh_actor_parameters": [name for name in target_state if name not in transferred_set],
        "fresh_training_components": _fresh_components(model),
    }
