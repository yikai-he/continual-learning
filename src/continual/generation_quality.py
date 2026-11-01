"""Offline trajectory-quality diagnostics shared by generation experiments."""

from __future__ import annotations

import numpy as np
import torch
from src.envs import CURRENT_STATE_SLICE

from .diffusion_data import unpack_trajectories


def expert_disagreement_by_timestep(expert, physical: torch.Tensor) -> dict:
    """Compare generated actions with deterministic expert actions without filtering."""
    observations, actions = unpack_trajectories(physical)
    flat = observations.flatten(0, 1).numpy()
    predicted = []
    for start in range(0, len(flat), 4096):
        values, _ = expert.predict(flat[start : start + 4096], deterministic=True)
        predicted.append(np.asarray(values, dtype=np.float32))
    predicted = np.concatenate(predicted).reshape(actions.shape)
    delta = actions.numpy() - predicted
    per_timestep = []
    for timestep in range(actions.shape[1]):
        item = delta[:, timestep]
        generated = actions[:, timestep].numpy()
        per_timestep.append(
            {
                "timestep": timestep,
                "mse": float(np.mean(item**2)),
                "mae": float(np.mean(np.abs(item))),
                "mean_l2": float(np.mean(np.linalg.norm(item, axis=1))),
                "action_oob_component_fraction": float(np.mean(np.abs(generated) > 1)),
                "action_oob_trajectory_fraction": float(
                    np.mean(np.any(np.abs(generated) > 1, axis=1))
                ),
            }
        )
    return {
        "overall": {
            "mse": float(np.mean(delta**2)),
            "mae": float(np.mean(np.abs(delta))),
            "mean_l2": float(np.mean(np.linalg.norm(delta, axis=2))),
            "sample_count": int(delta.shape[0] * delta.shape[1]),
        },
        "per_timestep": per_timestep,
        "used_for_filtering": False,
    }


def state_delta_discrepancy_by_timestep(
    real: torch.Tensor, generated: torch.Tensor
) -> dict:
    """Localize state-transition norm differences against the real reference."""
    real_observations, _ = unpack_trajectories(real)
    generated_observations, _ = unpack_trajectories(generated)
    real_state = real_observations[..., CURRENT_STATE_SLICE].numpy()
    generated_state = generated_observations[..., CURRENT_STATE_SLICE].numpy()
    real_delta = np.linalg.norm(np.diff(real_state, axis=1), axis=2)
    generated_delta = np.linalg.norm(np.diff(generated_state, axis=1), axis=2)
    result = []
    for offset in range(real_delta.shape[1]):
        rv, gv = real_delta[:, offset], generated_delta[:, offset]
        result.append(
            {
                "timestep": offset + 1,
                "real_mean": float(np.mean(rv)),
                "generated_mean": float(np.mean(gv)),
                "mean_difference": float(np.mean(gv) - np.mean(rv)),
                "real_p95": float(np.percentile(rv, 95)),
                "generated_p95": float(np.percentile(gv, 95)),
                "fraction_above_real_p95": float(np.mean(gv > np.percentile(rv, 95))),
            }
        )
    return {"per_timestep": result}
