import torch

from .diffusion_data import PHYSICAL_ACTION_SLICE, PHYSICAL_OBSERVATION_SLICE


def generated_action_diagnostics(before, after, projection):
    """Describe decoded replay before and after optional action projection."""
    if before.shape != after.shape or not torch.equal(
        before[..., PHYSICAL_OBSERVATION_SLICE],
        after[..., PHYSICAL_OBSERVATION_SLICE],
    ):
        raise ValueError("Projection changed replay shape or observations.")
    prior = before[..., PHYSICAL_ACTION_SLICE]
    result = after[..., PHYSICAL_ACTION_SLICE]
    pre_oob = prior.abs() > 1
    post_oob = result.abs() > 1
    return {
        "generated_action_projection": projection,
        "pre_projection_oob_component_fraction": float(pre_oob.float().mean()),
        "pre_projection_oob_timestep_fraction": float(
            pre_oob.any(dim=-1).float().mean()
        ),
        "post_projection_oob_component_fraction": float(post_oob.float().mean()),
        "clipped_component_fraction": float((prior != result).float().mean()),
        "post_projection_abs_eq_1_fraction": float((result.abs() == 1).float().mean()),
        "post_projection_abs_gt_0_99_fraction": float(
            (result.abs() > 0.99).float().mean()
        ),
        "post_projection_abs_eq_1_fraction_by_action_coordinate": (result.abs() == 1)
        .float()
        .mean(dim=(0, 1))
        .tolist(),
        "post_projection_abs_gt_0_99_fraction_by_action_coordinate": (
            result.abs() > 0.99
        )
        .float()
        .mean(dim=(0, 1))
        .tolist(),
    }
