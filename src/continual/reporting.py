"""Human-readable console reports for recorded continual evaluations."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

from src.config import REPLAY_PRIORITY_STRATEGY
from src.continual.bc_policy import BC_CLAMP_EPSILON
from continual_manipulation.evaluation import (
    format_final_report as _format_final_report,
    format_stage_report as _format_stage_report,
)

if TYPE_CHECKING:
    from continual_manipulation.evaluation import EvaluationMatrix


def build_diffcrl_stage_report(
    config,
    context,
    data,
    diffusion_training,
    policy_training,
    artifacts,
    *,
    policy_after_state_sha256: str,
    diffusion_sha256: str | None,
    policy_sha256: str,
    flow_matching_sha256: str | None = None,
    best_mse_policy_sha256: str | None = None,
) -> dict[str, object]:
    """Construct one DiffCRL stage report without side effects."""
    normalizer = diffusion_training.normalizer
    goal_normalizer = diffusion_training.goal_normalizer
    configured_weights = config.continual.replay_task_weights
    allocation_mode = config.continual.replay_allocation
    historical_weights = configured_weights[: context.index]
    historical_weight_sum = sum(historical_weights)
    replay_allocation = data.replay_allocation or {}
    report = {
        "stage": context.index,
        "task": context.task.task_name,
        **(
            {
                "requested_current_trajectory_count": context.trajectory_count,
                "actual_current_trajectory_count": data.splits[
                    context.task.task_name
                ]["trajectories"],
                "current_trajectory_schedule": (
                    config.continual.current_trajectory_schedule
                ),
                "runtime_seed": config.runtime.seed,
            }
            if config.continual.current_trajectory_schedule
            else {}
        ),
        "bc_normalization_mode": "layer-norm",
        "bc_use_layer_norm": True,
        "bc_loss": config.bc.loss,
        "bc_initial_log_std": config.bc.initial_log_std,
        "bc_activation": config.bc.activation,
        "bc_clamp_epsilon": BC_CLAMP_EPSILON,
        "diffusion_action_space": config.diffusion.action_space,
        "diffusion_action_clamp_epsilon": config.diffusion.action_clamp_epsilon,
        "generated_action_projection": config.diffusion.generated_action_projection,
        "sources": data.sources,
        "splits": data.splits,
        "diffusion_before_state_sha256": context.diffusion_before_hash,
        "flow_matching_before_state_sha256": context.flow_matching_before_hash,
        "policy_before_state_sha256": context.policy_before_hash,
        "policy_after_state_sha256": policy_after_state_sha256,
        "diffusion_checkpoint": str(artifacts.diffusion_path)
        if artifacts.diffusion_path is not None
        else None,
        "diffusion_sha256": diffusion_sha256,
        "flow_matching_checkpoint": str(artifacts.flow_matching_path)
        if artifacts.flow_matching_path is not None
        else None,
        "flow_matching_sha256": flow_matching_sha256,
        **(
            {
                "replay_budget": config.continual.replay_budget,
                "replay_priority_strategy": REPLAY_PRIORITY_STRATEGY,
                "replay_allocation_mode": config.continual.replay_allocation,
                "replay_allocation": {
                    config.continual.tasks[task_id]: count
                    for task_id, count in replay_allocation.items()
                },
                **(
                    {
                        "replay_task_weights": configured_weights,
                        "historical_tasks_considered": config.continual.tasks[
                            : context.index
                        ],
                        "normalized_historical_weights": [
                            weight / historical_weight_sum
                            for weight in historical_weights
                        ],
                        "actual_generated_trajectory_counts": {
                            name: int(source.get("trajectory_count", 0))
                            for name, source in data.sources.items()
                            if source.get("kind") == "generated"
                        },
                    }
                    if allocation_mode == "fixed-weighted"
                    else {}
                ),
            }
            if config.continual.replay_mode
            in ("flow-matching", "expert-recollection")
            or config.continual.replay_mode == "diffusion"
            and config.continual.replay_allocation
            in ("fixed-total", "fixed-weighted")
            else {}
        ),
        "generator_mode": config.continual.replay_mode,
        "generator_checkpoint": str(
            artifacts.flow_matching_path or artifacts.diffusion_path
        )
        if artifacts.flow_matching_path is not None
        or artifacts.diffusion_path is not None
        else None,
        "policy_checkpoint": str(artifacts.policy_path),
        "policy_sha256": policy_sha256,
        "normalization_mean": normalizer.mean.tolist(),
        "normalization_scale": normalizer.scale.tolist(),
        "goal_normalization_mean": goal_normalizer.mean.tolist()
        if goal_normalizer is not None
        else None,
        "goal_normalization_scale": goal_normalizer.scale.tolist()
        if goal_normalizer is not None
        else None,
        "diffusion_losses": diffusion_training.losses
        if config.continual.replay_mode == "diffusion"
        else None,
        "flow_matching_losses": diffusion_training.losses
        if config.continual.replay_mode == "flow-matching"
        else None,
        "bc_losses": policy_training,
        "generator_training_seconds": diffusion_training.training_seconds,
    }
    checkpoint_selection = config.bc.checkpoint_selection
    if checkpoint_selection != "validation_nll":
        report["bc_checkpoint_selection"] = checkpoint_selection
    if artifacts.best_mse_policy_path is not None:
        report["dual_checkpoint_selection"] = {
            "authoritative": {
                "selection_metric": "validation_nll",
                "selected_epoch": policy_training["best_epoch"],
                "validation_nll": policy_training["validation_loss"],
                "validation_action_mse": policy_training[
                    "validation_action_mse"
                ],
                "policy_checkpoint": str(artifacts.policy_path),
                "policy_sha256": policy_sha256,
            },
            "alternative": {
                "selection_metric": "validation_action_mse",
                "selected_epoch": policy_training["best_mse_checkpoint"]["epoch"],
                "validation_nll": policy_training["best_mse_checkpoint"][
                    "validation_loss"
                ],
                "validation_action_mse": policy_training[
                    "best_mse_checkpoint"
                ]["validation_action_mse"],
                "policy_checkpoint": str(artifacts.best_mse_policy_path),
                "policy_sha256": best_mse_policy_sha256,
                "evaluation_matrix": str(
                    context.directory / "evaluation_matrix_best_mse.json"
                ),
            },
            "same_training_trajectory": True,
        }
    return report


def print_stage_report(
    matrix: EvaluationMatrix,
    stage: int,
    *,
    file: TextIO | None = None,
    **kwargs,
):
    print(
        _format_stage_report(matrix, stage, **kwargs),
        file=file or sys.stdout,
        flush=True,
    )


def print_final_report(matrix: EvaluationMatrix, *, file: TextIO | None = None):
    print(_format_final_report(matrix), file=file or sys.stdout, flush=True)
