"""Human-readable console reports for recorded continual evaluations."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

from src.continual.bc_policy import BC_CLAMP_EPSILON
from src.config import REPLAY_PRIORITY_STRATEGY
from src.continual.metrics import average_performance, forgetting
from src.continual.replay_allocation import replay_counts

if TYPE_CHECKING:
    from src.continual.evaluation import EvaluationMatrix


MISSING = "—"


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
) -> dict[str, object]:
    """Construct one DiffCRL stage report without side effects."""
    normalizer = diffusion_training.normalizer
    goal_normalizer = diffusion_training.goal_normalizer
    return {
        "stage": context.index,
        "task": context.task.task_name,
        "bc_normalization_mode": "layer-norm",
        "bc_use_layer_norm": True,
        "bc_loss": config.bc.loss,
        "bc_clamp_epsilon": BC_CLAMP_EPSILON,
        "diffusion_action_space": config.diffusion.action_space,
        "diffusion_action_clamp_epsilon": config.diffusion.action_clamp_epsilon,
        "generated_action_projection": config.diffusion.generated_action_projection,
        "real_diffusion_encode_clip_fraction": data.real_encode_clip_fraction,
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
                "replay_allocation_mode": getattr(
                    config.continual, "replay_allocation", "generator-default"
                ),
                "replay_allocation": {
                    config.continual.tasks[task_id]: count
                    for task_id, count in replay_counts(
                        range(context.index),
                        generator=(
                            "diffusion"
                            if config.continual.replay_mode == "real"
                            else config.continual.replay_mode
                        ),
                        allocation=getattr(
                            config.continual,
                            "replay_allocation",
                            "generator-default",
                        ),
                        trajectories_per_task=getattr(
                            config.continual,
                            "trajectories_per_task",
                            context.trajectory_count,
                        ),
                        total_budget=(
                            config.continual.replay_budget if context.index else 0
                        ),
                        strategy=REPLAY_PRIORITY_STRATEGY,
                    ).items()
                },
            }
            if config.continual.replay_mode in ("flow-matching", "real")
            or config.continual.replay_mode == "diffusion"
            and getattr(config.continual, "replay_allocation", "generator-default")
            == "fixed-total"
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
        "bc_losses": policy_training.losses,
        "generator_training_seconds": diffusion_training.training_seconds,
        "nan_count": 0,
        "inf_count": 0,
        "generated_success": None,
        "action_targets_modified": any(
            source.get("clipped_component_fraction", 0) > 0
            for source in data.sources.values()
        ),
    }


def _percentage(value):
    return MISSING if value is None else f"{100 * value:.1f}%"


def _matrix_percentage(value):
    return MISSING if value is None else f"{100 * value:.1f}"


def _percentage_points(value):
    return MISSING if value is None else f"{100 * value:.1f} percentage points"


def _return(value):
    return MISSING if value is None else f"{value:.3f}"


def format_evaluation_result(task_name: str, result: dict) -> str:
    """Format one completed evaluation result without recalculating it."""
    return (
        f"Result: {task_name} | "
        f"Success Rate: {_percentage(result['success_rate'])} | "
        f"Mean Return: {_return(result['mean_return'])}"
    )


def _table(headers, rows):
    widths = [
        max(len(str(value)) for value in (header, *(row[index] for row in rows)))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(headers)
        )
    ]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(
                str(value).ljust(widths[index])
                if index == 0
                else str(value).rjust(widths[index])
                for index, value in enumerate(row)
            )
        )
    return lines


def format_stage_report(
    matrix: EvaluationMatrix,
    stage: int,
    *,
    task_training_steps: int | None,
    global_timesteps: int | None,
    training_detail: str | None = None,
) -> str:
    """Format one already-recorded evaluation stage without changing it."""
    task = matrix.sequence.task(stage)
    rows = []
    for index, measured_task in enumerate(matrix.sequence.tasks):
        cell = matrix.cells[stage][index]
        if cell is not None:
            rows.append(
                (
                    measured_task.task_name,
                    _percentage(cell["success_rate"]),
                    _return(cell.get("mean_return")),
                )
            )

    current = matrix.cells[stage][stage]
    current_success = None if current is None else current["success_rate"]
    current_ap = average_performance(matrix.R, stage)
    current_forgetting = forgetting(matrix.R, stage)
    lines = [
        "",
        f"Continual evaluation — Stage {stage + 1}/{matrix.sequence.num_tasks}",
        f"Task: {task.task_name}",
        "Task training steps: "
        f"{task_training_steps if task_training_steps is not None else MISSING}",
        f"Global timesteps: {global_timesteps if global_timesteps is not None else MISSING}",
    ]
    if training_detail:
        lines.append(training_detail)
    lines += [
        "",
        "Measured tasks",
        *_table(("Task", "Success Rate", "Mean Return"), rows),
        "",
        f"Current-task Success Rate: {_percentage(current_success)}",
        f"Current Average Performance: {_percentage(current_ap)}",
        f"Forgetting: {_percentage_points(current_forgetting)}",
    ]
    return "\n".join(lines)


def format_final_report(matrix: EvaluationMatrix) -> str:
    """Format the recorded success matrix and final endpoint metrics."""
    names = [task.task_name for task in matrix.sequence.tasks]
    rows = [
        (
            names[stage],
            *(
                _matrix_percentage(
                    None
                    if matrix.cells[stage][task] is None
                    else matrix.cells[stage][task]["success_rate"]
                )
                for task in range(matrix.sequence.num_tasks)
            ),
        )
        for stage in range(matrix.sequence.num_tasks)
    ]
    final_stage = matrix.sequence.num_tasks - 1
    final_cells = matrix.cells[final_stage]
    final_ap = average_performance(matrix.R, final_stage)
    final_forgetting = forgetting(matrix.R, final_stage)
    final_rows = [
        (name, _percentage(None if cell is None else cell["success_rate"]))
        for name, cell in zip(names, final_cells)
    ]
    lines = [
        "",
        "Success matrix (%)",
        "",
        *_table(("After task", *names), rows),
        "",
        "Final continual-learning summary",
        "",
        "Final per-task Success Rates",
        *_table(("Task", "Success Rate"), final_rows),
        "",
        f"Final Average Performance: {_percentage(final_ap)}",
        f"Average Forgetting: {_percentage_points(final_forgetting)}",
    ]
    return "\n".join(lines)


def print_stage_report(
    matrix: EvaluationMatrix,
    stage: int,
    *,
    file: TextIO | None = None,
    **kwargs,
):
    print(
        format_stage_report(matrix, stage, **kwargs),
        file=file or sys.stdout,
        flush=True,
    )


def print_final_report(matrix: EvaluationMatrix, *, file: TextIO | None = None):
    print(format_final_report(matrix), file=file or sys.stdout, flush=True)
