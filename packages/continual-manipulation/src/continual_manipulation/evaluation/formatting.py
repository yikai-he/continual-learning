"""Human-readable formatting for recorded benchmark evaluations."""

from ..metrics import average_performance, forgetting
from .matrix import EvaluationMatrix

MISSING = "—"


def _percentage(value):
    return MISSING if value is None else f"{100 * value:.1f}%"


def _matrix_percentage(value):
    return MISSING if value is None else f"{100 * value:.1f}"


def _percentage_points(value):
    return MISSING if value is None else f"{100 * value:.1f} percentage points"


def _return(value):
    return MISSING if value is None else f"{value:.3f}"


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
        f"Current Average Performance: {_percentage(average_performance(matrix.R, stage))}",
        f"Forgetting: {_percentage_points(forgetting(matrix.R, stage))}",
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
    final_rows = [
        (name, _percentage(None if cell is None else cell["success_rate"]))
        for name, cell in zip(names, matrix.cells[final_stage])
    ]
    lines = [
        "", "Success matrix (%)", "",
        *_table(("After task", *names), rows),
        "", "Final continual-learning summary", "",
        "Final per-task Success Rates",
        *_table(("Task", "Success Rate"), final_rows),
        "",
        f"Final Average Performance: {_percentage(average_performance(matrix.R, final_stage))}",
        f"Average Forgetting: {_percentage_points(forgetting(matrix.R, final_stage))}",
    ]
    return "\n".join(lines)
