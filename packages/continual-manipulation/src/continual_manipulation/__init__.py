"""Public API for the ContinualManipulation benchmark."""

from .environments import EnvironmentSettings, make_env
from .evaluation import (
    EvaluationMatrix,
    evaluate_stage,
    evaluate_task,
    format_evaluation_result,
    format_final_report,
    format_stage_report,
)
from .interfaces import Policy
from .metrics import average_performance, forgetting, forward_transfer
from .sequences import TaskSequence, TaskSpec, resolve_named_sequence
from .switching import TaskSwitcher

__version__ = "0.1.0"

__all__ = [
    "EnvironmentSettings",
    "EvaluationMatrix",
    "Policy",
    "TaskSequence",
    "TaskSpec",
    "TaskSwitcher",
    "average_performance",
    "evaluate_stage",
    "evaluate_task",
    "forgetting",
    "format_evaluation_result",
    "format_final_report",
    "format_stage_report",
    "forward_transfer",
    "make_env",
    "resolve_named_sequence",
]
