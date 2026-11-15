import continual_manipulation as cm


def test_public_api_and_distribution_version() -> None:
    assert cm.__version__ == "0.1.0"
    for name in (
        "EnvironmentSettings", "EvaluationMatrix", "Policy", "TaskSequence",
        "TaskSpec", "TaskSwitcher", "average_performance", "evaluate_stage",
        "evaluate_task", "forgetting", "format_evaluation_result",
        "format_final_report", "format_stage_report", "forward_transfer",
        "make_env", "resolve_named_sequence",
    ):
        assert getattr(cm, name) is not None


def test_named_sequence_identity() -> None:
    assert cm.resolve_named_sequence("cm3") == (
        "reach-v3", "push-v3", "hammer-v3"
    )
    sequence = cm.TaskSequence.from_names(cm.resolve_named_sequence("cm3"))
    assert tuple(sequence.task_names) == tuple(task.task_id for task in sequence.tasks)
