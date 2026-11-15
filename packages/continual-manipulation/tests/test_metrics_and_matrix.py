import numpy as np
import pytest

from continual_manipulation import (
    EvaluationMatrix, TaskSequence, average_performance, forgetting, forward_transfer,
)


def result(success_rate: float) -> dict:
    return {
        "mean_return": success_rate,
        "std_return": 0.0,
        "success_rate": success_rate,
        "mean_episode_length": 200.0,
    }


def test_matrix_serialization_and_metrics_are_stable() -> None:
    matrix = EvaluationMatrix(TaskSequence.from_names(("reach-v3", "push-v3")))
    matrix.record(0, 0, result(0.8))
    matrix.record(1, 0, result(0.5))
    matrix.record(1, 1, result(0.7))
    assert matrix.to_dict()["kind"] == "continual_stage_evaluation"
    assert matrix.to_dict()["R"] == [[0.8, None], [0.5, 0.7]]
    assert average_performance(matrix.R, 1) == 0.6
    assert forgetting(matrix.R, 1) == pytest.approx(0.15)


def test_forward_transfer_requires_aligned_curves() -> None:
    assert forward_transfer() is None
    assert forward_transfer({"a": np.array([0.2, 0.4])}, {"a": np.array([0.1, 0.3])}) == pytest.approx(0.125)
