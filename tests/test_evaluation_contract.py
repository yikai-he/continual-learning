"""Persistence and protocol-consistency contracts for continual evaluation."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.continual.evaluation import EvaluationMatrix, evaluate_stage
from src.continual.task_sequence import TaskSequence


def evaluation_result(*, mode="fixed-tasks", seed=10000, task_set_seed=10000):
    return {
        "mean_return": 1.5,
        "std_return": 0.25,
        "success_rate": 0.5,
        "mean_episode_length": 200.0,
        "evaluation_mode": mode,
        "task_set_seed": task_set_seed if mode == "fixed-tasks" else None,
        "seeds": [seed],
    }


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.sequence = TaskSequence.from_names(["reach-v3", "push-v3"])

    def test_matrix_serialization_round_trip(self):
        matrix = EvaluationMatrix(self.sequence)
        matrix.record(0, 0, evaluation_result())
        matrix.stage_provenance["0"] = {"checkpoint": "policy.pt"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps(matrix.to_dict()))
            loaded = EvaluationMatrix.load(path)

        self.assertEqual(loaded.sequence, matrix.sequence)
        self.assertEqual(loaded.cells, matrix.cells)
        self.assertEqual(loaded.stage_provenance, matrix.stage_provenance)
        self.assertEqual(loaded.to_dict(), matrix.to_dict())

    def test_stage_rejects_protocol_mismatch_before_evaluation(self):
        base = EvaluationMatrix(self.sequence)
        base.record(0, 0, evaluation_result())

        cases = (
            ({"evaluation_mode": "sampled"}, "mix evaluation modes"),
            ({"evaluation_mode": "fixed-tasks", "seed": 10001}, "bank or reset seeds"),
            (
                {"evaluation_mode": "fixed-tasks", "task_set_seed": 10001},
                "bank or reset seeds",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), patch(
                "src.continual.evaluation.evaluate_task"
            ) as evaluate:
                matrix = deepcopy(base)
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_stage(None, matrix, 1, **overrides)
                evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
