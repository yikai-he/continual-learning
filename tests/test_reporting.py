"""Console reporting uses recorded continual results without changing them."""

import unittest
from copy import deepcopy

from src.continual.evaluation import EvaluationMatrix, running_evaluation_statistics
from src.continual.reporting import (
    format_evaluation_result,
    format_final_report,
    format_stage_report,
)
from src.continual.task_sequence import TaskSequence


def result(success, mean_return):
    return {
        "success_rate": success,
        "mean_return": mean_return,
        "std_return": 1.0,
        "mean_episode_length": 200.0,
    }


class ReportingTests(unittest.TestCase):
    def setUp(self):
        sequence = TaskSequence.from_names(["reach-v3", "handle-v3", "hammer-v3"])
        self.matrix = EvaluationMatrix(sequence)
        self.matrix.record(0, 0, result(0.98, 10.0))
        self.matrix.record(1, 0, result(0.25, 4.0))
        self.matrix.record(1, 1, result(1.0, 20.0))

    def test_final_matrix_formats_percentages_missing_cells_and_task_order(self):
        report = format_final_report(self.matrix)
        self.assertIn("Success matrix (%)", report)
        self.assertIn("98.0", report)
        self.assertIn("100.0", report)
        hammer_row = next(
            line for line in report.splitlines() if line.startswith("hammer-v3")
        )
        self.assertEqual(hammer_row.count("—"), 3)
        header = next(
            line for line in report.splitlines() if line.startswith("After task")
        )
        self.assertLess(header.index("reach-v3"), header.index("handle-v3"))
        self.assertLess(header.index("handle-v3"), header.index("hammer-v3"))

    def test_stage_report_uses_recorded_returns_and_established_metrics(self):
        report = format_stage_report(
            self.matrix,
            1,
            task_training_steps=500,
            global_timesteps=1000,
        )
        self.assertIn("Task training steps: 500", report)
        self.assertIn("Global timesteps: 1000", report)
        self.assertIn("reach-v3", report)
        self.assertIn("25.0%", report)
        self.assertIn("4.000", report)
        self.assertIn("Current Average Performance: 62.5%", report)
        self.assertIn("Forgetting: 36.5 percentage points", report)
        self.assertNotIn("Forward Transfer", report)

    def test_running_statistics_and_final_line_match_stored_values(self):
        success, mean_return = running_evaluation_statistics(
            [100.0, 148.407, 200.0], [True, False, True]
        )
        stored = {"success_rate": success, "mean_return": mean_return}
        self.assertAlmostEqual(success, 2 / 3)
        self.assertAlmostEqual(mean_return, 149.469)
        self.assertEqual(
            format_evaluation_result("hammer-v3", stored),
            "Result: hammer-v3 | Success Rate: 66.7% | Mean Return: 149.469",
        )

    def test_formatting_does_not_mutate_evaluation_matrix(self):
        cells = deepcopy(self.matrix.cells)
        provenance = deepcopy(self.matrix.stage_provenance)
        format_stage_report(
            self.matrix,
            1,
            task_training_steps=None,
            global_timesteps=None,
        )
        format_final_report(self.matrix)
        self.assertEqual(self.matrix.cells, cells)
        self.assertEqual(self.matrix.stage_provenance, provenance)


if __name__ == "__main__":
    unittest.main()
