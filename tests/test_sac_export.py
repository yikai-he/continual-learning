"""Compact result exports for completed standalone SAC runs."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_sac import export_run_summaries


class SacExportTests(unittest.TestCase):
    def test_exports_existing_evaluations_without_rerunning_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "evaluations").mkdir()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "task": "push-v3",
                        "env_backend": "metaworld-v3",
                        "seed": 7,
                        "steps": 10_000,
                        "reward_function_version": "v2",
                        "effective_horizon": 200,
                    }
                )
            )
            np.savez(
                run_dir / "evaluations/evaluations.npz",
                timesteps=np.array([0, 5_000, 10_000]),
                results=np.array([[1.0, 3.0], [9.0, 11.0], [5.0, 7.0]]),
                successes=np.array([[False, False], [False, False], [True, False]]),
                return_stddevs=np.array([1.0, 1.0, 1.0]),
            )

            export_run_summaries(run_dir)

            with (run_dir / "evaluation_history.csv").open(newline="") as source:
                csv_rows = list(csv.DictReader(source))
            self.assertEqual([int(row["step"]) for row in csv_rows], [0, 5_000, 10_000])
            self.assertEqual(float(csv_rows[-1]["success_rate"]), 0.5)

            summary = json.loads((run_dir / "run_summary.json").read_text())
            self.assertEqual(summary["initial"]["mean_return"], 2.0)
            self.assertEqual(summary["final"]["step"], 10_000)
            self.assertEqual(summary["best_success"]["step"], 10_000)
            self.assertEqual(summary["best_return"]["step"], 5_000)
            self.assertNotEqual(
                summary["best_success"]["model_path"],
                summary["best_return"]["model_path"],
            )

            markdown = (run_dir / "EXPERIMENT_SUMMARY.md").read_text()
            self.assertIn("| 0 | 0.0% | 2.0000 | 1.0000 |", markdown)
            self.assertIn("| 10000 | 50.0% | 6.0000 | 1.0000 |", markdown)
            self.assertNotIn("interpretation", markdown.lower())


if __name__ == "__main__":
    unittest.main()
