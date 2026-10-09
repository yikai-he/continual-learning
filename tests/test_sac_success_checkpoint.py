"""Success-primary checkpoint selection for standalone SAC experts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from train_sac import FixedSeedEvalCallback, early_stopping_metadata
from src.config import EarlyStoppingConfig


class FixedSeedEvalCallbackTests(unittest.TestCase):
    def _early_stopping_callback(self, **overrides):
        settings = {
            "enabled": True,
            "min_steps": 0,
            "success_threshold": 0.95,
            "patience": 3,
            **overrides,
        }
        return FixedSeedEvalCallback(
            MagicMock(),
            eval_freq=20_000,
            n_eval_episodes=1,
            seed=0,
            log_path=Path("evaluations"),
            best_model_save_path=Path("best"),
            best_success_model_save_path=Path("best-success"),
            early_stopping=EarlyStoppingConfig(**settings),
        )

    def test_early_stopping_disabled_never_stops(self):
        callback = self._early_stopping_callback(enabled=False)
        callback.num_timesteps = 1_000_000
        for rate in (1.0, 1.0, 1.0, 1.0):
            self.assertTrue(callback._update_early_stopping(rate))
        self.assertFalse(callback.early_stopped)

    def test_one_qualifying_evaluation_does_not_stop(self):
        callback = self._early_stopping_callback()
        callback.num_timesteps = 100_000
        self.assertTrue(callback._update_early_stopping(0.96))
        self.assertEqual(callback.consecutive_successful_evaluations, 1)

    def test_three_consecutive_successes_stop_on_third(self):
        callback = self._early_stopping_callback()
        callback.num_timesteps = 100_000
        decisions = [callback._update_early_stopping(rate) for rate in (0.96, 0.97, 0.98)]
        self.assertEqual(decisions, [True, True, False])
        self.assertEqual(callback.stop_timestep, 100_000)
        self.assertEqual(callback.trigger_success_rate, 0.98)

    def test_failure_resets_consecutive_counter(self):
        callback = self._early_stopping_callback()
        callback.num_timesteps = 100_000
        rates = (0.96, 0.97, 0.90, 0.96, 0.97, 0.98)
        decisions = [callback._update_early_stopping(rate) for rate in rates]
        self.assertEqual(decisions, [True, True, True, True, True, False])

    def test_min_steps_uses_global_timestep(self):
        callback = self._early_stopping_callback(min_steps=100_000, patience=1)
        callback.num_timesteps = 80_000
        self.assertTrue(callback._update_early_stopping(1.0))
        self.assertFalse(callback.early_stopped)
        callback.num_timesteps = 100_000
        self.assertFalse(callback._update_early_stopping(1.0))

    def test_resumed_global_timestep_satisfies_min_steps(self):
        callback = self._early_stopping_callback(min_steps=100_000, patience=1)
        callback.num_timesteps = 500_000
        self.assertFalse(callback._update_early_stopping(0.95))
        self.assertEqual(callback.stop_timestep, 500_000)

    def test_exact_threshold_counts_as_success(self):
        callback = self._early_stopping_callback(patience=1)
        callback.num_timesteps = 100_000
        self.assertFalse(callback._update_early_stopping(0.95))

    def test_early_stopping_metadata_records_normal_and_triggered_runs(self):
        disabled = EarlyStoppingConfig()
        self.assertFalse(early_stopping_metadata(disabled)["triggered"])

        enabled = EarlyStoppingConfig(
            enabled=True, min_steps=100_000, success_threshold=0.95, patience=1
        )
        normal_callback = self._early_stopping_callback(
            min_steps=100_000, patience=2
        )
        normal_callback.num_timesteps = 1_000_000
        self.assertTrue(normal_callback._update_early_stopping(0.95))
        self.assertFalse(
            early_stopping_metadata(enabled, normal_callback)["triggered"]
        )

        callback = self._early_stopping_callback(min_steps=100_000, patience=1)
        callback.num_timesteps = 320_000
        self.assertFalse(callback._update_early_stopping(1.0))
        metadata = early_stopping_metadata(enabled, callback)
        self.assertTrue(metadata["triggered"])
        self.assertEqual(metadata["stop_timestep"], 320_000)
        self.assertEqual(metadata["trigger_success_rate"], 1.0)

    def test_fixed_bank_is_reused_and_task_set_seed_controls_identity(self) -> None:
        class Env:
            def __init__(self):
                self.unwrapped = self
                self.current = None
                self.seen = []

            def set_task(self, task):
                self.current = task

            def reset(self, seed=None):
                self.seen.append((self.current, seed))
                return np.zeros(39, dtype=np.float32), {}

            def step(self, _action):
                return np.zeros(39), 0.0, True, False, {"success": False}

        def bank(_task, seed):
            tasks = [f"{seed}:{index}" for index in range(2)]
            identities = [
                {"task_data_sha256": f"hash-{seed}-{index}"}
                for index in range(2)
            ]
            return tasks, identities

        with tempfile.TemporaryDirectory() as tmp, patch(
            "train_sac.fixed_mt1_tasks", side_effect=bank
        ), patch("train_sac.disable_task_sampling"):
            root = Path(tmp)
            for name in ("evaluations", "best_model", "best_success_model"):
                (root / name).mkdir()
            env = Env()
            callback = FixedSeedEvalCallback(
                env,
                eval_freq=1,
                n_eval_episodes=2,
                seed=20_000,
                task_name="reach-v3",
                task_set_seed=10_000,
                evaluation_mode="fixed-tasks",
                log_path=root / "evaluations",
                best_model_save_path=root / "best_model",
                best_success_model_save_path=root / "best_success_model",
            )
            callback.model = MagicMock()
            callback.model.predict.return_value = (np.zeros(4), None)
            callback._evaluate()
            callback.model.predict.return_value = (np.ones(4), None)
            callback._evaluate()
            self.assertEqual(
                env.seen,
                [
                    ("10000:0", 20_000),
                    ("10000:1", 20_001),
                    ("10000:0", 20_000),
                    ("10000:1", 20_001),
                ],
            )
            self.assertEqual(
                callback.task_hashes, ("hash-10000-0", "hash-10000-1")
            )
            other = FixedSeedEvalCallback(
                Env(),
                eval_freq=1,
                n_eval_episodes=2,
                seed=20_000,
                task_name="reach-v3",
                task_set_seed=10_001,
                evaluation_mode="fixed-tasks",
                log_path=root / "evaluations",
                best_model_save_path=root / "best_model",
                best_success_model_save_path=root / "best_success_model",
            )
            self.assertNotEqual(callback.task_hashes, other.task_hashes)

    def test_success_is_primary_and_return_breaks_ties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("evaluations", "best_model", "best_success_model"):
                (root / name).mkdir()
            callback = FixedSeedEvalCallback(
                MagicMock(),
                eval_freq=1,
                n_eval_episodes=2,
                seed=10_000,
                log_path=root / "evaluations",
                best_model_save_path=root / "best_model",
                best_success_model_save_path=root / "best_success_model",
            )
            callback.model = MagicMock()
            callback._logger = MagicMock()
            callback.n_calls = 1
            callback._evaluate = MagicMock(
                side_effect=[
                    ([10.0, 10.0], [300, 300], [True, False]),
                    ([20.0, 20.0], [300, 300], [True, False]),
                    ([5.0, 5.0], [300, 300], [True, True]),
                    ([100.0, 100.0], [300, 300], [True, False]),
                ]
            )

            for step in (5_000, 10_000, 15_000, 20_000):
                callback.num_timesteps = step
                self.assertTrue(callback._on_step())

            paths = [call.args[0] for call in callback.model.save.call_args_list]
            self.assertEqual(
                paths,
                [
                    root / "best_model/best_model",
                    root / "best_success_model/best_success_model",
                    root / "best_model/best_model",
                    root / "best_success_model/best_success_model",
                    root / "best_success_model/best_success_model",
                    root / "best_model/best_model",
                ],
            )
            archive = np.load(root / "evaluations/evaluations.npz")
            np.testing.assert_array_equal(archive["seeds"], [10_000, 10_001])
            np.testing.assert_array_equal(
                archive["timesteps"], [5_000, 10_000, 15_000, 20_000]
            )
            np.testing.assert_array_equal(archive["successes"][-1], [True, False])
            np.testing.assert_allclose(archive["return_stddevs"], 0.0)


if __name__ == "__main__":
    unittest.main()
