"""Success-primary checkpoint selection for standalone SAC experts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from train_sac import FixedSeedEvalCallback


class FixedSeedEvalCallbackTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
