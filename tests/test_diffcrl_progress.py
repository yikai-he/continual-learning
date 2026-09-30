"""DiffCRL progress flags remain explicit for piped console output."""

import inspect
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

from src.continual import (
    bc_policy,
    collector,
    diffcrl,
    diffcrl_replay,
    evaluation,
    trajectory_diffusion,
)


class DiffCRLProgressTests(unittest.TestCase):
    def test_enabled_tqdm_renders_to_non_tty_stream(self):
        stream = io.StringIO()
        for _ in tqdm(range(1), disable=False, file=stream, desc="visible"):
            pass
        self.assertIn("visible", stream.getvalue())
        self.assertIn("100%", stream.getvalue())

    def test_progress_disable_flags_are_explicit(self):
        self.assertEqual(
            inspect.getsource(diffcrl).count("disable=not self.progress"), 0
        )
        self.assertIn("disable=not request.progress", inspect.getsource(diffcrl_replay))
        for module in (bc_policy, collector, trajectory_diffusion, evaluation):
            source = inspect.getsource(module)
            self.assertIn("disable=not progress", source)
            self.assertNotIn("disable=None if progress else True", source)

    def test_bc_progress_boolean_reaches_tqdm_as_explicit_disable(self):
        seen = []

        def recording_tqdm(iterable, **kwargs):
            seen.append(kwargs["disable"])
            return iterable

        data = TensorDataset(torch.zeros(2, 39), torch.zeros(2, 4))
        with patch.object(bc_policy, "tqdm", side_effect=recording_tqdm):
            for progress in (True, False):
                policy = bc_policy.GeneralPolicy(
                    -np.ones(4), np.ones(4), hidden_sizes=(4, 4)
                )
                bc_policy.fit_bc(policy, data, data, epochs=1, progress=progress)
        self.assertEqual(seen, [False, True])

    def test_phase_messages_respect_progress_flag(self):
        trainer = object.__new__(diffcrl.DiffCRLTrainer)
        with patch.object(diffcrl.tqdm, "write") as write:
            trainer.progress = True
            result = trainer._progress_message("Training diffusion", Epochs=3)
            self.assertIsNone(result)
            write.assert_called_once_with("Training diffusion...\n  Epochs: 3")

            write.reset_mock()
            trainer.progress = False
            result = trainer._progress_message("Training diffusion", Epochs=3)
            self.assertIsNone(result)
            write.assert_not_called()

    def test_trainer_passes_progress_to_long_running_helpers(self):
        source = inspect.getsource(diffcrl.DiffCRLTrainer)
        self.assertGreaterEqual(source.count("progress=self.progress"), 3)


if __name__ == "__main__":
    unittest.main()
