"""Lightweight tests for faithful single-task SAC checkpoint resume."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import train_sac


class SACResumeTests(unittest.TestCase):
    def test_replay_buffer_path_matches_sb3_checkpoint_naming(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "sac_500000_steps.zip"
            replay = Path(tmp) / "sac_replay_buffer_500000_steps.pkl"
            checkpoint.write_bytes(b"model")
            replay.write_bytes(b"buffer")
            self.assertEqual(train_sac.replay_buffer_path(checkpoint), replay)

    def test_ambiguous_replay_buffers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "sac_500000_steps.zip"
            checkpoint.write_bytes(b"model")
            (Path(tmp) / "sac_replay_buffer_500000_steps.pkl").write_bytes(b"one")
            (Path(tmp) / "sac_replay_buffer_500000_steps_copy.pkl").write_bytes(b"two")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                train_sac.replay_buffer_path(checkpoint)

    def test_global_checkpoint_boundaries(self):
        for start in (500_000, 525_000, 549_999):
            with self.subTest(start=start):
                self.assertEqual(
                    train_sac.next_global_boundary(start, 50_000), 550_000
                )

    def test_global_checkpoint_callback_saves_model_and_replay_at_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = train_sac.GlobalCheckpointCallback(
                save_freq=50_000,
                save_path=tmp,
                name_prefix="sac",
                save_replay_buffer=True,
            )
            model = MagicMock(num_timesteps=525_000)
            model.replay_buffer = MagicMock()
            callback.model = model
            callback._on_training_start()
            callback.num_timesteps = 549_999
            self.assertTrue(callback._on_step())
            model.save.assert_not_called()
            callback.num_timesteps = 550_000
            model.num_timesteps = 550_000
            self.assertTrue(callback._on_step())
            model.save.assert_called_once_with(
                str(Path(tmp) / "sac_550000_steps.zip")
            )
            model.save_replay_buffer.assert_called_once_with(
                str(Path(tmp) / "sac_replay_buffer_550000_steps.pkl")
            )

    def test_global_evaluation_boundaries(self):
        self.assertEqual(
            train_sac.next_global_boundary(500_000, 20_000), 520_000
        )
        self.assertEqual(
            train_sac.next_global_boundary(505_000, 20_000), 520_000
        )

    def test_resume_does_not_evaluate_at_training_start(self):
        callback = train_sac.FixedSeedEvalCallback(
            MagicMock(),
            eval_freq=20_000,
            n_eval_episodes=1,
            seed=0,
            log_path=Path("evaluations"),
            best_model_save_path=Path("best"),
            best_success_model_save_path=Path("best-success"),
            evaluate_on_training_start=False,
        )
        callback.model = MagicMock(num_timesteps=500_000)
        callback._record_evaluation = MagicMock()
        callback._on_training_start()
        callback._record_evaluation.assert_not_called()
        self.assertEqual(callback._next_evaluation_timestep, 520_000)

    def test_fresh_run_evaluates_at_zero_and_schedules_first_boundary(self):
        callback = train_sac.FixedSeedEvalCallback(
            MagicMock(),
            eval_freq=20_000,
            n_eval_episodes=1,
            seed=0,
            log_path=Path("evaluations"),
            best_model_save_path=Path("best"),
            best_success_model_save_path=Path("best-success"),
        )
        callback.model = MagicMock(num_timesteps=0)
        callback._record_evaluation = MagicMock()
        callback._on_training_start()
        callback._record_evaluation.assert_called_once_with(allow_early_stop=False)
        self.assertEqual(callback._next_evaluation_timestep, 20_000)

    def test_remaining_timesteps_uses_total_target(self):
        self.assertEqual(train_sac.remaining_timesteps(1_000_000, 400_000), 600_000)

    def test_target_at_or_before_checkpoint_is_rejected(self):
        for target in (399_999, 400_000):
            with self.subTest(target=target), self.assertRaisesRegex(
                ValueError, "meets or exceeds"
            ):
                train_sac.remaining_timesteps(target, 400_000)

    def test_missing_replay_buffer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "sac_10_steps.zip"
            checkpoint.write_bytes(b"model")
            with self.assertRaisesRegex(FileNotFoundError, "Model-only"):
                train_sac.load_resumable_sac(checkpoint, MagicMock(), device="cpu")

    def test_load_preserves_timesteps_and_restores_replay_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "sac_400000_steps.zip"
            replay = Path(tmp) / "sac_replay_buffer_400000_steps.pkl"
            checkpoint.write_bytes(b"model")
            replay.write_bytes(b"buffer")
            model = MagicMock()
            model.num_timesteps = 400_000
            model.replay_buffer = MagicMock()
            env = MagicMock()
            with patch.object(train_sac.SAC, "load", return_value=model) as load:
                loaded, replay_path = train_sac.load_resumable_sac(
                    checkpoint, env, device="cpu"
                )
            self.assertIs(loaded, model)
            self.assertEqual(loaded.num_timesteps, 400_000)
            self.assertEqual(replay_path, replay.resolve())
            load.assert_called_once_with(checkpoint.resolve(), env=env, device="cpu")
            model.load_replay_buffer.assert_called_once_with(replay.resolve())

    def test_incompatible_spaces_have_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "sac_10_steps.zip"
            replay = Path(tmp) / "sac_replay_buffer_10_steps.pkl"
            checkpoint.write_bytes(b"model")
            replay.write_bytes(b"buffer")
            with patch.object(
                train_sac.SAC,
                "load",
                side_effect=ValueError("Observation spaces do not match"),
            ), self.assertRaisesRegex(ValueError, "current task.*observation or action"):
                train_sac.load_resumable_sac(checkpoint, MagicMock(), device="cpu")


if __name__ == "__main__":
    unittest.main()
