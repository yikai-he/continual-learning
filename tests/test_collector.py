"""Generic repeated trajectory collection and trainer cleanup behavior."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.continual import collector
from src.continual.diffcrl import DiffCRLTrainer, StageContext
from src.continual.task_sequence import TaskSequence


class CollectorBatchTests(unittest.TestCase):
    def test_ordered_seeds_callbacks_and_progress_forwarding(self):
        env, policy = object(), object()
        resets, calls, progress = [], [], []

        def fake_collect(
            current_env,
            current_policy,
            task_id,
            *,
            seed,
            deterministic,
            on_reset,
        ):
            calls.append((current_env, current_policy, task_id, seed, deterministic))
            observation = np.full(39, seed, dtype=np.float32)
            on_reset(current_env, observation, seed)
            return f"trajectory-{seed}"

        def fake_tqdm(iterable, **kwargs):
            progress.append(kwargs)
            return iterable

        with (
            patch.object(collector, "collect_trajectory", side_effect=fake_collect),
            patch.object(collector, "tqdm", side_effect=fake_tqdm),
        ):
            trajectories = collector.collect_trajectories(
                env,
                policy,
                "hammer-v3",
                episodes=3,
                initial_seed=31,
                deterministic=True,
                on_reset=lambda e, observation, seed, episode: resets.append(
                    (e, observation.copy(), seed, episode)
                ),
                progress=True,
                description="Expert collection: hammer-v3",
            )

        self.assertEqual(
            trajectories, ["trajectory-31", "trajectory-32", "trajectory-33"]
        )
        self.assertEqual([call[3] for call in calls], [31, 32, 33])
        self.assertTrue(all(call[4] for call in calls))
        self.assertEqual([item[2:] for item in resets], [(31, 0), (32, 1), (33, 2)])
        self.assertEqual(
            progress,
            [
                {
                    "desc": "Expert collection: hammer-v3",
                    "unit": "trajectory",
                    "disable": False,
                }
            ],
        )

    def test_optional_reset_callback_remains_none(self):
        seen = []

        def fake_collect(*args, **kwargs):
            seen.append(kwargs["on_reset"])
            return Mock()

        with patch.object(collector, "collect_trajectory", side_effect=fake_collect):
            collector.collect_trajectories(
                object(), object(), "reach-v3", episodes=2, initial_seed=5
            )
        self.assertEqual(seen, [None, None])


class TrainerCollectionCleanupTests(unittest.TestCase):
    def trainer_and_context(self, directory):
        sequence = TaskSequence.from_names(["hammer-v3"])
        observation_space = Mock(shape=(39,))
        action_space = Mock(
            shape=(4,),
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
        )
        model = Mock(observation_space=observation_space, action_space=action_space)
        env = Mock(observation_space=observation_space, action_space=action_space)
        trainer = object.__new__(DiffCRLTrainer)
        trainer.config = SimpleNamespace(
            runtime=SimpleNamespace(seed=7, device="cpu"),
            environment=SimpleNamespace(reward_function_version="v2"),
            bc=SimpleNamespace(hidden_sizes=[8, 6]),
            diffusion=SimpleNamespace(action_clamp_epsilon=1e-4),
        )
        trainer.sequence = sequence
        trainer.experts = {"hammer-v3": Path("/tmp/expert.zip")}
        trainer.expert_hashes = {"hammer-v3": "sha"}
        trainer.policy = None
        trainer.training_banks = {}
        trainer.progress = False
        context = StageContext(0, sequence.task(0), 2, directory, None, None)
        return trainer, context, model, env

    def test_environment_closes_when_task_bank_creation_fails(self):
        with self.subTest("task bank failure"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp:
                trainer, context, model, env = self.trainer_and_context(Path(tmp))
                with (
                    patch("src.continual.diffcrl.SAC.load", return_value=model),
                    patch("src.continual.diffcrl.make_metaworld_env", return_value=env),
                    patch(
                        "src.continual.diffcrl.training_bank",
                        side_effect=RuntimeError("bank failure"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "bank failure"),
                ):
                    trainer._collect_current_task(context)
                env.close.assert_called_once_with()

    def test_environment_closes_when_batch_collection_fails(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            trainer, context, model, env = self.trainer_and_context(Path(tmp))
            with (
                patch("src.continual.diffcrl.SAC.load", return_value=model),
                patch("src.continual.diffcrl.make_metaworld_env", return_value=env),
                patch("src.continual.diffcrl.training_bank", return_value={}),
                patch(
                    "src.continual.diffcrl.collect_trajectories",
                    side_effect=RuntimeError("rollout failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "rollout failure"),
            ):
                trainer._collect_current_task(context)
            env.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
