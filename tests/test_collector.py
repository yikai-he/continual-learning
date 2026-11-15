"""Generic repeated trajectory collection and trainer cleanup behavior."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.continual import collector
from src.continual.diffcrl import DiffCRLTrainer, StageContext
from continual_manipulation.evaluation.rollout import Trajectory
from continual_manipulation.sequences import TaskSequence
from src.support.expert_manifest import ExpertIdentity


class CollectorBatchTests(unittest.TestCase):
    def test_action_mode_is_forwarded_independently_of_collection_mode(self):
        for deterministic in (True, False):
            for mode in ("all", "success_preferred"):
                calls = []

                def fake_collect(*args, **kwargs):
                    calls.append(kwargs["deterministic"])
                    return SimpleNamespace(success=True, episode_return=1.0)

                with self.subTest(deterministic=deterministic, mode=mode), patch.object(
                    collector, "_collect_trajectory", side_effect=fake_collect
                ):
                    _, metadata = collector.collect_trajectories(
                        object(), object(), "reach-v3", episodes=2,
                        initial_seed=9, deterministic=deterministic,
                        collection_mode=mode, max_attempts=4,
                        return_metadata=True,
                    )
                    self.assertEqual(calls, [deterministic, deterministic])
                    self.assertEqual(
                        metadata["expert_action_mode"],
                        "deterministic" if deterministic else "stochastic",
                    )

    def test_stochastic_collection_rng_is_repeatable_isolated_and_seeded(self):
        def collect(seed):
            draws = []

            def fake_collect(*args, **kwargs):
                draw = float(torch.rand(()))
                draws.append(draw)
                return SimpleNamespace(success=True, episode_return=draw)

            with patch.object(collector, "_collect_trajectory", side_effect=fake_collect):
                collector.collect_trajectories(
                    object(), object(), "reach-v3", episodes=3,
                    initial_seed=seed, deterministic=False,
                )
            return draws

        torch.manual_seed(1234)
        state = torch.get_rng_state().clone()
        first = collect(17)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertEqual(first, collect(17))
        self.assertNotEqual(first, collect(18))

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

        with patch.object(
            collector, "_collect_trajectory", side_effect=fake_collect
        ), patch.object(collector, "tqdm", side_effect=fake_tqdm):
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

        with patch.object(collector, "_collect_trajectory", side_effect=fake_collect):
            collector.collect_trajectories(
                object(), object(), "reach-v3", episodes=2, initial_seed=5
            )
        self.assertEqual(seen, [None, None])

    def test_success_preferred_selects_successes_then_attempt_order_fallback(self):
        outcomes = iter([False, True, False, False])
        calls, resets = [], []

        def fake_collect(env, policy, task_id, *, seed, deterministic, on_reset):
            attempt = seed - 10
            calls.append(seed)
            on_reset(env, np.zeros(39, dtype=np.float32), seed)
            return SimpleNamespace(success=next(outcomes), attempt=attempt)

        with patch.object(collector, "_collect_trajectory", side_effect=fake_collect):
            selected, metadata = collector.collect_trajectories(
                object(),
                object(),
                "reach-v3",
                episodes=3,
                initial_seed=10,
                deterministic=True,
                on_reset=lambda env, obs, seed, attempt: resets.append(
                    (seed, attempt)
                ),
                collection_mode="success_preferred",
                max_attempts=4,
                return_metadata=True,
            )

        self.assertEqual([item.attempt for item in selected], [1, 0, 2])
        self.assertEqual(calls, [10, 11, 12, 13])
        self.assertEqual(resets, [(10, 0), (11, 1), (12, 2), (13, 3)])
        self.assertEqual(metadata["selected_attempt_indices"], [1, 0, 2])
        self.assertEqual(metadata["selected_seeds"], [11, 10, 12])
        self.assertEqual(metadata["actual_attempts"], 4)
        self.assertEqual(metadata["successful_attempts"], 1)
        self.assertEqual(metadata["failed_attempts"], 3)
        self.assertEqual(metadata["accepted_successful"], 1)
        self.assertEqual(metadata["accepted_unsuccessful"], 2)
        self.assertEqual(metadata["accepted_fallback"], 2)
        self.assertEqual(metadata["rejected_trajectories"], 1)
        self.assertEqual(metadata["selected_success_fraction"], 1 / 3)

    def test_success_preferred_stops_as_soon_as_all_successes_are_collected(self):
        calls = []

        def fake_collect(*args, **kwargs):
            calls.append(kwargs["seed"])
            return SimpleNamespace(success=True, seed=kwargs["seed"])

        with patch.object(collector, "_collect_trajectory", side_effect=fake_collect):
            selected, metadata = collector.collect_trajectories(
                object(), object(), "reach-v3",
                episodes=2, initial_seed=20,
                collection_mode="success_preferred", max_attempts=6,
                return_metadata=True,
            )
        self.assertEqual([item.seed for item in selected], [20, 21])
        self.assertEqual(calls, [20, 21])
        self.assertEqual(metadata["actual_attempts"], 2)
        self.assertEqual(metadata["accepted_successful"], 2)
        self.assertEqual(metadata["accepted_fallback"], 0)

    def test_all_mode_preserves_attempt_order_and_exact_episode_budget(self):
        outcomes = iter([False, True, False])

        def fake_collect(*args, **kwargs):
            return SimpleNamespace(success=next(outcomes), seed=kwargs["seed"])

        with patch.object(collector, "_collect_trajectory", side_effect=fake_collect):
            selected, metadata = collector.collect_trajectories(
                object(), object(), "reach-v3",
                episodes=3, initial_seed=5,
                collection_mode="all", max_attempts=30,
                return_metadata=True,
            )
        self.assertEqual([item.seed for item in selected], [5, 6, 7])
        self.assertEqual(metadata["maximum_attempts"], 3)
        self.assertEqual(metadata["accepted_successful"], 1)
        self.assertEqual(metadata["accepted_unsuccessful"], 2)
        self.assertEqual(metadata["accepted_fallback"], 0)
        self.assertEqual(metadata["rejected_trajectories"], 0)

    def test_collection_mode_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, "collection_mode"):
            collector.collect_trajectories(
                object(), object(), "reach-v3", episodes=2, initial_seed=0,
                collection_mode="successful_only",
            )
        with self.assertRaisesRegex(ValueError, "max_attempts"):
            collector.collect_trajectories(
                object(), object(), "reach-v3", episodes=2, initial_seed=0,
                collection_mode="success_preferred", max_attempts=1,
            )


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
            environment=SimpleNamespace(
                backend="metaworld-v3", reward_function_version="v2"
            ),
            continual=SimpleNamespace(
                collection_mode="all",
                collection_max_attempt_multiplier=2,
                expert_action_mode="deterministic",
            ),
            bc=SimpleNamespace(hidden_sizes=[8, 6]),
            diffusion=SimpleNamespace(action_clamp_epsilon=1e-4),
        )
        trainer.sequence = sequence
        trainer.experts = {"hammer-v3": Path("/tmp/expert.zip")}
        trainer.expert_hashes = {"hammer-v3": "sha"}
        trainer.expert_identities = {
            "hammer-v3": ExpertIdentity(
                task_name="hammer-v3",
                checkpoint_path="/tmp/expert.zip",
                checkpoint_sha256="sha",
                backend="metaworld-v3",
                algorithm="SAC",
                reward_function_version="v2",
                training_seed=0,
                training_config_reference=None,
                training_config_sha256=None,
                observation_shape=(39,),
                action_shape=(4,),
                horizon=200,
                qualification=None,
                manifest_path=None,
                legacy=True,
            )
        }
        trainer.policy = None
        trainer.training_banks = {}
        trainer.progress = False
        trainer.horizon = 200
        context = StageContext(0, sequence.task(0), 2, directory, None, None)
        return trainer, context, model, env

    def test_environment_closes_when_task_bank_creation_fails(self):
        with self.subTest("task bank failure"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp:
                trainer, context, model, env = self.trainer_and_context(Path(tmp))
                with patch(
                    "src.continual.diffcrl.SAC.load", return_value=model
                ), patch(
                    "src.continual.diffcrl.make_env", return_value=env
                ), patch(
                    "src.continual.diffcrl.training_bank",
                    side_effect=RuntimeError("bank failure"),
                ), self.assertRaisesRegex(RuntimeError, "bank failure"):
                    trainer._collect_current_task(context)
                env.close.assert_called_once_with()

    def test_environment_closes_when_batch_collection_fails(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            trainer, context, model, env = self.trainer_and_context(Path(tmp))
            with patch(
                "src.continual.diffcrl.SAC.load", return_value=model
            ), patch(
                "src.continual.diffcrl.make_env", return_value=env
            ), patch(
                "src.continual.diffcrl.training_bank", return_value={}
            ), patch(
                "src.continual.diffcrl.collect_trajectories",
                side_effect=RuntimeError("rollout failure"),
            ), self.assertRaisesRegex(RuntimeError, "rollout failure"):
                trainer._collect_current_task(context)
            env.close.assert_called_once_with()

    def test_diffcrl_records_selected_attempt_provenance(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trainer, context, model, env = self.trainer_and_context(directory)
            trainer.config.continual = SimpleNamespace(
                collection_mode="success_preferred",
                collection_max_attempt_multiplier=2,
                expert_action_mode="deterministic",
            )
            trainer.config.bc = SimpleNamespace(
                task_conditioning="none",
                initial_log_std=-3.0,
                activation="relu",
                hidden_dim=256,
                loss="deterministic_mse",
            )
            trainer.config.diffusion = SimpleNamespace(action_clamp_epsilon=1e-4)
            observations = np.zeros((200, 39), dtype=np.float32)
            actions = np.zeros((200, 4), dtype=np.float32)
            rewards = np.zeros(200, dtype=np.float64)
            trajectories = [
                Trajectory(
                    observations=observations.copy(),
                    actions=actions.copy(),
                    rewards=rewards.copy(),
                    task_id="hammer-v3",
                    episode_return=0.0,
                    success=success,
                    final_success=success,
                    length=200,
                )
                for success in (True, False)
            ]
            collection = {
                "mode": "success_preferred",
                "requested_trajectories": 2,
                "maximum_attempts": 4,
                "actual_attempts": 4,
                "successful_attempts": 1,
                "failed_attempts": 3,
                "accepted_successful": 1,
                "accepted_unsuccessful": 1,
                "accepted_fallback": 1,
                "rejected_trajectories": 2,
                "selected_attempt_indices": [1, 0],
                "selected_seeds": [8, 7],
                "selected_success_fraction": 0.5,
            }

            def fake_collection(*args, **kwargs):
                for attempt in range(4):
                    kwargs["on_reset"](
                        env, np.zeros(39, dtype=np.float32), 7 + attempt, attempt
                    )
                self.assertEqual(kwargs["collection_mode"], "success_preferred")
                self.assertEqual(kwargs["max_attempts"], 4)
                self.assertTrue(kwargs["deterministic"])
                return trajectories, collection

            with patch("src.continual.diffcrl.SAC.load", return_value=model), patch(
                "src.continual.diffcrl.make_env", return_value=env
            ), patch(
                "src.continual.diffcrl.validate_expert_spaces"
            ), patch(
                "src.continual.diffcrl.training_bank", return_value={}
            ), patch(
                "src.continual.diffcrl.selected_configuration",
                side_effect=lambda env, bank, obs, attempt, reset_seed=None: {
                    "configuration_index": attempt
                },
            ), patch(
                "src.continual.diffcrl.collect_trajectories",
                side_effect=fake_collection,
            ):
                result = trainer._collect_current_task(context)

            self.assertEqual(result.source["seeds"], [8, 7])
            self.assertEqual(result.source["collection"], {
                key: value
                for key, value in collection.items()
                if key not in ("selected_attempt_indices", "selected_seeds")
            })
            self.assertEqual(
                [item["configuration_index"] for item in result.collected_configurations],
                [1, 0],
            )
            env.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
