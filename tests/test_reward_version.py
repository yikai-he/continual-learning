"""Reward selection reaches environment creation without changing its contract."""

import unittest
from importlib.metadata import version
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from continual_manipulation.evaluation import protocols as evaluation
from continual_manipulation.sequences import TaskSequence
from continual_manipulation.switching import TaskSwitcher
from continual_manipulation.task_banks import bank_hashes
from src.continual.task_bank import reconstruct_goals
from continual_manipulation.environments import make_metaworld_env


class RewardVersionTests(unittest.TestCase):
    @pytest.mark.integration
    def test_real_environment_versions_and_unchanged_contract(self):
        observations = []
        for reward_version in ("v1", "v2"):
            with self.subTest(version=reward_version):
                env = make_metaworld_env(
                    "reach-v3", 0, reward_function_version=reward_version
                )
                try:
                    self.assertEqual(
                        env.unwrapped.reward_function_version, reward_version
                    )
                    self.assertEqual(env.observation_space.shape, (39,))
                    self.assertEqual(env.action_space.shape, (4,))
                    self.assertEqual(env.spec.max_episode_steps, 200)
                    observation, _ = env.reset(seed=0)
                    observations.append(observation)
                    self.assertTrue(
                        np.isfinite(env.step(np.zeros(4, dtype=np.float32))[1])
                    )
                finally:
                    env.close()
        np.testing.assert_array_equal(*observations)
        env = make_metaworld_env("reach-v3", 0)
        try:
            self.assertEqual(env.unwrapped.reward_function_version, "v2")
        finally:
            env.close()

    def test_invalid_version_rejected_before_gym_make(self):
        with patch(
            "continual_manipulation.environments.metaworld_v3.gym.make"
        ) as make:
            for value in ("v3", None, 1):
                with self.assertRaisesRegex(ValueError, "reward_function_version"):
                    make_metaworld_env("reach-v3", 0, reward_function_version=value)
            make.assert_not_called()

    def test_task_switch_and_stage_evaluation_forward_version(self):
        sequence = TaskSequence.from_names(["reach-v3"])
        for reward_version in ("v1", "v2"):
            factory = MagicMock()
            with TaskSwitcher(
                sequence, factory=factory, reward_function_version=reward_version
            ) as switcher:
                switcher.switch(0, seed=3)
            factory.assert_called_once_with(
                "reach-v3", 3, reward_function_version=reward_version
            )
            matrix = evaluation.EvaluationMatrix(sequence)
            result = dict(
                mean_return=0.0,
                std_return=0.0,
                success_rate=0.0,
                mean_episode_length=200.0,
            )
            with patch.object(
                evaluation, "evaluate_task", return_value=result
            ) as evaluate:
                evaluation.evaluate_stage(
                    object(),
                    matrix,
                    0,
                    episodes=1,
                    reward_function_version=reward_version,
                )
                self.assertEqual(
                    evaluate.call_args.kwargs["reward_function_version"], reward_version
                )
                evaluate.assert_called_once()

    def test_goal_reconstruction_forwards_version(self):
        task = MagicMock()
        task.data = b"task"
        bank = dict(
            task_name="reach-v3",
            bank_seed=0,
            metaworld_version=version("metaworld"),
            ordered_task_hashes=bank_hashes([task]),
        )
        for reward_version in ("v1", "v2"):
            env = MagicMock()
            env.reset.return_value = (np.zeros(39, dtype=np.float32), {})
            env.unwrapped._target_pos = np.zeros(3)
            with patch("src.continual.task_bank._backend_tasks") as backend_tasks, patch(
                "src.continual.task_bank.make_metaworld_env", return_value=env
            ) as make:
                backend_tasks.return_value = [task]
                goals = reconstruct_goals(
                    bank, [0], reward_function_version=reward_version
                )
                make.assert_called_once_with(
                    "reach-v3", 0, reward_function_version=reward_version
                )
                np.testing.assert_array_equal(goals, np.zeros((1, 3), dtype=np.float32))
                env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
