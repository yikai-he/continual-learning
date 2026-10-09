"""KUKA-v3 deterministic fixed-task continual-evaluation contracts."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import load_config
from src.continual.evaluation import evaluate_task, fixed_mt1_tasks
from src.continual.task_bank import (
    bank_hashes,
    reconstruct_goals,
    training_bank,
    verify_disjoint_task_banks,
)
from src.continual.task_sequence import TaskSequence
from src.envs import make_env


class _ZeroPolicy:
    def act(self, _observation, deterministic=True):
        return np.zeros(4, dtype=np.float32)


def _config(path):
    return load_config(f"configs/diffcrl/{path}")


def test_kuka_fixed_configs_and_method_parity() -> None:
    replay = _config("kuka_diffcrl_smoke.yaml")
    control = _config("kuka_diffcrl_no_replay_smoke.yaml")
    hammer = _config("kuka_diffcrl_hammer_smoke.yaml")
    hammer_control = _config("kuka_diffcrl_no_replay_hammer_smoke.yaml")
    for config in (replay, control, hammer, hammer_control):
        assert config.evaluation.mode == "fixed-tasks"
        assert config.evaluation.task_set_seed == 10_000
        assert config.evaluation.episodes == 2
    assert replay.evaluation == control.evaluation
    assert hammer.evaluation == hammer_control.evaluation
    assert hammer.continual.tasks[-1] == "kuka-hammer-v3"
    assert hammer.environment.hammer_reward_variant == "nail_progress"
    assert hammer.environment.hammer_nail_progress_weight == 4.0


def test_kuka_formal_configs_match_except_replay_and_output() -> None:
    replay = _config("kuka_diffcrl_5task.yaml")
    control = _config("kuka_diffcrl_no_replay_5task.yaml")
    expected = [
        "kuka-reach-v3",
        "kuka-handle-press-side-v3",
        "kuka-drawer-close-v3",
        "kuka-faucet-close-v3",
        "kuka-hammer-v3",
    ]
    assert replay.continual.tasks == control.continual.tasks == expected
    assert replay.continual.experts == control.continual.experts
    assert replay.continual.trajectories_per_task == 200
    assert control.continual.trajectories_per_task == 200
    assert replay.bc == control.bc
    assert replay.environment == control.environment
    assert replay.evaluation == control.evaluation
    assert replay.runtime.seed == control.runtime.seed == 0
    assert replay.continual.replay_mode == "diffusion"
    assert control.continual.replay_mode == "none"


def test_kuka_bank_hashes_repeat_across_methods_and_stages() -> None:
    first, identities = fixed_mt1_tasks(
        "kuka-reach-v3", 10_000, backend="kuka-v3"
    )
    second, repeated = fixed_mt1_tasks(
        "kuka-reach-v3", 10_000, backend="kuka-v3"
    )
    assert bank_hashes(first) == bank_hashes(second)
    assert identities == repeated
    assert identities[0]["task_hash"] != identities[1]["task_hash"]


def test_kuka_training_and_evaluation_banks_are_hash_disjoint() -> None:
    check = verify_disjoint_task_banks(
        "kuka-reach-v3", 0, 10_000, backend="kuka-v3"
    )
    assert check["backend"] == "kuka-v3"
    assert check["disjoint"]
    assert len(check["ordered_training_task_hashes"]) == 50
    assert len(check["ordered_evaluation_task_hashes"]) == 50
    assert not set(check["ordered_training_task_hashes"]) & set(
        check["ordered_evaluation_task_hashes"]
    )


@pytest.mark.parametrize(
    "name", ("kuka-reach-v3", "kuka-handle-press-side-v3")
)
def test_kuka_fixed_evaluation_uses_ordered_identity(name) -> None:
    task = TaskSequence.from_names([name]).task(0)
    result = evaluate_task(
        _ZeroPolicy(),
        task,
        episodes=1,
        seed=20_000,
        evaluation_mode="fixed-tasks",
        task_set_seed=10_000,
        backend="kuka-v3",
    )
    tasks, identities = fixed_mt1_tasks(name, 10_000, backend="kuka-v3")
    assert result["ordered_task_hashes"] == bank_hashes(tasks)
    assert result["episode_provenance"][0]["task_hash"] == identities[0]["task_hash"]
    assert result["episode_provenance"][0]["reset_seed"] == 20_000


def test_hammer_fixed_evaluation_forwards_environment_settings() -> None:
    task = TaskSequence.from_names(["kuka-hammer-v3"]).task(0)
    env = make_env(
        "kuka-v3",
        "kuka-hammer-v3",
        20_000,
        reward_function_version="v2",
        hammer_reward_variant="nail_progress",
        hammer_nail_progress_weight=4.0,
    )
    switcher = MagicMock()
    switcher.__enter__.return_value = switcher
    switcher.switch.return_value = env
    with patch(
        "src.continual.evaluation.TaskSwitcher", return_value=switcher
    ) as switcher_type:
        evaluate_task(
            _ZeroPolicy(),
            task,
            episodes=1,
            seed=20_000,
            evaluation_mode="fixed-tasks",
            task_set_seed=10_000,
            backend="kuka-v3",
            reward_function_version="v2",
            hammer_reward_variant="nail_progress",
            hammer_nail_progress_weight=4.0,
        )
    env.close()
    assert env.unwrapped.hammer_reward_variant == "nail_progress"
    assert env.unwrapped.hammer_nail_progress_weight == 4.0
    switcher_type.assert_called_once_with(
        TaskSequence((task,)),
        backend="kuka-v3",
        reward_function_version="v2",
        hammer_reward_variant="nail_progress",
        hammer_nail_progress_weight=4.0,
        render_mode=None,
    )


def test_replay_reconstruction_remains_on_training_bank() -> None:
    training = training_bank("kuka-reach-v3", 0, backend="kuka-v3")
    evaluation_tasks, _ = fixed_mt1_tasks(
        "kuka-reach-v3", 10_000, backend="kuka-v3"
    )
    assert not set(training["ordered_task_hashes"]) & set(
        bank_hashes(evaluation_tasks)
    )
    goals = reconstruct_goals(training, [0], backend="kuka-v3")
    assert goals.shape == (1, 3)


@pytest.mark.parametrize(
    "name",
    (
        "kuka-button-press-v3",
        "kuka-drawer-close-v3",
        "kuka-window-open-v3",
        "kuka-window-close-v3",
        "kuka-faucet-open-v3",
        "kuka-door-open-v3",
        "kuka-faucet-close-v3",
        "kuka-reach-v3",
        "kuka-handle-press-side-v3",
        "kuka-hammer-v3",
    ),
)
def test_every_kuka_task_bank_target_is_representable_without_clipping(name) -> None:
    """The public observation contract must represent every canonical target."""
    from src.continual.evaluation import disable_task_sampling

    tasks, _ = fixed_mt1_tasks(name, 0, backend="kuka-v3")
    env = make_env(
        "kuka-v3",
        name,
        0,
        reward_function_version="v2",
        hammer_reward_variant="nail_progress",
        hammer_nail_progress_weight=4.0,
    )
    try:
        disable_task_sampling(env)
        goal_low = env.observation_space.low[36:39]
        goal_high = env.observation_space.high[36:39]
        assert env.observation_space.shape == (39,)
        assert env.action_space.shape == (4,)
        for index, task in enumerate(tasks):
            env.unwrapped.set_task(task)
            observation, _ = env.reset(seed=index)
            target = np.asarray(env.unwrapped._target_pos, dtype=np.float32)
            assert np.all(target >= goal_low)
            assert np.all(target <= goal_high)
            np.testing.assert_array_equal(observation[36:39], target)
            stepped, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
            np.testing.assert_array_equal(stepped[36:39], target)
    finally:
        env.close()


@pytest.mark.parametrize(
    "name",
    (
        "kuka-reach-v3",
        "kuka-handle-press-side-v3",
        "kuka-drawer-close-v3",
        "kuka-faucet-close-v3",
        "kuka-hammer-v3",
    ),
)
def test_formal_kuka_task_goal_is_exactly_constant_for_full_episode(name) -> None:
    tasks, _ = fixed_mt1_tasks(name, 0, backend="kuka-v3")
    env = make_env(
        "kuka-v3",
        name,
        0,
        reward_function_version="v2",
        hammer_reward_variant="nail_progress",
        hammer_nail_progress_weight=4.0,
    )
    try:
        from src.continual.evaluation import disable_task_sampling, validate_active_task

        disable_task_sampling(env)
        env.unwrapped.set_task(tasks[2])
        observation, _ = env.reset(seed=0)
        target = np.asarray(env.unwrapped._target_pos, dtype=np.float32).copy()
        goals = []
        for step in range(200):
            goals.append(np.asarray(observation[36:39]).copy())
            observation, _, terminated, truncated, _ = env.step(
                np.zeros(4, dtype=np.float32)
            )
            assert not terminated
            assert truncated == (step == 199)
        validate_active_task(env, tasks[2], "kuka-v3")
    finally:
        env.close()
    goals = np.asarray(goals)
    np.testing.assert_array_equal(goals, np.broadcast_to(goals[0], goals.shape))
    np.testing.assert_array_equal(goals, np.broadcast_to(target, goals.shape))
