from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box

from src.envs import EpisodeSuccessWrapper, make_env
from src.env_adapters.kuka_v2 import (
    KUKA_HORIZON,
    KUKA_TASKS,
    KukaV2GymnasiumAdapter,
)


EXPECTED_KUKA_TASKS = (
    "kuka-reach-v2",
    "kuka-push-v2",
    "kuka-hammer-v2",
    "kuka-handle-press-side-v2",
    "kuka-button-press-v2",
)


def test_all_target_tasks_are_exposed_in_order() -> None:
    assert KUKA_TASKS == EXPECTED_KUKA_TASKS


class FakeLegacyKukaEnv:
    def __init__(self) -> None:
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.np_random = np.random.RandomState()
        self.step_calls = 0
        self.target = np.zeros(3, dtype=np.float32)

    def seed(self, seed: int) -> None:
        self.np_random = np.random.RandomState(seed)

    def reset(self) -> np.ndarray:
        self.step_calls = 0
        self.target = self.np_random.uniform(0.1, 0.9, size=3).astype(np.float32)
        observation = np.zeros(39, dtype=np.float32)
        observation[36:39] = self.target
        return observation

    def step(self, action: np.ndarray):
        self.step_calls += 1
        observation = np.zeros(39, dtype=np.float32)
        observation[36:39] = self.target
        return observation, 1.0, False, {"success": float(self.step_calls == 2)}

    def close(self) -> None:
        pass


def test_adapter_api_horizon_goal_and_success() -> None:
    raw_env = FakeLegacyKukaEnv()
    env = EpisodeSuccessWrapper(KukaV2GymnasiumAdapter(raw_env))
    observation, info = env.reset(seed=7)
    assert info == {}
    assert observation.shape == (39,)
    assert observation.dtype == np.float32
    assert env.action_space.shape == (4,)
    assert env.spec.max_episode_steps == KUKA_HORIZON
    assert np.any(observation[36:39])

    for step in range(1, KUKA_HORIZON + 1):
        result = env.step(np.zeros(4, dtype=np.float32))
        assert len(result) == 5
        _, _, terminated, truncated, step_info = result
        assert terminated is False
        assert truncated is (step == KUKA_HORIZON)
        if step == 2:
            assert step_info["success"] == 1.0
        if step == KUKA_HORIZON:
            assert step_info["is_success"] is True

    assert raw_env.step_calls == KUKA_HORIZON
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4, dtype=np.float32))
    assert raw_env.step_calls == KUKA_HORIZON


def test_adapter_seed_reproduces_sequence_without_freezing_targets() -> None:
    first = KukaV2GymnasiumAdapter(FakeLegacyKukaEnv())
    second = KukaV2GymnasiumAdapter(FakeLegacyKukaEnv())

    first_sequence = [first.reset(seed=11)[0][36:39]]
    second_sequence = [second.reset(seed=11)[0][36:39]]
    for _ in range(3):
        first_sequence.append(first.reset()[0][36:39])
        second_sequence.append(second.reset()[0][36:39])

    np.testing.assert_allclose(first_sequence, second_sequence)
    assert np.unique(np.asarray(first_sequence), axis=0).shape[0] > 1


@pytest.mark.parametrize("task", KUKA_TASKS)
def test_kuka_factory_constructs_and_configures_raw_environment(task: str) -> None:
    try:
        from metaworld.envs.mujoco.env_dict import KUKA_V2_ENVIRONMENTS
    except ImportError:
        pytest.skip("requires the legacy KUKA MetaWorld checkout")
    if task not in KUKA_V2_ENVIRONMENTS:
        pytest.skip(f"requires registered legacy KUKA task {task}")

    env = make_env("kuka-v2", task, seed=13)
    try:
        observation, info = env.reset(seed=13)
        raw_env = env.unwrapped.raw_env
        assert observation.shape == (39,)
        assert observation.dtype == np.float32
        assert info == {}
        assert np.any(observation[36:39])
        assert raw_env._set_task_called is True
        assert raw_env._freeze_rand_vec is False
        assert raw_env.seeded_rand_vec is True
        assert raw_env._partially_observable is False

        step_result = env.step(np.zeros(4, dtype=np.float32))
        assert len(step_result) == 5
        assert step_result[2] is False
        assert "success" in step_result[4]

        comparison_slice = slice(4, 7) if task == "kuka-hammer-v2" else slice(36, 39)
        first = observation[comparison_slice].copy()
        second = env.reset()[0][comparison_slice].copy()
        first_repeated = env.reset(seed=13)[0][comparison_slice].copy()
        second_repeated = env.reset()[0][comparison_slice].copy()
        np.testing.assert_allclose(first_repeated, first)
        np.testing.assert_allclose(second_repeated, second)
        assert not np.allclose(first, second)
    finally:
        env.close()
