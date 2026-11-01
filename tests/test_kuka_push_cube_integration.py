"""Training-adapter and renderer integration for KUKA Cube Push."""

from pathlib import Path
import importlib.util

import numpy as np

from src.envs import make_env


def test_cube_push_training_adapter_contract() -> None:
    env = make_env(
        "kuka-v3",
        "kuka-push-cube-v3",
        10_000,
        reward_function_version="v3_1",
    )
    try:
        observation, _ = env.reset(seed=10_019)
        assert observation.shape == (39,)
        assert observation.dtype == np.float32
        assert env.action_space.shape == (4,)
        assert env.spec.max_episode_steps == 200
    finally:
        env.close()


def test_renderer_discovers_cube_task_without_a_special_case() -> None:
    script = Path("scripts/render_kuka_push_policy.py")
    spec = importlib.util.spec_from_file_location("cube_renderer", script)
    assert spec is not None and spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    args = renderer.parse_args(
        ["--env-name", "kuka-push-cube-v3", "--reward-version", "v3_1"]
    )
    assert args.env_name == "kuka-push-cube-v3"
    assert args.reward_version == "v3_1"
