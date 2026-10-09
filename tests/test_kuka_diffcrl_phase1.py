"""Phase-1 KUKA DiffCRL configuration, adapter, and provenance contracts."""

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
import torch
from gymnasium import Env
from gymnasium.spaces import Box

from src.config import load_config
from src.continual.bc_policy import GeneralPolicy
from src.continual.collector import collect_trajectory
from src.continual.task_bank import (
    _task_rand_vec,
    _task_wrapper,
    bank_hashes,
    reconstruct_goals,
    selected_configuration,
    training_bank,
)
from src.continual.diffusion_data import (
    decode_trajectory,
    encode_trajectory,
    pack_trajectories,
    unpack_trajectories,
)
from src.envs import make_env, target_position
from src.env_adapters.kuka_v3 import make_kuka_v3_env
from src.support.expert_manifest import validate_expert_checkpoint


def _write_expert(tmp_path: Path, *, task: str, backend: str, **extra) -> Path:
    checkpoint = tmp_path / f"{task}.zip"
    checkpoint.write_bytes(task.encode())
    manifest = {
        "backend": backend,
        "task_name": task,
        "algorithm": "SAC",
        "observation_shape": [39],
        "action_shape": [4],
        "horizon": 200,
        "reward_function_version": "v2",
        "training_seed": 0,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        **extra,
    }
    checkpoint.with_suffix(".zip.manifest.json").write_text(json.dumps(manifest))
    return checkpoint


def _validate(checkpoint: Path, *, task="kuka-reach-v3", backend="kuka-v3", **kw):
    return validate_expert_checkpoint(
        checkpoint,
        expected_task=task,
        expected_backend=backend,
        expected_reward_function_version="v2",
        expected_observation_shape=(39,),
        expected_action_shape=(4,),
        expected_horizon=200,
        allow_legacy=False,
        **kw,
    )


def test_kuka_smoke_configs_parse() -> None:
    replay = load_config("configs/diffcrl/kuka_diffcrl_smoke.yaml")
    no_replay = load_config("configs/diffcrl/kuka_diffcrl_no_replay_smoke.yaml")
    assert replay.continual.replay_mode == "diffusion"
    assert no_replay.continual.replay_mode == "none"
    assert replay.environment.hammer_reward_variant == "nail_progress"
    assert replay.continual.tasks == [
        "kuka-reach-v3",
        "kuka-handle-press-side-v3",
    ]


class _FactoryEnv(Env):
    observation_space = Box(-np.inf, np.inf, shape=(39,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        observation = np.zeros(39, dtype=np.float32)
        observation[36:39] = 1.0
        return observation, {}


def test_kuka_adapter_scopes_hammer_settings(monkeypatch) -> None:
    calls = []

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return _FactoryEnv()

    monkeypatch.setattr("src.env_adapters.kuka_v3.gym.make", factory)
    reach = make_kuka_v3_env(
        "kuka-reach-v3",
        0,
        hammer_reward_variant="nail_progress",
        hammer_nail_progress_weight=4.0,
    )
    hammer = make_kuka_v3_env(
        "kuka-hammer-v3",
        0,
        hammer_reward_variant="nail_progress",
        hammer_nail_progress_weight=4.0,
    )
    reach.close()
    hammer.close()
    assert calls[0]["hammer_reward_variant"] == "original"
    assert calls[0]["hammer_nail_progress_weight"] == 0.0
    assert calls[1]["hammer_reward_variant"] == "nail_progress"
    assert calls[1]["hammer_nail_progress_weight"] == 4.0


def test_selected_configuration_works_for_metaworld_v3() -> None:
    bank = training_bank("reach-v3", 17, backend="metaworld-v3")
    env = make_env("metaworld-v3", "reach-v3", 17)
    try:
        observation, _ = env.reset(seed=17)
        selected = selected_configuration(env, bank, observation, 0)
        assert selected["env_name"] == "reach-v3"
        np.testing.assert_allclose(
            observation[36:39], target_position(env, "metaworld-v3"), rtol=1e-6
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    "task", ["kuka-reach-v3", "kuka-handle-press-side-v3"]
)
def test_kuka_configuration_capture_and_reconstruction(task: str) -> None:
    bank = training_bank(task, 23, backend="kuka-v3")
    env = make_env("kuka-v3", task, 23)
    try:
        assert not hasattr(env.unwrapped, "raw_env")
        observation, _ = env.reset(seed=23)
        selected = selected_configuration(env, bank, observation, 0)
        index = selected["configuration_index"]
        assert selected["task_bank_index"] == index
        assert selected["task_hash"] == bank["ordered_task_hashes"][index]
        np.testing.assert_array_equal(
            selected["last_rand_vec"], _task_rand_vec(_task_wrapper(env).tasks[index])
        )
        np.testing.assert_allclose(
            observation[36:39], target_position(env, "kuka-v3"), rtol=1e-6
        )
        np.testing.assert_allclose(bank["goals"][0], observation[36:39], rtol=1e-6)
    finally:
        env.close()

    reconstructed = reconstruct_goals(bank, [index], backend="kuka-v3")
    np.testing.assert_allclose(reconstructed[0], observation[36:39], rtol=1e-6)


def test_explicit_kuka_task_survives_reset_seed_and_cannot_be_resampled() -> None:
    env = make_env("kuka-v3", "kuka-reach-v3", 31)
    try:
        wrapper = _task_wrapper(env)
        selected = wrapper.tasks[4]
        selected_hash = bank_hashes([selected])[0]
        wrapper.toggle_sample_tasks_on_reset(False)
        env.unwrapped.set_task(selected)
        first, _ = env.reset(seed=1)
        second, _ = env.reset(seed=9999)
        assert bank_hashes([selected])[0] == selected_hash
        np.testing.assert_array_equal(env.unwrapped._last_rand_vec, _task_rand_vec(selected))
        np.testing.assert_allclose(first[36:39], second[36:39], rtol=1e-6)
    finally:
        env.close()


def test_multiple_kuka_bank_entries_reconstruct_distinct_identities() -> None:
    bank = training_bank("kuka-reach-v3", 37, backend="kuka-v3")
    goals = reconstruct_goals(bank, [0, 1], backend="kuka-v3")
    assert bank["ordered_task_hashes"][0] != bank["ordered_task_hashes"][1]
    assert not np.array_equal(goals[0], goals[1])


def test_incorrect_kuka_task_identity_is_rejected() -> None:
    bank = training_bank("kuka-reach-v3", 41, backend="kuka-v3")
    bank["ordered_task_hashes"][0] = "0" * 64
    with pytest.raises(ValueError, match="task-bank hash order"):
        reconstruct_goals(bank, [0], backend="kuka-v3")


def test_valid_kuka_expert_is_accepted(tmp_path: Path) -> None:
    assert _validate(_write_expert(tmp_path, task="kuka-reach-v3", backend="kuka-v3")).backend == "kuka-v3"


def test_sawyer_expert_is_rejected_by_kuka(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend"):
        _validate(_write_expert(tmp_path, task="kuka-reach-v3", backend="metaworld-v3"))


def test_kuka_expert_is_rejected_by_sawyer(tmp_path: Path) -> None:
    checkpoint = _write_expert(tmp_path, task="reach-v3", backend="kuka-v3")
    with pytest.raises(ValueError, match="backend"):
        _validate(checkpoint, task="reach-v3", backend="metaworld-v3")


def test_kuka_checkpoint_without_manifest_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "expert.zip"
    checkpoint.write_bytes(b"same dimensions are not provenance")
    with pytest.raises(FileNotFoundError, match="manifest is required"):
        _validate(checkpoint)


def test_matching_shapes_do_not_override_backend_mismatch(tmp_path: Path) -> None:
    checkpoint = _write_expert(tmp_path, task="kuka-reach-v3", backend="metaworld-v3")
    with pytest.raises(ValueError, match="backend"):
        _validate(checkpoint)


@pytest.mark.parametrize(
    "variant,weight,match",
    [("original", 4.0, "variant"), ("nail_progress", 2.0, "weight")],
)
def test_hammer_manifest_settings_must_match(tmp_path, variant, weight, match) -> None:
    checkpoint = _write_expert(
        tmp_path,
        task="kuka-hammer-v3",
        backend="kuka-v3",
        hammer_reward_variant=variant,
        hammer_nail_progress_weight=weight,
    )
    with pytest.raises(ValueError, match=match):
        _validate(
            checkpoint,
            task="kuka-hammer-v3",
            expected_hammer_reward_variant="nail_progress",
            expected_hammer_nail_progress_weight=4.0,
        )


def test_canonical_kuka_manifests_match_checkpoints() -> None:
    root = Path("runs/experts/kuka-v3")
    for checkpoint in root.glob("*/expert_model.zip"):
        data = json.loads((checkpoint.parent / "expert_manifest.json").read_text())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == data["checkpoint_sha256"]
        assert data["backend"] == "kuka-v3"
        assert data["horizon"] == 200


class _ZeroPolicy:
    def act(self, _observation, deterministic=True):
        return np.zeros(4, dtype=np.float32)


def test_collected_kuka_reach_trajectory_matches_diffusion_and_bc_contracts() -> None:
    env = make_env("kuka-v3", "kuka-reach-v3", seed=7)
    try:
        trajectory = collect_trajectory(
            env, _ZeroPolicy(), "kuka-reach-v3", seed=7
        )
        assert env.observation_space.shape == (39,)
        assert env.action_space.shape == (4,)
        assert env.spec.max_episode_steps == trajectory.length == 200
        assert trajectory.observations.dtype == np.float32
        assert np.isfinite(trajectory.observations).all()
        np.testing.assert_array_equal(
            trajectory.observations[0, 18:36], trajectory.observations[0, :18]
        )
        np.testing.assert_array_equal(
            trajectory.observations[1:, 18:36], trajectory.observations[:-1, :18]
        )
        np.testing.assert_array_equal(
            trajectory.observations[:, 36:39],
            np.repeat(trajectory.observations[:1, 36:39], 200, axis=0),
        )

        packed = pack_trajectories(
            [trajectory], horizon=200, task_names=("kuka-reach-v3",)
        )
        observations, actions = unpack_trajectories(packed)
        dynamic, goals = encode_trajectory(observations, actions)
        assert torch.equal(decode_trajectory(dynamic, goals), packed)

        policy = GeneralPolicy(env.action_space.low, env.action_space.high)
        assert policy(observations.reshape(-1, 39)).shape == (200, 4)
    finally:
        env.close()
