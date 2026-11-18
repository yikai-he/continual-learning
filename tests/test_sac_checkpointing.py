"""Focused contracts for synchronized SAC model/replay checkpoints."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import train_sac
from src.config import default_config
from src.support.reproducibility import file_hash


class Buffer:
    buffer_size = 20
    observation_space = SimpleNamespace(shape=(39,))
    action_space = SimpleNamespace(shape=(4,))

    def size(self):
        return 7


def model(step=123):
    value = MagicMock()
    value.num_timesteps = step
    value._n_updates = 99
    value.observation_space = SimpleNamespace(shape=(39,))
    value.action_space = SimpleNamespace(shape=(4,))
    value.replay_buffer = Buffer()

    def save(path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", "model")

    value.save.side_effect = save
    value.save_replay_buffer.side_effect = lambda path: Path(path).write_bytes(b"replay")
    return value


def context():
    return {
        "task_name": "reach-v3",
        "backend": "metaworld-v3",
        "reward_function_version": "v2",
        "training_config_sha256": "config-hash",
    }


def test_best_model_replay_and_metadata_share_one_step(tmp_path):
    saved = train_sac.save_sac_checkpoint(model(800_000), tmp_path, "best", context())
    assert saved["checkpoint_timestep"] == 800_000
    assert saved["replay_buffer"] == {
        "class": f"{Buffer.__module__}.Buffer",
        "capacity": 20,
        "stored_transitions": 7,
    }
    assert saved["model"]["sha256"] == file_hash(tmp_path / "best_model.zip")
    assert saved["replay"]["sha256"] == file_hash(tmp_path / "best_replay_buffer.pkl")
    assert json.loads((tmp_path / "best_checkpoint.json").read_text()) == saved


def test_failed_best_replacement_restores_previous_complete_pair(tmp_path):
    train_sac.save_sac_checkpoint(model(10), tmp_path, "best", context())
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    real_replace = train_sac.os.replace

    def fail_on_new_replay(source, target):
        if Path(target).name == "best_replay_buffer.pkl" and ".tmp" in Path(source).name:
            raise OSError("disk failure")
        return real_replace(source, target)

    with patch.object(train_sac.os, "replace", side_effect=fail_on_new_replay):
        with pytest.raises(OSError, match="disk failure"):
            train_sac.save_sac_checkpoint(model(20), tmp_path, "best", context())
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before


def write_pair(tmp_path, *, step=50, task="reach-v3", reward="v2"):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment: sac\n")
    ctx = context()
    ctx["task_name"] = task
    ctx["reward_function_version"] = reward
    ctx["training_config_sha256"] = file_hash(config_path)
    train_sac.save_sac_checkpoint(model(step), tmp_path, "best", ctx)
    return config_path


def test_matching_model_and_replay_pair_loads(tmp_path):
    write_pair(tmp_path)
    loaded = model(50)
    loaded.load_replay_buffer.side_effect = lambda _path: setattr(loaded, "replay_buffer", Buffer())
    metadata = train_sac.validate_replay_checkpoint_pair(
        tmp_path / "best_model.zip",
        tmp_path / "best_replay_buffer.pkl",
        tmp_path / "best_checkpoint.json",
        default_config("sac"),
        loaded,
    )
    assert metadata["checkpoint_timestep"] == 50
    loaded.load_replay_buffer.assert_called_once()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_name", "push-v3", "task identity"),
        ("reward_function_version", "v1", "reward version"),
        ("observation_shape", [38], "observation shape"),
        ("action_shape", [3], "action shape"),
        ("checkpoint_timestep", 49, "checkpoint timestep"),
    ],
)
def test_pair_identity_mismatches_fail(tmp_path, field, value, message):
    write_pair(tmp_path)
    metadata_path = tmp_path / "best_checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        train_sac.validate_replay_checkpoint_pair(
            tmp_path / "best_model.zip",
            tmp_path / "best_replay_buffer.pkl",
            metadata_path,
            default_config("sac"),
            model(50),
        )


def test_pair_hash_mismatch_fails_before_loading(tmp_path):
    write_pair(tmp_path)
    (tmp_path / "best_replay_buffer.pkl").write_bytes(b"corrupt")
    loaded = model(50)
    with pytest.raises(ValueError, match="replay SHA-256"):
        train_sac.validate_replay_checkpoint_pair(
            tmp_path / "best_model.zip",
            tmp_path / "best_replay_buffer.pkl",
            tmp_path / "best_checkpoint.json",
            default_config("sac"),
            loaded,
        )
    loaded.load_replay_buffer.assert_not_called()


def test_best_pair_is_saved_before_early_stopping_decision(tmp_path):
    callback = train_sac.FixedSeedEvalCallback(
        MagicMock(), eval_freq=1, n_eval_episodes=1, seed=0,
        log_path=tmp_path, best_success_model_save_path=tmp_path,
    )
    callback.model = MagicMock()
    callback._logger = MagicMock()
    callback._evaluate = MagicMock(return_value=([1.0], [200], [True]))
    callback._save_evaluations = MagicMock()
    order = []
    callback._save_model = MagicMock(side_effect=lambda _path: order.append("save"))
    callback._update_early_stopping = MagicMock(
        side_effect=lambda _rate: order.append("stop") or False
    )
    assert not callback._record_evaluation()
    assert order == ["save", "stop"]
