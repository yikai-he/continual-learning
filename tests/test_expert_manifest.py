import hashlib
import json
import warnings
from pathlib import Path

import pytest

from src.support.expert_manifest import validate_expert_checkpoint


def _manifest(checkpoint: Path, **updates) -> Path:
    value = {
        "task_name": "reach-v3",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "backend": "metaworld-v3",
        "algorithm": "SAC",
        "reward_function_version": "v2",
        "training_seed": 0,
        "observation_shape": [39],
        "action_shape": [4],
        "qualification": None,
    }
    value.update(updates)
    path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _validate(checkpoint: Path, **kwargs):
    return validate_expert_checkpoint(
        checkpoint,
        expected_task="reach-v3",
        expected_backend="metaworld-v3",
        expected_reward_function_version="v2",
        expected_observation_shape=(39,),
        expected_action_shape=(4,),
        **kwargs,
    )


def test_correct_task_hash_and_shapes_pass(tmp_path):
    checkpoint = tmp_path / "expert.zip"
    checkpoint.write_bytes(b"checkpoint")
    _manifest(checkpoint)
    identity = _validate(checkpoint)
    assert identity.task_name == "reach-v3"
    assert not identity.legacy


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"task_name": "push-v3"}, "task name"),
        ({"checkpoint_sha256": "0" * 64}, "checkpoint SHA-256"),
        ({"observation_shape": [18]}, "observation shape"),
    ],
)
def test_identity_mismatches_fail(tmp_path, updates, match):
    checkpoint = tmp_path / "expert.zip"
    checkpoint.write_bytes(b"checkpoint")
    _manifest(checkpoint, **updates)
    with pytest.raises(ValueError, match=match):
        _validate(checkpoint)


def test_legacy_behavior_is_explicit(tmp_path):
    checkpoint = tmp_path / "legacy.zip"
    checkpoint.write_bytes(b"checkpoint")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        identity = _validate(checkpoint, allow_legacy=True)
    assert identity.legacy
    assert "LEGACY EXPERT" in str(caught[0].message)
    with pytest.raises(FileNotFoundError, match="manifest is required"):
        _validate(checkpoint, allow_legacy=False)
