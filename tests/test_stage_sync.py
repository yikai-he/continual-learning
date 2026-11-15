"""Filesystem-only tests for verified stage synchronization and restoration."""

import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.continual.diffcrl_artifacts import marker_path, write_stage_marker
from src.support.reproducibility import file_hash
from src.support.stage_sync import (
    backup_status,
    restore_backup,
    sync_all_completed,
    sync_completed_stage,
)

TASKS = ["reach-v3", "push-v3", "hammer-v3"]


def make_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text("experiment: diffcrl\n", encoding="utf-8")
    (run / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    return run


def add_boundary(run, stage):
    task = TASKS[stage]
    directory = run / f"stage_{stage}_{task}"
    directory.mkdir()
    policy = run / f"general_policy_after_task_{stage}_{task}.pt"
    policy.write_bytes(f"policy {stage}".encode())
    report = directory / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    matrix = directory / "evaluation_matrix.json"
    matrix.write_text("{}\n", encoding="utf-8")
    return write_stage_marker(
        run,
        stage=stage,
        task=task,
        policy_path=policy,
        diffusion_path=None,
        report_path=report,
        matrix_path=matrix,
    )


def _complete_source(tmp_path, count=1):
    run = make_run(tmp_path)
    (run / "config.json").write_text("{}\n", encoding="utf-8")
    (run / "RUNNING").write_text("running\n", encoding="utf-8")
    for stage in range(count):
        add_boundary(run, stage)
    return run


def test_success_duplicate_and_multiple_stage_sync(tmp_path):
    source = _complete_source(tmp_path, 3)
    backup = tmp_path / "drive"
    assert sync_all_completed(source, backup, TASKS) == 2
    assert sync_all_completed(source, backup, TASKS) == 2
    status = backup_status(backup, TASKS)
    assert status["completed_stages"] == [0, 1, 2]
    for stage in range(3):
        assert marker_path(backup, stage).is_file()


def test_marker_is_published_last(tmp_path):
    source = _complete_source(tmp_path)
    backup = tmp_path / "drive"
    copied = []
    from src.support import stage_sync

    real_copy = stage_sync._copy_verified

    def recording_copy(source_path, destination_path):
        copied.append(Path(destination_path).name)
        return real_copy(source_path, destination_path)

    with patch.object(stage_sync, "_copy_verified", side_effect=recording_copy):
        sync_completed_stage(source, backup, TASKS, 0)
    assert copied[-1] == "stage_0_COMPLETE.json"


def test_missing_or_corrupt_source_is_rejected_without_remote_marker(tmp_path):
    source = _complete_source(tmp_path)
    backup = tmp_path / "drive"
    (source / "general_policy_after_task_0_reach-v3.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        sync_completed_stage(source, backup, TASKS, 0)
    assert not marker_path(backup, 0).exists()


def test_interrupted_copy_can_be_retried(tmp_path):
    source = _complete_source(tmp_path)
    backup = tmp_path / "drive"
    from src.support import stage_sync

    real_copy = stage_sync._copy_verified
    calls = 0

    def fail_once(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise OSError("simulated Drive disconnect")
        return real_copy(source_path, destination_path)

    with patch.object(stage_sync, "_copy_verified", side_effect=fail_once):
        with pytest.raises(OSError, match="disconnect"):
            sync_completed_stage(source, backup, TASKS, 0)
    assert not marker_path(backup, 0).exists()
    sync_completed_stage(source, backup, TASKS, 0)
    assert marker_path(backup, 0).is_file()


def test_conflicting_destination_is_not_overwritten(tmp_path):
    source = _complete_source(tmp_path)
    backup = tmp_path / "drive"
    backup.mkdir()
    (backup / "config.yaml").write_text("different\n", encoding="utf-8")
    before = (backup / "config.yaml").read_bytes()
    with pytest.raises(FileExistsError, match="Conflicting"):
        sync_completed_stage(source, backup, TASKS, 0)
    assert (backup / "config.yaml").read_bytes() == before
    assert not marker_path(backup, 0).exists()


def test_corrupt_remote_and_missing_marker_are_not_valid(tmp_path):
    source = _complete_source(tmp_path)
    backup = tmp_path / "drive"
    sync_completed_stage(source, backup, TASKS, 0)
    marker_path(backup, 0).unlink()
    assert backup_status(backup, TASKS)["completed_stages"] == []
    sync_completed_stage(source, backup, TASKS, 0)
    (backup / "general_policy_after_task_0_reach-v3.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupted"):
        backup_status(backup, TASKS)


def test_restore_fresh_directory_and_relocation(tmp_path):
    source = _complete_source(tmp_path, 2)
    backup = tmp_path / "drive"
    sync_all_completed(source, backup, TASKS)
    restored = tmp_path / "new-content" / "run"
    assert restore_backup(backup, restored, TASKS) == 1
    assert backup_status(restored, TASKS)["completed_stages"] == [0, 1]
    for stage in range(2):
        assert file_hash(marker_path(restored, stage)) == file_hash(
            marker_path(backup, stage)
        )


def test_restore_rejects_nonempty_destination_and_corrupt_backup(tmp_path):
    source = _complete_source(tmp_path)
    backup = tmp_path / "drive"
    sync_completed_stage(source, backup, TASKS, 0)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("user data")
    with pytest.raises(FileExistsError, match="not empty"):
        restore_backup(backup, occupied, TASKS)
    (backup / "stage_0_reach-v3" / "report.json").write_text("changed")
    with pytest.raises(ValueError, match="corrupted"):
        restore_backup(backup, tmp_path / "fresh", TASKS)


def test_sync_does_not_consume_rng(tmp_path):
    source = _complete_source(tmp_path)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    sync_completed_stage(source, tmp_path / "drive", TASKS, 0)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def test_completed_local_run_preserves_completion_lifecycle(tmp_path):
    source = _complete_source(tmp_path)
    (source / "RUNNING").rename(source / "RUN_COMPLETE")
    backup = tmp_path / "drive"
    sync_completed_stage(source, backup, TASKS, 0)
    assert (backup / "RUN_COMPLETE").is_file()
    assert not (backup / "RUNNING").exists()
    assert restore_backup(backup, tmp_path / "restored", TASKS) == 0
