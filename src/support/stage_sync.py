"""Verified, retryable synchronization of completed DiffCRL stages."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.continual.resume import completed_stage_markers, marker_path
from src.support.reproducibility import file_hash

REQUIRED_RUN_METADATA = ("config.yaml", "config.json", "run_manifest.json")


def _copy_verified(source: Path, destination: Path) -> None:
    """Copy one immutable file, reusing identical destinations."""
    digest = file_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or file_hash(destination) != digest:
            raise FileExistsError(f"Conflicting synchronized file: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.syncing")
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if file_hash(temporary) != digest:
            raise OSError(f"Copied file failed checksum validation: {destination}")
        if destination.exists():
            if file_hash(destination) != digest:
                raise FileExistsError(f"Conflicting synchronized file: {destination}")
            temporary.unlink()
        else:
            os.rename(temporary, destination)
        if file_hash(destination) != digest:
            raise OSError(f"Published file failed checksum validation: {destination}")
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe stage artifact path: {value}")
    return path


def validate_backup(run_dir, task_names) -> list[dict]:
    """Validate metadata and the complete contiguous stage prefix in a backup."""
    run_dir = Path(run_dir).resolve()
    for name in REQUIRED_RUN_METADATA:
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Backup metadata is missing: {run_dir / name}")
    return completed_stage_markers(run_dir, task_names)


def sync_completed_stage(source_run, destination_run, task_names, stage: int) -> Path:
    """Synchronize one locally complete stage and publish its marker last."""
    source = Path(source_run).resolve()
    destination = Path(destination_run).resolve()
    if source == destination:
        raise ValueError("Synchronization source and destination must differ.")
    markers = completed_stage_markers(source, task_names)
    if stage < 0 or stage >= len(markers) or markers[stage]["stage"] != stage:
        raise ValueError(f"Local stage {stage} is not a validated completed stage.")
    destination.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_RUN_METADATA:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Required run metadata is missing: {path}")
        _copy_verified(path, destination / name)
    # A restored incomplete run must retain the lifecycle marker expected by resume.
    if (source / "RUNNING").is_file():
        _copy_verified(source / "RUNNING", destination / "RUNNING")
    elif (source / "RUN_COMPLETE").is_file():
        # A retry may run after local completion.  Until the synchronized
        # prefix itself is finalized, it must retain resumable-run semantics.
        _copy_verified(source / "RUN_COMPLETE", destination / "RUNNING")
    else:
        raise FileNotFoundError("Source run has no lifecycle marker.")

    marker = markers[stage]
    for identity in marker["artifacts"].values():
        relative = _safe_relative(identity["path"])
        _copy_verified(source / relative, destination / relative)

    # Never expose completion until every required byte has been checked.
    local_marker = marker_path(source, stage)
    remote_marker = marker_path(destination, stage)
    _copy_verified(local_marker, remote_marker)

    remote = validate_backup(destination, task_names)
    if len(remote) <= stage or remote[stage]["stage"] != stage:
        raise RuntimeError(f"Remote stage {stage} did not validate after publication.")
    return remote_marker


def sync_all_completed(source_run, destination_run, task_names) -> int:
    """Retry synchronization without training and return the latest stage."""
    markers = completed_stage_markers(source_run, task_names)
    if not markers:
        raise ValueError("Source run has no validated completed stages.")
    for marker in markers:
        sync_completed_stage(source_run, destination_run, task_names, marker["stage"])
    return markers[-1]["stage"]


def restore_backup(source_run, destination_run, task_names) -> int:
    """Restore the latest validated prefix into a fresh local run directory."""
    source = Path(source_run).resolve()
    destination = Path(destination_run).resolve()
    markers = validate_backup(source, task_names)
    if not markers:
        raise ValueError("Backup has no validated completed stages.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Restore destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_RUN_METADATA:
        _copy_verified(source / name, destination / name)
    lifecycle = "RUN_COMPLETE" if (source / "RUN_COMPLETE").is_file() else "RUNNING"
    if not (source / lifecycle).is_file():
        raise FileNotFoundError(f"Backup lifecycle marker is missing: {lifecycle}")
    _copy_verified(source / lifecycle, destination / lifecycle)
    for marker in markers:
        for identity in marker["artifacts"].values():
            relative = _safe_relative(identity["path"])
            _copy_verified(source / relative, destination / relative)
        _copy_verified(
            marker_path(source, marker["stage"]),
            marker_path(destination, marker["stage"]),
        )
    restored = completed_stage_markers(destination, task_names)
    if len(restored) != len(markers):
        raise RuntimeError("Restored stage prefix differs from the validated backup.")
    return markers[-1]["stage"]


def backup_status(run_dir, task_names) -> dict:
    """Return machine-readable validation status for notebook inspection."""
    markers = validate_backup(run_dir, task_names)
    return {
        "run_dir": str(Path(run_dir).resolve()),
        "completed_stages": [marker["stage"] for marker in markers],
        "latest_completed_stage": markers[-1]["stage"] if markers else None,
    }
