"""Offline run provenance and atomic lifecycle markers for formal workflows."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch

from src.support.reproducibility import file_hash, git_output


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(path: Path) -> dict:
    changed = git_output(path, "status", "--short")
    diff = git_output(path, "diff", "--binary", "HEAD")
    return {
        "path": str(path.resolve()),
        "commit": git_output(path, "rev-parse", "HEAD"),
        "branch": git_output(path, "branch", "--show-current"),
        "dirty": bool(changed) if changed is not None else None,
        "changed_files": changed.splitlines() if changed else [],
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()
        if diff is not None
        else None,
    }


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _environment(config) -> dict:
    try:
        import metaworld

        checkout = Path(metaworld.__file__).resolve().parent.parent
        metaworld_git = _git(checkout)
    except (ImportError, AttributeError):
        checkout, metaworld_git = None, None
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "stable_baselines3_version": _version("stable-baselines3"),
        "gymnasium_version": _version("gymnasium"),
        "mujoco_version": _version("mujoco"),
        "metaworld_version": _version("metaworld"),
        "metaworld_checkout": str(checkout) if checkout is not None else None,
        "metaworld_git": metaworld_git,
        "resolved_device": config.runtime.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def create_run_manifest(
    run_dir: str | Path,
    config,
    *,
    experts: dict | None = None,
    task_banks: dict | None = None,
) -> dict:
    """Create a new manifest and RUNNING marker; never overwrite existing state."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved configuration is missing: {config_path}")
    evaluation_seed = getattr(config.evaluation, "seed", None)
    if evaluation_seed is None:
        evaluation_seed = config.runtime.seed + config.evaluation.seed_offset
    manifest = {
        "schema_version": 1,
        "status": "running",
        "run_started_at": _now(),
        "run_completed_at": None,
        "repository": _git(Path(__file__).resolve().parents[2]),
        "environment": _environment(config),
        "experiment": {
            "name": config.experiment,
            "config_path": "config.yaml",
            "config_sha256": file_hash(config_path),
            "task_sequence": list(config.continual.tasks),
            "runtime_seed": config.runtime.seed,
            "evaluation_seed": evaluation_seed,
            "task_set_seed": config.evaluation.task_set_seed,
        },
        "experts": experts or {},
        "task_banks": task_banks or {},
    }
    with (run_dir / "run_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    (run_dir / "RUNNING").write_text(
        manifest["run_started_at"] + "\n", encoding="utf-8"
    )
    return manifest


def complete_run(run_dir: str | Path) -> None:
    """Atomically mark a running workflow complete after final outputs exist."""
    run_dir = Path(run_dir)
    running = run_dir / "RUNNING"
    complete = run_dir / "RUN_COMPLETE"
    if complete.exists():
        raise FileExistsError(f"Run is already complete: {run_dir}")
    if not running.is_file():
        raise RuntimeError(f"RUNNING marker is missing: {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["run_completed_at"] = _now()
    temporary = run_dir / ".run_manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    running.replace(complete)
