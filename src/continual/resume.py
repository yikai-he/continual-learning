"""Validated, stage-boundary resume state for DiffCRL runs."""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch

from src.config import config_to_dict, load_config
from src.support.io import atomic_torch_save, atomic_write_json
from src.support.reproducibility import file_hash

STAGE_MARKER_SCHEMA = 1


def _relative(run_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError as error:
        raise ValueError(f"Stage artifact is outside the run directory: {path}") from error


def _scientific_config(config) -> dict:
    """Remove relocatable execution paths while retaining scientific settings."""
    value = config_to_dict(config)
    value["continual"] = dict(value["continual"])
    value["continual"]["experts"] = sorted(value["continual"]["experts"])
    value["runtime"] = {"seed": value["runtime"]["seed"]}
    return value


def validate_resume_configuration(run_dir, requested, expert_hashes) -> None:
    """Require identical science and expert bytes while allowing path relocation."""
    run_dir = Path(run_dir).resolve()
    saved = load_config(run_dir / "config.yaml", expected_experiment="diffcrl")
    if _scientific_config(saved) != _scientific_config(requested):
        raise ValueError("Resume configuration differs from the saved scientific configuration.")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    recorded = manifest.get("experts", {})
    if set(recorded) != set(expert_hashes):
        raise ValueError("Resume expert task set differs from the saved run.")
    for task, digest in expert_hashes.items():
        if recorded[task].get("checkpoint_sha256") != digest:
            raise ValueError(f"Resume expert checkpoint mismatch for {task}.")


def capture_rng_state() -> dict:
    """Capture process RNGs at a completed stage boundary."""
    np_state = np.random.get_state()
    return {
        "python": list(random.getstate()),
        "numpy": {
            "kind": np_state[0],
            "keys": torch.from_numpy(np_state[1].copy()),
            "position": np_state[2],
            "has_gauss": np_state[3],
            "cached_gaussian": np_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def restore_rng_state(state: dict) -> None:
    random.setstate(_tuples(state["python"]))
    numpy = state["numpy"]
    np.random.set_state(
        (
            numpy["kind"],
            numpy["keys"].cpu().numpy().astype(np.uint32, copy=False),
            numpy["position"],
            numpy["has_gauss"],
            numpy["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda = state.get("torch_cuda", [])
    if cuda:
        if torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda])
        else:
            warnings.warn(
                "Saved CUDA RNG topology is unavailable; CPU/Python/NumPy RNGs "
                "were restored, but cross-device numerical equivalence is not guaranteed.",
                RuntimeWarning,
                stacklevel=2,
            )


def marker_path(run_dir, stage: int) -> Path:
    return Path(run_dir) / f"stage_{stage}_COMPLETE.json"


def write_stage_boundary(
    run_dir,
    *,
    stage,
    task,
    policy_path,
    diffusion_path,
    report_path,
    matrix_path,
    training_banks,
    train_configurations,
    artifact_suffix="",
) -> Path:
    """Persist restorable state and write the completion marker last."""
    run_dir = Path(run_dir).resolve()
    state_path = (
        run_dir / f"stage_state_after_task_{stage}_{task}{artifact_suffix}.pt"
    )
    state = {
        "kind": "diffcrl_stage_boundary_v1",
        "stage": stage,
        "task": task,
        "training_banks": training_banks,
        "train_configurations": train_configurations,
        "rng": capture_rng_state(),
    }
    atomic_torch_save(state_path, state)
    paths = {
        "policy": Path(policy_path),
        "stage_state": state_path,
        "report": Path(report_path),
        "evaluation_matrix": Path(matrix_path),
    }
    if diffusion_path is not None:
        paths["diffusion"] = Path(diffusion_path)
    artifacts = {
        name: {"path": _relative(run_dir, path), "sha256": file_hash(path)}
        for name, path in paths.items()
    }
    marker = {
        "schema_version": STAGE_MARKER_SCHEMA,
        "stage": stage,
        "task": task,
        "artifacts": artifacts,
    }
    result = marker_path(run_dir, stage)
    atomic_write_json(result, marker)
    return result


def completed_stage_markers(run_dir, task_names) -> list[dict]:
    """Load a contiguous marker prefix and validate every artifact hash."""
    run_dir = Path(run_dir).resolve()
    markers = []
    missing_seen = False
    for stage, task in enumerate(task_names):
        path = marker_path(run_dir, stage)
        if not path.is_file():
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError("Stage completion markers are not contiguous.")
        marker = json.loads(path.read_text(encoding="utf-8"))
        if (
            marker.get("schema_version") != STAGE_MARKER_SCHEMA
            or marker.get("stage") != stage
            or marker.get("task") != task
        ):
            raise ValueError(f"Invalid stage completion marker: {path}")
        required = {"policy", "stage_state", "report", "evaluation_matrix"}
        if not required.issubset(marker.get("artifacts", {})):
            raise ValueError(f"Stage completion marker is incomplete: {path}")
        for name, identity in marker["artifacts"].items():
            artifact = (run_dir / identity["path"]).resolve()
            try:
                artifact.relative_to(run_dir)
            except ValueError as error:
                raise ValueError(f"Marker artifact escapes run directory: {name}") from error
            if not artifact.is_file():
                raise FileNotFoundError(f"Completed-stage artifact is missing: {artifact}")
            if file_hash(artifact) != identity["sha256"]:
                raise ValueError(f"Completed-stage artifact is corrupted: {artifact}")
            identity["resolved_path"] = str(artifact)
        markers.append(marker)
    return markers


def load_boundary_state(marker: dict) -> dict:
    path = marker["artifacts"]["stage_state"]["resolved_path"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("kind") != "diffcrl_stage_boundary_v1"
        or state.get("stage") != marker["stage"]
        or state.get("task") != marker["task"]
    ):
        raise ValueError("Stage boundary state does not match its completion marker.")
    return state
