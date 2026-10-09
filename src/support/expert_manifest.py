"""Lightweight, backward-compatible identity records for SAC expert checkpoints."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from src.support.reproducibility import file_hash


@dataclass(frozen=True)
class ExpertIdentity:
    task_name: str
    checkpoint_path: str
    checkpoint_sha256: str
    backend: str
    algorithm: str
    reward_function_version: str
    training_seed: int
    training_config_reference: str | None
    training_config_sha256: str | None
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    horizon: int | None
    hammer_reward_variant: str | None
    hammer_nail_progress_weight: float | None
    qualification: dict | None
    manifest_path: str | None
    legacy: bool

    def to_dict(self) -> dict:
        value = dict(self.__dict__)
        value["observation_shape"] = list(self.observation_shape)
        value["action_shape"] = list(self.action_shape)
        return value


def manifest_candidates(checkpoint: str | Path) -> tuple[Path, ...]:
    """Return supported sidecar locations, most specific first."""
    checkpoint = Path(checkpoint)
    return (
        checkpoint.with_suffix(checkpoint.suffix + ".manifest.json"),
        checkpoint.with_suffix(".manifest.json"),
        checkpoint.parent / "expert_manifest.json",
    )


def find_expert_manifest(checkpoint: str | Path) -> Path | None:
    return next((path for path in manifest_candidates(checkpoint) if path.is_file()), None)


def load_expert_manifest(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expert manifest must contain a JSON object.")
    return data


def validate_expert_checkpoint(
    checkpoint: str | Path,
    *,
    expected_task: str,
    expected_backend: str,
    expected_reward_function_version: str,
    expected_observation_shape: tuple[int, ...],
    expected_action_shape: tuple[int, ...],
    expected_horizon: int | None = None,
    expected_hammer_reward_variant: str | None = None,
    expected_hammer_nail_progress_weight: float | None = None,
    manifest_path: str | Path | None = None,
    allow_legacy: bool = True,
) -> ExpertIdentity:
    """Validate a checkpoint against its sidecar, or explicitly admit legacy use."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expert checkpoint not found: {checkpoint}")
    digest = file_hash(checkpoint)
    located = Path(manifest_path).resolve() if manifest_path is not None else find_expert_manifest(checkpoint)
    if located is None:
        if not allow_legacy:
            raise FileNotFoundError(
                f"Expert manifest is required for {expected_task}: {checkpoint}"
            )
        warnings.warn(
            f"LEGACY EXPERT without manifest for {expected_task}: {checkpoint}; "
            "task and reward identity cannot be proven.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ExpertIdentity(
            task_name=expected_task,
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=digest,
            backend=expected_backend,
            algorithm="SAC (legacy-unverified)",
            reward_function_version=expected_reward_function_version,
            training_seed=-1,
            training_config_reference=None,
            training_config_sha256=None,
            observation_shape=tuple(expected_observation_shape),
            action_shape=tuple(expected_action_shape),
            horizon=expected_horizon,
            hammer_reward_variant=expected_hammer_reward_variant,
            hammer_nail_progress_weight=expected_hammer_nail_progress_weight,
            qualification=None,
            manifest_path=None,
            legacy=True,
        )

    data = load_expert_manifest(located)
    required = {
        "task_name",
        "checkpoint_sha256",
        "backend",
        "algorithm",
        "reward_function_version",
        "training_seed",
        "observation_shape",
        "action_shape",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Expert manifest missing field: {missing[0]}.")
    if expected_horizon is not None and "horizon" not in data:
        raise ValueError("Expert manifest missing field: horizon.")
    if expected_hammer_reward_variant is not None:
        for field in ("hammer_reward_variant", "hammer_nail_progress_weight"):
            if field not in data:
                raise ValueError(f"Expert manifest missing field: {field}.")
    checks = (
        (data["task_name"] == expected_task, "task name"),
        (data["checkpoint_sha256"] == digest, "checkpoint SHA-256"),
        (data["backend"] == expected_backend, "backend"),
        (
            data["reward_function_version"] == expected_reward_function_version,
            "reward version",
        ),
        (
            tuple(data["observation_shape"]) == tuple(expected_observation_shape),
            "observation shape",
        ),
        (
            tuple(data["action_shape"]) == tuple(expected_action_shape),
            "action shape",
        ),
        (expected_horizon is None or data["horizon"] == expected_horizon, "horizon"),
        (
            expected_hammer_reward_variant is None
            or data["hammer_reward_variant"] == expected_hammer_reward_variant,
            "Hammer reward variant",
        ),
        (
            expected_hammer_nail_progress_weight is None
            or data["hammer_nail_progress_weight"]
            == expected_hammer_nail_progress_weight,
            "Hammer nail progress weight",
        ),
    )
    for valid, label in checks:
        if not valid:
            raise ValueError(
                f"Expert manifest {label} mismatch for {expected_task}: {located}"
            )
    qualification = data.get("qualification")
    if qualification is not None and not isinstance(qualification, dict):
        raise ValueError("Expert manifest qualification must be an object or null.")
    return ExpertIdentity(
        task_name=data["task_name"],
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=digest,
        backend=data["backend"],
        algorithm=data["algorithm"],
        reward_function_version=data["reward_function_version"],
        training_seed=data["training_seed"],
        training_config_reference=data.get("training_config_reference"),
        training_config_sha256=data.get("training_config_sha256"),
        observation_shape=tuple(data["observation_shape"]),
        action_shape=tuple(data["action_shape"]),
        horizon=data.get("horizon"),
        hammer_reward_variant=data.get("hammer_reward_variant"),
        hammer_nail_progress_weight=data.get("hammer_nail_progress_weight"),
        qualification=qualification,
        manifest_path=str(located),
        legacy=False,
    )


def qualification_record(
    *,
    success_rate: float,
    mean_return: float,
    return_std: float,
    episodes: int,
    task_set_seed: int,
    evaluation_seed: int,
    deterministic: bool,
    task_hashes: list[str],
) -> dict:
    """Canonical fixed-bank qualification metrics; selection is success then return."""
    return {
        "protocol": "fixed-mt1-task-bank-v1",
        "episodes": episodes,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "return_std": return_std,
        "task_set_seed": task_set_seed,
        "evaluation_seed": evaluation_seed,
        "deterministic": deterministic,
        "ordered_task_data_sha256": list(task_hashes),
        "selection_key": [success_rate, mean_return],
        "selection_rule": "maximum success_rate; mean_return breaks ties",
    }


def write_expert_manifest(
    checkpoint: str | Path,
    *,
    task_name: str,
    backend: str,
    reward_function_version: str,
    training_seed: int,
    training_config_reference: str | None,
    training_config_sha256: str | None,
    observation_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    horizon: int | None = None,
    hammer_reward_variant: str | None = None,
    hammer_nail_progress_weight: float | None = None,
    qualification: dict | None,
) -> Path:
    """Write the checkpoint-specific sidecar consumed by validation."""
    checkpoint = Path(checkpoint).resolve()
    path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
    value = {
        "schema_version": 1,
        "task_name": task_name,
        "checkpoint_identifier": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "backend": backend,
        "algorithm": "SAC",
        "reward_function_version": reward_function_version,
        "training_seed": training_seed,
        "training_config_reference": training_config_reference,
        "training_config_sha256": training_config_sha256,
        "observation_shape": list(observation_shape),
        "action_shape": list(action_shape),
        "horizon": horizon,
        "qualification_status": "qualified"
        if qualification is not None
        else "not-qualified",
        "qualification": qualification,
    }
    if hammer_reward_variant is not None:
        value["hammer_reward_variant"] = hammer_reward_variant
        value["hammer_nail_progress_weight"] = hammer_nail_progress_weight
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path
