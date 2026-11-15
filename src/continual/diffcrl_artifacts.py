"""Pure construction helpers for DiffCRL run artifacts."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.config import (
    FLOW_CHECKPOINT_SELECTION,
    GENERATOR_GRADIENT_CLIP_NORM,
    GENERATOR_VALIDATION_DRAWS,
    REPLAY_PRIORITY_STRATEGY,
)
from src.support.io import atomic_replace_json, atomic_write_json, write_json
from src.support.reproducibility import file_hash, state_hash

from .bc_policy import BC_CLAMP_EPSILON
from .reporting import build_diffcrl_stage_report, print_stage_report

STAGE_MARKER_SCHEMA = 1


def marker_path(run_dir, stage: int) -> Path:
    return Path(run_dir) / f"stage_{stage}_COMPLETE.json"


def _relative(run_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError as error:
        raise ValueError(f"Stage artifact is outside the run directory: {path}") from error


def write_stage_marker(
    run_dir,
    *,
    stage,
    task,
    policy_path,
    diffusion_path,
    report_path,
    matrix_path,
    flow_matching_path=None,
    best_mse_policy_path=None,
    best_mse_matrix_path=None,
    artifact_hashes=None,
) -> Path:
    """Authenticate durable stage artifacts and publish completion last."""
    run_dir = Path(run_dir).resolve()
    paths = {
        "policy": Path(policy_path),
        "report": Path(report_path),
        "evaluation_matrix": Path(matrix_path),
    }
    if diffusion_path is not None:
        paths["diffusion"] = Path(diffusion_path)
    if flow_matching_path is not None:
        paths["flow_matching"] = Path(flow_matching_path)
    if best_mse_policy_path is not None:
        paths["best_mse_policy"] = Path(best_mse_policy_path)
    if best_mse_matrix_path is not None:
        paths["best_mse_evaluation_matrix"] = Path(best_mse_matrix_path)
    if diffusion_path is not None and flow_matching_path is not None:
        raise ValueError("A stage cannot contain two replay generators.")
    marker = {
        "schema_version": STAGE_MARKER_SCHEMA,
        "stage": stage,
        "task": task,
        "artifacts": {
            name: {
                "path": _relative(run_dir, path),
                "sha256": (
                    artifact_hashes[name]
                    if artifact_hashes is not None and name in artifact_hashes
                    else file_hash(path)
                ),
            }
            for name, path in paths.items()
        },
    }
    result = marker_path(run_dir, stage)
    atomic_write_json(result, marker)
    return result


def completed_stage_markers(run_dir, task_names) -> list[dict]:
    """Validate the contiguous prefix of completed, authenticated stages."""
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
        required = {"policy", "report", "evaluation_matrix"}
        if not required.issubset(marker.get("artifacts", {})):
            raise ValueError(f"Stage completion marker is incomplete: {path}")
        for name, identity in marker["artifacts"].items():
            artifact = (run_dir / identity["path"]).resolve()
            try:
                artifact.relative_to(run_dir)
            except ValueError as error:
                raise ValueError(
                    f"Marker artifact escapes run directory: {name}"
                ) from error
            if not artifact.is_file():
                raise FileNotFoundError(
                    f"Completed-stage artifact is missing: {artifact}"
                )
            if file_hash(artifact) != identity["sha256"]:
                raise ValueError(f"Completed-stage artifact is corrupted: {artifact}")
            identity["resolved_path"] = str(artifact)
        markers.append(marker)
    return markers


def synchronize_timing_device(device) -> None:
    """Make wall-clock boundaries reliable when work runs asynchronously on CUDA."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


@dataclass(frozen=True)
class StageArtifacts:
    """Immutable checkpoint locations produced for a completed stage update."""

    diffusion_path: Path | None
    policy_path: Path
    flow_matching_path: Path | None = None
    best_mse_policy_path: Path | None = None


def stage_checkpoint_paths(
    output,
    *,
    stage: int,
    task: str,
    has_diffusion: bool,
    has_flow_matching: bool,
) -> tuple[Path | None, Path, Path | None]:
    """Construct the established immutable checkpoint paths for one stage."""
    output = Path(output)
    diffusion_path = (
        output / f"diffusion_after_task_{stage}_{task}.pt"
        if has_diffusion
        else None
    )
    flow_matching_path = (
        output / f"flow_matching_after_task_{stage}_{task}.pt"
        if has_flow_matching
        else None
    )
    policy_path = output / f"general_policy_after_task_{stage}_{task}.pt"
    return diffusion_path, policy_path, flow_matching_path


def validate_finite_model_state(*models) -> None:
    """Reject a selected model set containing any nonfinite state tensor."""
    for model in models:
        if model is None:
            continue
        for value in model.state_dict().values():
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite trained state.")


def save_stage_checkpoints(coordinator, context, generator_training, policy_training):
    """Persist the established generator and policy checkpoint payloads."""
    cfg = coordinator.config
    stage = context.index
    task = context.task
    normalizer = generator_training.normalizer
    goal_normalizer = generator_training.goal_normalizer
    validate_finite_model_state(
        coordinator.policy, coordinator.diffusion, coordinator.flow_matching
    )
    diffusion_path, policy_path, flow_matching_path = stage_checkpoint_paths(
        coordinator.output,
        stage=stage,
        task=task.task_name,
        has_diffusion=coordinator.diffusion is not None,
        has_flow_matching=coordinator.flow_matching is not None,
    )
    generator_metadata = {
        "task_order": coordinator.sequence.task_names,
        "task_conditioning": "integer task embedding plus projected normalized 3D goal",
        "training_banks": coordinator.training_banks,
        "train_configurations": coordinator.train_configurations,
        "generated_action_projection": cfg.diffusion.generated_action_projection,
    }
    if coordinator.diffusion is not None:
        coordinator.diffusion.save(
            diffusion_path,
            normalizer,
            goal_normalizer,
            {
                **generator_metadata,
                "diffusion_action_space": cfg.diffusion.action_space,
                "diffusion_action_clamp_epsilon": cfg.diffusion.action_clamp_epsilon,
                "diffusion_fit": {
                    "epochs": cfg.diffusion.epochs,
                    "batch_size": cfg.diffusion.batch_size,
                    "learning_rate": cfg.diffusion.learning_rate,
                },
            },
        )
    if coordinator.flow_matching is not None:
        coordinator.flow_matching.save(
            flow_matching_path,
            normalizer,
            goal_normalizer,
            {
                "stage": stage,
                "task": task.task_name,
                "seed": cfg.runtime.seed,
                **generator_metadata,
                "action_space": cfg.diffusion.action_space,
                "action_clamp_epsilon": cfg.diffusion.action_clamp_epsilon,
                "flow_matching_fit": asdict(cfg.flow_matching),
            },
        )
    policy_metadata = {
        "stage": stage,
        "task": task.task_name,
        "tasks": coordinator.sequence.task_names,
        "seed": cfg.runtime.seed,
        "best_epoch": policy_training["best_epoch"],
        "bc_loss": cfg.bc.loss,
        "bc_objective": coordinator.policy.objective,
        "bc_initial_log_std": cfg.bc.initial_log_std,
        "bc_clamp_epsilon": BC_CLAMP_EPSILON,
        "bc_normalization_mode": "layer-norm",
        "bc_use_layer_norm": True,
        "bc_hidden_sizes": [cfg.bc.hidden_dim] * 4,
        "diffusion_action_space": cfg.diffusion.action_space,
        "diffusion_action_clamp_epsilon": cfg.diffusion.action_clamp_epsilon,
        "generated_action_projection": cfg.diffusion.generated_action_projection,
    }
    dual_selection = cfg.bc.dual_checkpoint_selection
    authoritative_selection = cfg.bc.checkpoint_selection
    if dual_selection or authoritative_selection != "validation_nll":
        selection_id = (
            f"{cfg.runtime.run_name or 'unnamed-run'}-seed{cfg.runtime.seed}-"
            f"stage{stage}-shared-nll-fit"
        )
        policy_metadata["checkpoint_selection"] = {
            "metric": authoritative_selection,
            **(
                {"training_objective": coordinator.policy.objective}
                if authoritative_selection != "validation_nll"
                else {}
            ),
            "selected_epoch": policy_training["best_epoch"],
            "validation_nll": policy_training["validation_loss"],
            "validation_action_mse": policy_training["validation_action_mse"],
            "shared_training_trajectory_id": selection_id,
        }
    if cfg.bc.task_conditioning == "one_hot":
        policy_metadata.update(
            bc_task_conditioning=cfg.bc.task_conditioning,
            task_order=list(coordinator.sequence.task_names),
        )
    coordinator.policy.save(
        policy_path,
        optimizer=coordinator.policy_optimizer,
        metadata=policy_metadata,
    )
    best_mse_policy_path = None
    if dual_selection:
        best_mse_policy_path = coordinator.output / (
            f"general_policy_best_mse_after_task_{stage}_{task.task_name}.pt"
        )
        alternative = deepcopy(coordinator.policy)
        alternative.load_state_dict(coordinator.best_mse_policy_state)
        validate_finite_model_state(alternative)
        best_mse = policy_training["best_mse_checkpoint"]
        alternative.save(
            best_mse_policy_path,
            metadata={
                **policy_metadata,
                "best_epoch": best_mse["epoch"],
                "checkpoint_selection": {
                    "metric": "validation_action_mse",
                    "selected_epoch": best_mse["epoch"],
                    "validation_nll": best_mse["validation_loss"],
                    "validation_action_mse": best_mse[
                        "validation_action_mse"
                    ],
                    "shared_training_trajectory_id": selection_id,
                },
            },
        )
    return StageArtifacts(
        diffusion_path=diffusion_path,
        policy_path=policy_path,
        flow_matching_path=flow_matching_path,
        best_mse_policy_path=best_mse_policy_path,
    )


def finish_stage(
    coordinator,
    context,
    data,
    generator_training,
    policy_training,
    artifacts,
    *,
    timing=None,
):
    """Write reports and the stage completion marker, then advance state."""
    cfg = coordinator.config
    stage = context.index
    task = context.task
    directory = context.directory
    diffusion_path = artifacts.diffusion_path
    flow_matching_path = artifacts.flow_matching_path
    policy_path = artifacts.policy_path
    best_mse_policy_path = artifacts.best_mse_policy_path
    checkpoint_paths = {
        "policy": policy_path,
        "diffusion": diffusion_path,
        "flow_matching": flow_matching_path,
        "best_mse_policy": best_mse_policy_path,
    }
    checkpoint_hashes = {
        name: file_hash(path)
        for name, path in checkpoint_paths.items()
        if path is not None
    }
    report = build_diffcrl_stage_report(
        cfg,
        context,
        data,
        generator_training,
        policy_training,
        artifacts,
        policy_after_state_sha256=state_hash(coordinator.policy),
        diffusion_sha256=checkpoint_hashes.get("diffusion"),
        flow_matching_sha256=checkpoint_hashes.get("flow_matching"),
        policy_sha256=checkpoint_hashes["policy"],
        best_mse_policy_sha256=checkpoint_hashes.get("best_mse_policy"),
    )
    if timing is not None:
        report["timing"] = timing
    coordinator.matrix.stage_provenance[str(stage)] = {
        "report": str(directory / "report.json"),
        "bc_normalization_mode": "layer-norm",
        "bc_use_layer_norm": True,
        "bc_loss": cfg.bc.loss,
        "bc_clamp_epsilon": BC_CLAMP_EPSILON,
        "generated_action_projection": cfg.diffusion.generated_action_projection,
    }
    best_mse_matrix_path = None
    best_mse_matrix = getattr(coordinator, "best_mse_matrix", None)
    if best_mse_matrix is not None:
        best_mse_matrix.stage_provenance[str(stage)] = {
            "report": str(directory / "report.json"),
            "checkpoint_selection": "validation_action_mse",
            "policy_checkpoint": str(best_mse_policy_path),
        }
    if timing is not None:
        synchronize_timing_device(cfg.runtime.device)
        timing["total_stage_seconds"] = time.perf_counter() - timing.pop(
            "_stage_started_perf_counter"
        )
        timing["total_stage_scope"] = (
            "stage entry through checkpointing, evaluation, report construction, "
            "and provenance preparation; excludes final artifact serialization"
        )
    write_json(directory / "report.json", report)
    matrix_payload = coordinator.matrix.to_dict()
    write_json(directory / "evaluation_matrix.json", matrix_payload)
    if best_mse_matrix is not None:
        best_mse_matrix_path = directory / "evaluation_matrix_best_mse.json"
        best_mse_matrix_payload = best_mse_matrix.to_dict()
        write_json(best_mse_matrix_path, best_mse_matrix_payload)
        atomic_replace_json(
            coordinator.output / "evaluation_matrix_best_mse.json",
            best_mse_matrix_payload,
            allow_nan=False,
        )
    for name, path in coordinator.experts.items():
        if file_hash(path) != coordinator.expert_hashes[name]:
            raise RuntimeError("Source expert changed during experiment.")
    coordinator.previous_diffusion = diffusion_path
    coordinator.previous_flow_matching = flow_matching_path
    coordinator.next_stage += 1
    write_stage_marker(
        coordinator.output,
        stage=stage,
        task=task.task_name,
        policy_path=policy_path,
        diffusion_path=diffusion_path,
        flow_matching_path=flow_matching_path,
        report_path=directory / "report.json",
        matrix_path=directory / "evaluation_matrix.json",
        best_mse_policy_path=best_mse_policy_path,
        best_mse_matrix_path=best_mse_matrix_path,
        artifact_hashes=checkpoint_hashes,
    )
    callback = getattr(coordinator, "stage_complete_callback", None)
    if callback is not None:
        callback(coordinator.output, stage, coordinator.sequence.task_names)
    print_stage_report(
        coordinator.matrix,
        stage,
        task_training_steps=None,
        global_timesteps=None,
        training_detail=f"Training trajectories/task: {context.trajectory_count}",
    )
    return report


def build_initial_run_summary(
    config,
    sequence,
    horizon: int,
    diffusion,
    flow_matching,
    experts,
    expert_hashes,
) -> dict[str, object]:
    """Build the established initial ``config.json`` payload without I/O."""
    diffusion_model_config = asdict(diffusion.config) if diffusion is not None else None
    flow_model_config = (
        asdict(flow_matching.config) if flow_matching is not None else None
    )
    return {
        "trajectories_per_task": config.continual.trajectories_per_task,
        **(
            {"current_trajectory_schedule": config.continual.current_trajectory_schedule}
            if config.continual.current_trajectory_schedule
            else {}
        ),
        "replay_allocation": config.continual.replay_allocation,
        "replay_budget": config.continual.replay_budget,
        "replay_priority_strategy": REPLAY_PRIORITY_STRATEGY,
        "diffusion_epochs": config.diffusion.epochs,
        "bc_epochs": config.bc.epochs,
        "eval_episodes": config.evaluation.episodes,
        "seed": config.runtime.seed,
        "eval_seed": config.evaluation.seed,
        "diffusion_steps": config.diffusion.steps,
        "diffusion_width": config.diffusion.width,
        "bc_normalization_mode": "layer-norm",
        "bc_use_layer_norm": True,
        "replay_mode": config.continual.replay_mode,
        **(
            {"replay_task_weights": config.continual.replay_task_weights}
            if config.continual.replay_allocation == "fixed-weighted"
            else {}
        ),
        "bc_loss": config.bc.loss,
        "diffusion_action_space": config.diffusion.action_space,
        "diffusion_action_clamp_epsilon": config.diffusion.action_clamp_epsilon,
        "generated_action_projection": config.diffusion.generated_action_projection,
        "evaluation_mode": config.evaluation.mode,
        "tasks": sequence.task_names,
        "environment_backend": config.environment.backend,
        "horizon": horizon,
        "task_set_seed": config.evaluation.task_set_seed
        if config.evaluation.mode == "fixed-tasks"
        else None,
        "diffusion": diffusion_model_config,
        **({"flow_matching": flow_model_config} if flow_matching is not None else {}),
        "experts": {
            name: {"path": str(path), "sha256": expert_hashes[name]}
            for name, path in experts.items()
        },
        "historical_replay_budget": config.continual.replay_budget
        if config.continual.replay_allocation in ("fixed-total", "fixed-weighted")
        or flow_matching is not None
        else None,
        "split": "per-task whole episodes; rounded 20% validation",
        "bc": {
            "hidden_sizes": [config.bc.hidden_dim] * 4,
            "use_layer_norm": True,
            "epochs": config.bc.epochs,
            "batch_size": config.bc.batch_size,
            "learning_rate": config.bc.learning_rate,
            "objective": config.bc.loss,
            "initial_log_std": config.bc.initial_log_std,
            "activation": config.bc.activation,
            "clamp_epsilon": BC_CLAMP_EPSILON,
            **(
                {"checkpoint_selection": config.bc.checkpoint_selection}
                if config.bc.checkpoint_selection != "validation_nll"
                else {}
            ),
            **(
                {"dual_checkpoint_selection": True}
                if config.bc.dual_checkpoint_selection
                else {}
            ),
        },
        "diffusion_fit": {
            "batch_size": config.diffusion.batch_size,
            "learning_rate": config.diffusion.learning_rate,
            "prediction": "x0",
        },
        **(
            {
                "flow_matching_fit": {
                    "epochs": config.flow_matching.epochs,
                    "batch_size": config.flow_matching.batch_size,
                    "learning_rate": config.flow_matching.learning_rate,
                    "gradient_clip_norm": GENERATOR_GRADIENT_CLIP_NORM,
                    "validation_samples": GENERATOR_VALIDATION_DRAWS,
                    "checkpoint_selection": FLOW_CHECKPOINT_SELECTION,
                }
            }
            if flow_matching is not None
            else {}
        ),
        "normalization": "stage training split only, feature std floor .001; no normalization clipping",
        "optimizers": (
            "BC Adam fresh per stage with paired best-epoch restoration; "
            + (
                "Flow Matching Adam fresh per fit"
                if flow_matching is not None
                else "diffusion Adam fresh per fit"
            )
        ),
        "collection": (
            f"{config.continual.expert_action_mode}; "
            "all episodes retained; any-step success recorded"
            if config.continual.collection_mode == "all"
            else (
                f"{config.continual.expert_action_mode}; "
                "successful episodes preferred with attempt-order "
                "failure fallback; any-step success recorded"
            )
        ),
    }
