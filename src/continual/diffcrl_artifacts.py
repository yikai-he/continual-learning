"""Pure construction helpers for DiffCRL run artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .bc_policy import BC_CLAMP_EPSILON


def stage_checkpoint_paths(
    output,
    *,
    stage: int,
    task: str,
    artifact_suffix: str,
    has_diffusion: bool,
    has_flow_matching: bool,
) -> tuple[Path | None, Path, Path | None]:
    """Construct the established immutable checkpoint paths for one stage."""
    output = Path(output)
    diffusion_path = (
        output / f"diffusion_after_task_{stage}_{task}{artifact_suffix}.pt"
        if has_diffusion
        else None
    )
    flow_matching_path = (
        output / f"flow_matching_after_task_{stage}_{task}{artifact_suffix}.pt"
        if has_flow_matching
        else None
    )
    policy_path = output / f"general_policy_after_task_{stage}_{task}{artifact_suffix}.pt"
    return diffusion_path, policy_path, flow_matching_path


def validate_finite_model_state(*models) -> None:
    """Reject a selected model set containing any nonfinite state tensor."""
    if not all(
        torch.isfinite(value).all()
        for model in models
        if model is not None
        for value in model.state_dict().values()
    ):
        raise ValueError("Nonfinite trained state.")


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
    return {
        "trajectories_per_task": config.continual.trajectories_per_task,
        "replay_allocation": config.continual.replay_allocation,
        "replay_budget": config.continual.replay_budget,
        "replay_priority_strategy": config.continual.replay_priority_strategy,
        "diffusion_epochs": config.diffusion.epochs,
        "bc_epochs": config.bc.epochs,
        "eval_episodes": config.evaluation.episodes,
        "seed": config.runtime.seed,
        "eval_seed": config.evaluation.seed,
        "diffusion_steps": config.diffusion.steps,
        "diffusion_width": config.diffusion.width,
        "bc_normalization_mode": config.bc.normalization_mode,
        "bc_use_layer_norm": config.bc.use_layer_norm,
        "replay_mode": config.continual.replay_mode,
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
        "diffusion": asdict(diffusion.config) if diffusion is not None else None,
        **(
            {"flow_matching": asdict(flow_matching.config)}
            if flow_matching is not None
            else {}
        ),
        "experts": {
            name: {"path": str(path), "sha256": expert_hashes[name]}
            for name, path in experts.items()
        },
        "historical_replay_budget": config.continual.replay_budget
        if config.continual.replay_allocation == "fixed-total"
        or flow_matching is not None
        else None,
        "split": "per-task whole episodes; rounded 20% validation",
        "bc": {
            "hidden_sizes": config.bc.hidden_sizes,
            "use_layer_norm": config.bc.use_layer_norm,
            "epochs": config.bc.epochs,
            "batch_size": config.bc.batch_size,
            "learning_rate": config.bc.learning_rate,
            "objective": config.bc.loss,
            "clamp_epsilon": BC_CLAMP_EPSILON,
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
                    "gradient_clip_norm": config.flow_matching.gradient_clip_norm,
                    "validation_samples": config.flow_matching.validation_samples,
                    "checkpoint_selection": config.flow_matching.checkpoint_selection,
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
        "collection": "deterministic; all episodes retained; any-step success recorded",
    }
