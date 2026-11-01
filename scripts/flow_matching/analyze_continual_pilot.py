#!/usr/bin/env python3
"""Append-only offline scientific diagnostics for completed Flow pilot stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from scripts.experiments.run_flow_matching_sanity import _sha256, _summary_diagnostics
from src.config import load_config
from src.continual.generation_quality import (
    expert_disagreement_by_timestep, state_delta_discrepancy_by_timestep,
)
from src.continual.resume import completed_stage_markers
from src.continual.trajectory_inspection import compare_datasets
from src.support.io import write_json


def _validate_or_write_marker(directory: Path, stage: int, artifacts: dict) -> None:
    path = directory / f"stage_{stage}_diagnostics_COMPLETE.json"
    if path.exists():
        marker = json.loads(path.read_text())
        for item in marker["artifacts"].values():
            artifact = Path(item["path"])
            if not artifact.is_file() or _sha256(artifact) != item["sha256"]:
                raise ValueError(f"Diagnostic artifact changed: {artifact}")
        return
    write_json(path, {
        "kind": "flow_matching_continual_stage_diagnostics_v1", "stage": stage,
        "artifacts": {key: {"path": str(value), "sha256": _sha256(value)} for key, value in artifacts.items()},
    })


def analyze(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    config = load_config(run_dir / "config.yaml", expected_experiment="diffcrl")
    if config.continual.replay_mode != "flow-matching":
        raise ValueError("Diagnostics require a Flow Matching run.")
    markers = completed_stage_markers(run_dir, config.continual.tasks)
    if not markers:
        raise ValueError("No completed stages to diagnose.")
    output = run_dir / "scientific_diagnostics"
    output.mkdir(exist_ok=True)
    experts = {
        name: SAC.load(path, device="cpu")
        for name, path in config.continual.experts.items()
    }
    stage_summaries = []
    for stage, task in enumerate(config.continual.tasks[: len(markers)]):
        stage_dir = run_dir / f"stage_{stage}_{task}"
        report = json.loads((stage_dir / "report.json").read_text())
        replay = {}
        for old_index, old_task in enumerate(config.continual.tasks[:stage]):
            generated = torch.from_numpy(np.load(stage_dir / f"data_task_{old_index}.npy")).float()
            reference_dir = run_dir / f"stage_{old_index}_{old_task}"
            real = torch.from_numpy(np.load(reference_dir / f"data_task_{old_index}.npy")).float()
            comparison = compare_datasets(tuple(real.numpy()), tuple(generated.numpy()), top_k=10)
            source = report["sources"][old_task]
            replay[old_task] = {
                "sampling_seed": source["sampling_seed"],
                "integration_steps": source["integration_steps"],
                "nfe": source["neural_function_evaluations"],
                "generation_seconds": source.get("generation_seconds"),
                "expert_action_disagreement": expert_disagreement_by_timestep(experts[old_task], generated),
                "state_delta_discrepancy": state_delta_discrepancy_by_timestep(real, generated),
                "distribution_comparison": comparison,
                "summary": _summary_diagnostics(comparison),
                "diagnostic_only": True,
                "used_for_training_selection_or_filtering": False,
            }
        matrix = json.loads((stage_dir / "evaluation_matrix.json").read_text())
        result = {
            "stage": stage, "task": task,
            "generator_validation_loss": report["flow_matching_losses"]["validation_loss"],
            "generator_selected_epoch": report["flow_matching_losses"]["best_epoch"],
            "generator_training_seconds": report.get("generator_training_seconds"),
            "bc_validation_action_mse": report["bc_losses"]["validation_action_mse"],
            "per_task_evaluation": matrix["cells"][stage],
            "continual_metrics": matrix["metrics_by_stage"][str(stage)],
            "replay": replay,
        }
        result_path = output / f"stage_{stage}_{task}.json"
        if not result_path.exists():
            write_json(result_path, result)
        _validate_or_write_marker(output, stage, {"diagnostics": result_path, "stage_completion": run_dir / f"stage_{stage}_COMPLETE.json"})
        stage_summaries.append(result)
    return {"completed_stages": len(markers), "stages": stage_summaries}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args()
    summary = analyze(args.run_dir)
    print(json.dumps({"completed_stages": summary["completed_stages"]}, indent=2))


if __name__ == "__main__": main()
