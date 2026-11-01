#!/usr/bin/env python3
"""Resumable matched single-task Flow/Diffusion/real-data validation jobs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC

from scripts.experiments.run_flow_matching_sanity import (
    _fit_and_evaluate_bc, _sha256, _summary_diagnostics,
)
from scripts.flow_matching.artifact_markers import (
    canonical_sha as _canonical_sha,
    require_completion_marker as _require_marker,
    write_completion_marker as _marker,
)
from src.config import DiffusionModelConfig
from src.continual.diffusion_data import (
    decode_trajectory, encode_trajectory, transform_actions, unpack_trajectories,
    validate_stage_trajectory_group,
)
from src.continual.generation_quality import (
    expert_disagreement_by_timestep, state_delta_discrepancy_by_timestep,
)
from src.continual.trajectory_diffusion import (
    FeatureNormalizer, TrajectoryDiffusion, fit_diffusion,
)
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig, TrajectoryFlowMatching, fit_flow_matching,
)
from src.continual.trajectory_inspection import compare_datasets
from src.support.io import write_json


KIND = "flow_matching_harder_single_task_v1"
JOBS = (
    "train-flow", "train-diffusion", "generate-flow-10", "generate-flow-50",
    "generate-diffusion", "bc-real", "bc-flow-10", "bc-flow-50", "bc-diffusion",
)


def load_config(path: Path, debug: bool = False) -> dict:
    config = yaml.safe_load(path.read_text())
    schemas = {
        "assets": {"expert_dataset", "source_report", "expert_checkpoint"},
        "flow": {"epochs", "batch_size", "learning_rate", "width", "time_embedding_scale", "integration_steps"},
        "diffusion": {"epochs", "batch_size", "learning_rate", "width", "steps", "action_space", "action_clamp_epsilon"},
        "generation": {"trajectories", "seed", "action_projection"},
        "bc": {"epochs", "batch_size", "learning_rate", "hidden_sizes", "loss", "initialization_seed", "shuffle_seed"},
        "evaluation": {"episodes", "seed", "mode", "task_set_seed"},
        "debug": {"flow_epochs", "diffusion_epochs", "diffusion_steps", "trajectories", "bc_epochs", "evaluation_episodes"},
    }
    required = {"kind", "task", "backend", "reward_function_version", "seed", "device", "output_root", *schemas}
    if set(config) != required or config["kind"] != KIND or config["device"] != "cpu":
        raise ValueError("Unknown, missing, or incompatible harder-task configuration field.")
    for section, fields in schemas.items():
        if set(config[section]) != fields:
            raise ValueError(f"Unknown or missing {section} fields.")
    if config["flow"]["integration_steps"] != [10, 50]:
        raise ValueError("Harder-task Flow comparison must contain exactly 10 and 50 steps.")
    if config["flow"]["epochs"] != 200 or config["diffusion"]["action_space"] != "raw":
        raise ValueError("Generator settings differ from the approved validation design.")
    if config["generation"]["action_projection"] != "none" or config["generation"]["trajectories"] != 200:
        raise ValueError("Generation must use 200 raw, unprojected trajectories.")
    numeric = [
        config["flow"][key] for key in ("epochs", "batch_size", "learning_rate", "width", "time_embedding_scale")
    ] + [config["diffusion"][key] for key in ("epochs", "batch_size", "learning_rate", "width", "steps")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in numeric):
        raise ValueError("Training settings must be positive.")
    config = json.loads(json.dumps(config))
    config["output_dir"] = config["output_root"] + ("_debug" if debug else "")
    if debug:
        values = config["debug"]
        config["flow"]["epochs"] = values["flow_epochs"]
        config["diffusion"]["epochs"] = values["diffusion_epochs"]
        config["diffusion"]["steps"] = values["diffusion_steps"]
        config["generation"]["trajectories"] = values["trajectories"]
        config["bc"]["epochs"] = values["bc_epochs"]
        config["evaluation"]["episodes"] = values["evaluation_episodes"]
    return config


def initialize(config: dict, root: Path):
    output = (root / config["output_dir"]).resolve()
    config_sha = _canonical_sha(config)
    resolved = output / "resolved_config.yaml"
    if output.exists():
        if not resolved.is_file() or _canonical_sha(yaml.safe_load(resolved.read_text())) != config_sha:
            raise ValueError("Existing output directory has a different configuration.")
    else:
        output.mkdir(parents=True)
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    assets = {key: (root / value).resolve() for key, value in config["assets"].items()}
    if missing := [str(path) for path in assets.values() if not path.is_file()]:
        raise FileNotFoundError(f"Missing required assets: {missing}")
    provenance = output / "provenance.json"
    if not provenance.exists():
        write_json(provenance, {
            "config_sha256": config_sha,
            "assets": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in assets.items()},
        })
    return output, config_sha, assets


def dataset(config, assets):
    real = torch.from_numpy(np.load(assets["expert_dataset"])).float()
    report = json.loads(assets["source_report"].read_text())
    validate_stage_trajectory_group(real, stage=0, task=config["task"], source=report["sources"][config["task"]])
    observations, actions = unpack_trajectories(real)
    dynamic, goals = encode_trajectory(observations, actions)
    split = report["splits"][config["task"]]
    return real, transform_actions(dynamic, "raw"), goals, split["train_ids"], split["validation_ids"]


def _attempt(output, job):
    path = output / f'{job.replace("-", "_")}_attempt_{time.time_ns()}'
    path.mkdir()
    return path


def train_flow(config, root, progress):
    output, csha, assets = initialize(config, root)
    if (output / "train-flow.complete.json").exists():
        _require_marker(output, "train-flow", csha); return
    _, dynamic, goals, train_ids, validation_ids = dataset(config, assets)
    norm = FeatureNormalizer.fit(dynamic[train_ids]); goal_norm = FeatureNormalizer.fit(goals[train_ids, None, :])
    model = TrajectoryFlowMatching(FlowMatchingConfig(horizon=200, features=22, width=config["flow"]["width"], num_tasks=1, integration_steps=10, time_embedding_scale=config["flow"]["time_embedding_scale"]))
    labels = torch.zeros(len(dynamic), dtype=torch.long); start = time.perf_counter()
    fit = fit_flow_matching(model, norm.normalize(dynamic[train_ids]), norm.normalize(dynamic[validation_ids]), goal_norm.normalize(goals[train_ids, None, :])[:, 0], goal_norm.normalize(goals[validation_ids, None, :])[:, 0], train_task_ids=labels[train_ids], validation_task_ids=labels[validation_ids], epochs=config["flow"]["epochs"], batch_size=config["flow"]["batch_size"], learning_rate=config["flow"]["learning_rate"], seed=config["seed"], progress=progress)
    attempt = _attempt(output, "train-flow"); curve = attempt / "training_curve.json"; metrics = attempt / "training_metrics.json"; checkpoint = attempt / "flow_matching.pt"
    write_json(curve, fit); write_json(metrics, {"seconds": time.perf_counter() - start, "selected_epoch": fit["best_epoch"], "train_loss": fit["train_loss"], "validation_loss": fit["validation_loss"]})
    model.save(checkpoint, norm, goal_norm, {"experiment_kind": KIND, "task_order": [config["task"]], "training_banks": {}, "train_configurations": {}, "generated_action_projection": "none", "action_space": "raw", "action_clamp_epsilon": 1e-4, "fit": fit})
    _marker(output, "train-flow", csha, {"checkpoint": checkpoint, "curve": curve, "metrics": metrics})


def train_diffusion(config, root, progress):
    output, csha, assets = initialize(config, root)
    if (output / "train-diffusion.complete.json").exists():
        _require_marker(output, "train-diffusion", csha); return
    _, dynamic, goals, train_ids, validation_ids = dataset(config, assets)
    norm = FeatureNormalizer.fit(dynamic[train_ids]); goal_norm = FeatureNormalizer.fit(goals[train_ids, None, :]); dc = config["diffusion"]
    model = TrajectoryDiffusion(DiffusionModelConfig(horizon=200, features=22, steps=dc["steps"], width=dc["width"], num_tasks=1, diffusion_action_space="raw", diffusion_action_clamp_epsilon=dc["action_clamp_epsilon"]))
    labels = torch.zeros(len(dynamic), dtype=torch.long); start = time.perf_counter()
    fit = fit_diffusion(model, norm.normalize(dynamic[train_ids]), norm.normalize(dynamic[validation_ids]), goal_norm.normalize(goals[train_ids, None, :])[:, 0], goal_norm.normalize(goals[validation_ids, None, :])[:, 0], train_task_ids=labels[train_ids], validation_task_ids=labels[validation_ids], epochs=dc["epochs"], batch_size=dc["batch_size"], learning_rate=dc["learning_rate"], seed=config["seed"], progress=progress)
    attempt = _attempt(output, "train-diffusion"); curve = attempt / "training_curve.json"; metrics = attempt / "training_metrics.json"; checkpoint = attempt / "diffusion.pt"
    write_json(curve, fit); write_json(metrics, {"seconds": time.perf_counter() - start, "selected_epoch": fit["best_epoch"], "train_loss": fit["train_loss"], "validation_loss": fit["validation_loss"]})
    model.save(checkpoint, norm, goal_norm, {"task_order": [config["task"]], "generated_action_projection": "none", "experiment_kind": KIND})
    _marker(output, "train-diffusion", csha, {"checkpoint": checkpoint, "curve": curve, "metrics": metrics})


def generate(config, root, kind, steps, progress):
    output, csha, assets = initialize(config, root); job = f"generate-{kind}" + (f"-{steps}" if kind == "flow" else "")
    if (output / f"{job}.complete.json").exists(): _require_marker(output, job, csha); return
    marker = _require_marker(output, f"train-{kind}", csha); checkpoint = Path(marker["artifacts"]["checkpoint"]["path"])
    real, _, goals, train_ids, _ = dataset(config, assets); count = config["generation"]["trajectories"]; selected_goals = goals[train_ids][torch.arange(count) % len(train_ids)]
    if kind == "flow":
        model, norm, goal_norm, _ = TrajectoryFlowMatching.load(checkpoint, expected_task_order=[config["task"]]); kwargs = {"integration_steps": steps, "progress": progress}
    else:
        model, norm, goal_norm, _ = TrajectoryDiffusion.load(checkpoint, expected_action_space="raw", expected_projection="none", expected_task_order=[config["task"]]); kwargs = {}
    start = time.perf_counter(); normalized = model.sample(count, seed=config["generation"]["seed"], task_ids=torch.zeros(count, dtype=torch.long), goals=goal_norm.normalize(selected_goals[:, None, :])[:, 0], **kwargs); seconds = time.perf_counter() - start
    dynamic = norm.denormalize(normalized); physical = decode_trajectory(transform_actions(dynamic, "raw", decode=True), selected_goals); validate_stage_trajectory_group(physical, stage=0, task=config["task"], source={"kind": "generated"})
    attempt = _attempt(output, job); structured = attempt / "generated_structured.npy"; physical_path = attempt / "generated_physical.npy"; metrics = attempt / "generation_metrics.json"; diagnostics = attempt / "diagnostics.json"
    np.save(structured, dynamic.numpy()); np.save(physical_path, physical.numpy())
    write_json(metrics, {"seconds": seconds, "nfe": model.last_sample_nfe if kind == "flow" else model.config.steps, "seed": config["generation"]["seed"], "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "structured_shape": list(dynamic.shape), "physical_shape": list(physical.shape)})
    comparison = compare_datasets(tuple(real.numpy()), tuple(physical.numpy()), top_k=10); expert = SAC.load(assets["expert_checkpoint"], device="cpu")
    write_json(diagnostics, {"expert_action_disagreement": expert_disagreement_by_timestep(expert, physical), "state_delta_discrepancy": state_delta_discrepancy_by_timestep(real, physical), "distribution_comparison": comparison, "summary": _summary_diagnostics(comparison), "used_for_filtering": False})
    _marker(output, job, csha, {"structured": structured, "physical": physical_path, "metrics": metrics, "diagnostics": diagnostics})


def bc(config, root, source, progress):
    output, csha, assets = initialize(config, root); job = f"bc-{source}"
    if (output / f"{job}.complete.json").exists(): _require_marker(output, job, csha); return
    real, _, _, train_ids, validation_ids = dataset(config, assets)
    if source == "real": data = real
    else:
        generated_job = "generate-" + source
        marker = _require_marker(output, generated_job, csha); data = torch.from_numpy(np.load(marker["artifacts"]["physical"]["path"])).float()
    attempt = _attempt(output, job); result = _fit_and_evaluate_bc(source, data, train_ids, validation_ids, config, attempt, progress=progress); curve = attempt / "bc_training_curve.json"; results = attempt / "bc_results.json"
    write_json(curve, result["fit"]); write_json(results, result); _marker(output, job, csha, {"policy": attempt / f"bc_{source}.pt", "curve": curve, "results": results})


def run_job(config, root, job, progress):
    if job == "train-flow": train_flow(config, root, progress)
    elif job == "train-diffusion": train_diffusion(config, root, progress)
    elif job == "generate-flow-10": generate(config, root, "flow", 10, progress)
    elif job == "generate-flow-50": generate(config, root, "flow", 50, progress)
    elif job == "generate-diffusion": generate(config, root, "diffusion", None, progress)
    elif job.startswith("bc-"): bc(config, root, job[3:], progress)
    else: raise ValueError(f"Unknown job: {job}")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--job", choices=(*JOBS, "all"), required=True); parser.add_argument("--debug", action="store_true"); parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True); args = parser.parse_args()
    config = load_config(args.config.resolve(), args.debug); root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1); torch.manual_seed(config["seed"]); np.random.seed(config["seed"])
    for job in (JOBS if args.job == "all" else (args.job,)): run_job(config, root, job, args.progress)
    if args.job == "all":
        output, csha, _ = initialize(config, root)
        if not (output / "validation.complete.json").exists():
            _marker(output, "validation", csha, {job: output / f"{job}.complete.json" for job in JOBS})


if __name__ == "__main__": main()
