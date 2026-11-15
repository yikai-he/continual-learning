#!/usr/bin/env python3
"""Evaluate saved continual NLL checkpoints with deterministic or sampled actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continual.bc_policy import (  # noqa: E402
    LOG_STD_MAX,
    LOG_STD_MIN,
    SQUASHED_GAUSSIAN_NLL,
    GeneralPolicy,
)
from continual_manipulation.evaluation import EvaluationMatrix, evaluate_stage  # noqa: E402
from continual_manipulation.sequences import TaskSequence  # noqa: E402
from src.support.io import write_json  # noqa: E402
from src.support.reproducibility import file_hash  # noqa: E402


class EvaluationActionPolicy:
    """Select deterministic or learned-distribution actions without changing weights."""

    def __init__(self, policy: GeneralPolicy, action_mode: str) -> None:
        if action_mode not in ("deterministic", "stochastic"):
            raise ValueError("action_mode must be deterministic or stochastic.")
        if action_mode == "stochastic" and policy.objective != SQUASHED_GAUSSIAN_NLL:
            raise ValueError("Stochastic evaluation requires a learned NLL distribution.")
        self.policy = policy
        self.action_mode = action_mode
        self.objective = policy.objective
        self.task_conditioning = policy.task_conditioning

    def act(self, observation, deterministic=True, *, task_id=None):
        del deterministic
        return self.policy.act(
            observation,
            deterministic=self.action_mode == "deterministic",
            task_id=task_id,
        )


def load_run(source_run: Path):
    source_run = source_run.resolve()
    if not (source_run / "RUN_COMPLETE").is_file():
        raise ValueError("source-run must be a completed fixed-dataset NLL run.")
    config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    if config["bc"]["loss"] != SQUASHED_GAUSSIAN_NLL:
        raise ValueError("source-run must use squashed_gaussian_nll.")
    tasks = config["tasks"]
    checkpoints = [
        source_run / f"general_policy_after_task_{stage}_{task}.pt"
        for stage, task in enumerate(tasks)
    ]
    policies = [
        GeneralPolicy.load(path, expected_objective=SQUASHED_GAUSSIAN_NLL)
        for path in checkpoints
    ]
    data_config = yaml.safe_load(
        (Path(config["source_run"]) / "config.yaml").read_text(encoding="utf-8")
    )
    config["environment"] = data_config["environment"]
    config["evaluation_source_matrix"] = str(source_run / "evaluation_matrix.json")
    return config, tasks, checkpoints, policies


def uncertainty_diagnostics(config, tasks, policies):
    data_run = Path(config["source_run"])
    results, warnings = [], []
    for stage, current_task in enumerate(tasks):
        stage_dir = data_run / f"stage_{stage}_{current_task}"
        report = json.loads((stage_dir / "report.json").read_text(encoding="utf-8"))
        for task_index, task in enumerate(tasks[: stage + 1]):
            group = np.load(stage_dir / f"data_task_{task_index}.npy", mmap_mode="r")
            ids = report["splits"][task]["validation_ids"]
            observations = np.asarray(group[ids, :, :39]).reshape(-1, 39).astype(np.float32)
            with torch.no_grad():
                _, log_std = policies[stage].distribution_parameters(
                    torch.from_numpy(observations)
                )
            values = log_std.numpy()
            upper = float(np.mean(np.isclose(values, LOG_STD_MAX)))
            lower = float(np.mean(np.isclose(values, LOG_STD_MIN)))
            record = {
                "stage": stage, "task": task, "source_kind": report["sources"][task]["kind"],
                "samples": len(observations), "mean_log_std": float(values.mean()),
                "mean_std": float(np.exp(values).mean()), "minimum_log_std": float(values.min()),
                "maximum_log_std": float(values.max()), "upper_bound_fraction": upper,
                "lower_bound_fraction": lower, "all_finite": bool(np.isfinite(values).all()),
            }
            results.append(record)
            if upper >= 0.5 or lower >= 0.5:
                warnings.append(
                    f"Severe log-std saturation: stage={stage}, task={task}, "
                    f"upper={upper:.3f}, lower={lower:.3f}"
                )
    return {"records": results, "warnings": warnings}


def matrix_summary(matrix: dict) -> dict:
    count = len(matrix["R"])
    final = matrix["metrics_by_stage"][str(count - 1)]
    return {
        "final_average_performance": final["average_performance"],
        "average_forgetting": final["forgetting"],
        "immediate_performance": float(
            np.mean([matrix["R"][stage][stage] for stage in range(count)])
        ),
        "final_per_task_success": matrix["R"][-1],
    }


def verify_deterministic_parity(source_run: Path, parity_path: Path) -> None:
    expected = json.loads((source_run / "evaluation_matrix.json").read_text())
    actual = json.loads(parity_path.read_text())
    if actual["R"] != expected["R"]:
        raise ValueError("Standalone deterministic evaluation does not match source matrix.")


def evaluate_one(config, tasks, checkpoints, policies, action_mode, sampling_seed, progress):
    sequence = TaskSequence.from_names(tasks)
    matrix = EvaluationMatrix(sequence)
    evaluation = config["evaluation"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sampling_seed)
        for stage, policy in enumerate(policies):
            evaluate_stage(
                EvaluationActionPolicy(policy, action_mode), matrix, stage,
                episodes=evaluation["episodes"], seed=evaluation["seed"],
                evaluation_mode=evaluation["mode"],
                task_set_seed=evaluation["task_set_seed"],
                reward_function_version=config["environment"]["reward_function_version"],
                backend=config["environment"]["backend"], progress=progress,
            )
            matrix.stage_provenance[str(stage)] = {
                "checkpoint": str(checkpoints[stage]),
                "checkpoint_sha256": file_hash(checkpoints[stage]),
                "policy_objective": policies[stage].objective,
                "action_mode": action_mode,
                "sampling_seed": sampling_seed,
            }
    result = matrix.to_dict()
    result.update(
        source_run=str(Path(config["runtime"]["output"]).resolve()),
        action_mode=action_mode,
        sampling_seed=sampling_seed,
        protocol=(
            "fixed task bank; unchanged environment reset seeds; any-step success; "
            f"GeneralPolicy action mode={action_mode}"
        ),
    )
    result["summary"] = matrix_summary(result)
    return result


def aggregate(results: list[dict]) -> dict:
    summaries = [result["summary"] for result in results]
    keys = ("final_average_performance", "average_forgetting", "immediate_performance")
    output = {
        key: {"mean": float(np.mean([row[key] for row in summaries])),
              "std": float(np.std([row[key] for row in summaries]))}
        for key in keys
    }
    final = np.asarray([row["final_per_task_success"] for row in summaries])
    output["final_per_task_success"] = {
        "mean": final.mean(axis=0).tolist(), "std": final.std(axis=0).tolist()
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--action-mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deterministic-parity", type=Path)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    config, tasks, checkpoints, policies = load_run(args.source_run)
    diagnostics = uncertainty_diagnostics(config, tasks, policies)
    print(json.dumps(diagnostics, indent=2))
    if args.diagnostics_only:
        return
    if args.output is None:
        parser.error("--output is required unless --diagnostics-only is used.")
    if args.output.exists():
        parser.error("Output directory must not already exist.")
    if args.episodes != config["evaluation"]["episodes"]:
        parser.error("Episodes must match the source evaluation protocol.")
    if len(set(args.sampling_seeds)) != len(args.sampling_seeds):
        parser.error("Sampling seeds must be unique.")
    if args.action_mode == "stochastic":
        if args.deterministic_parity is None:
            parser.error("Stochastic evaluation requires --deterministic-parity.")
        verify_deterministic_parity(args.source_run.resolve(), args.deterministic_parity)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in args.sampling_seeds:
        result = evaluate_one(
            config, tasks, checkpoints, policies, args.action_mode, seed, args.progress
        )
        directory = args.output / f"sampling_seed_{seed}"
        directory.mkdir()
        write_json(directory / "evaluation_matrix.json", result)
        results.append(result)
    report = {
        "source_run": str(args.source_run.resolve()),
        "checkpoint_sha256": {str(path): file_hash(path) for path in checkpoints},
        "policy_objective": SQUASHED_GAUSSIAN_NLL,
        "action_mode": args.action_mode,
        "sampling_seeds": args.sampling_seeds,
        "episodes_per_task": args.episodes,
        "uncertainty_diagnostics": diagnostics,
        "per_seed": [{"sampling_seed": r["sampling_seed"], **r["summary"]} for r in results],
        "aggregate": aggregate(results),
        "sampling_seed_scope": "evaluation randomness only; not independent training seeds",
    }
    write_json(args.output / "summary.json", report)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
