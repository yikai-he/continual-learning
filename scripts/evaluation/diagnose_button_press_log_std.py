#!/usr/bin/env python3
"""Controlled Button Press initial-log-std ablation for stochastic NLL BC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.diagnose_button_press_bc import (  # noqa: E402
    DEFAULT_PILOT,
    TASK,
    file_sha256,
    metric_at,
    stochastic_from_state,
    tensor_dataset,
    train_stochastic,
)
from src.continual.bc_policy import GeneralPolicy  # noqa: E402
from continual_manipulation.evaluation import evaluate_task  # noqa: E402
from continual_manipulation.sequences import TaskSequence  # noqa: E402
from src.support.io import write_json  # noqa: E402

DEFAULT_BASELINE = ROOT / "runs/diagnostics/button_press_bc_mse_vs_nll_seed0"
DEFAULT_OUTPUT = ROOT / "runs/diagnostics/button_press_log_std_ablation_seed0"
EXPECTED_DATASET_SHA256 = "a8a80902b0f40d942317b1e70c4eb308dcb67ed210236cc4973d7660d9c3be82"
EXPECTED_TRAIN_IDS = [28, 16, 23, 6, 18, 25, 3, 29, 8, 0, 19, 12, 20, 13, 7, 5, 17, 14, 22, 9, 27, 24, 1, 15]
EXPECTED_VALIDATION_IDS = [2, 11, 26, 21, 10, 4]


def initialized_policy(low, high, seed: int, bias: float) -> GeneralPolicy:
    torch.manual_seed(seed)
    policy = GeneralPolicy(low, high)
    with torch.no_grad():
        policy.log_std_head.bias.fill_(bias)
    return policy


def compare_initializations(reference, candidate, expected_bias: float) -> dict:
    ref, other = reference.state_dict(), candidate.state_dict()
    differing = [name for name in ref if not torch.equal(ref[name], other[name])]
    expected = ["log_std_head.bias"] if expected_bias != -3 else []
    if differing != expected:
        raise RuntimeError(f"Unexpected initialization differences: {differing}")
    if not torch.all(candidate.log_std_head.bias == expected_bias):
        raise RuntimeError("Requested log-std bias was not installed exactly.")
    return {
        "exactly_equal_except_log_std_head_bias": differing == expected,
        "differing_state_dict_keys": differing,
        "shared_trunk_exact": all(
            torch.equal(ref[name], other[name]) for name in ref if name.startswith("shared_network.")
        ),
        "mean_head_exact": all(
            torch.equal(ref[name], other[name]) for name in ref if name.startswith("mean_head.")
        ),
        "log_std_weight_exact": torch.equal(ref["log_std_head.weight"], other["log_std_head.weight"]),
        "reference_bias": ref["log_std_head.bias"].tolist(),
        "candidate_bias": other["log_std_head.bias"].tolist(),
    }


def saturation_summary(history: list[dict]) -> dict:
    first_over_half = next(
        (row["epoch"] for row in history if row["validation"]["upper_clamp_fraction"] > 0.5),
        None,
    )
    first_full = next(
        (row["epoch"] for row in history if row["validation"]["upper_clamp_fraction"] == 1.0),
        None,
    )
    first_zero_gradient = next(
        (row["epoch"] for row in history if row["gradients"]["log_std_head"] == 0.0),
        None,
    )
    mse_at_full = (
        history[first_full - 1]["validation"]["physical_action_mse"] if first_full else None
    )
    later_best = (
        min(row["validation"]["physical_action_mse"] for row in history[first_full:])
        if first_full and first_full < len(history)
        else None
    )
    return {
        "first_epoch_upper_clamp_fraction_over_50_percent": first_over_half,
        "first_epoch_upper_clamp_fraction_100_percent": first_full,
        "first_epoch_log_std_head_gradient_zero": first_zero_gradient,
        "validation_mse_at_full_saturation": mse_at_full,
        "best_validation_mse_after_full_saturation": later_best,
        "mean_action_mse_improved_after_full_saturation": (
            later_best < mse_at_full if later_best is not None else None
        ),
    }


def selected_result(history, selection, evaluation):
    epoch = selection[1]
    return {
        "selected_epoch": epoch,
        **metric_at(history, epoch),
        **{
            key: evaluation[key]
            for key in (
                "success_rate",
                "success_count",
                "mean_return",
                "std_return",
                "mean_episode_length",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    pilot, baseline, output = args.pilot.resolve(), args.baseline.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")

    resolved = json.loads((pilot / "config.json").read_text())
    stage = pilot / "stage_0_button-press-v3"
    report = json.loads((stage / "report.json").read_text())
    dataset_path = stage / "data_task_0.npy"
    if file_sha256(dataset_path) != EXPECTED_DATASET_SHA256:
        raise ValueError("Stage-1 dataset identity does not match the audited pilot.")
    packed_array = np.load(dataset_path)
    if packed_array.shape != (30, 200, 43):
        raise ValueError(f"Unexpected persisted dataset shape: {packed_array.shape}")
    split = report["splits"][TASK]
    if split["train_ids"] != EXPECTED_TRAIN_IDS or split["validation_ids"] != EXPECTED_VALIDATION_IDS:
        raise ValueError("Stage-1 split identity does not match the audited pilot.")
    packed = torch.from_numpy(packed_array).float()
    train = tensor_dataset(packed, EXPECTED_TRAIN_IDS)
    validation = tensor_dataset(packed, EXPECTED_VALIDATION_IDS)

    baseline_config = json.loads((baseline / "resolved_config.json").read_text())
    baseline_comparison = json.loads((baseline / "comparison_report.json").read_text())
    baseline_history = json.loads((baseline / "stochastic_history.json").read_text())
    baseline_evaluations = json.loads((baseline / "evaluation_results.json").read_text())
    if baseline_config["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise ValueError("Existing -3 baseline used a different dataset.")
    low = np.asarray(baseline_config["action_low"], dtype=np.float32)
    high = np.asarray(baseline_config["action_high"], dtype=np.float32)
    config = {
        "kind": "button_press_initial_log_std_ablation_v1",
        "task": TASK,
        "backend": resolved["environment_backend"],
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "train_ids": EXPECTED_TRAIN_IDS,
        "validation_ids": EXPECTED_VALIDATION_IDS,
        "training_seed": resolved["seed"],
        "epochs": resolved["bc"]["epochs"],
        "batch_size": resolved["bc"]["batch_size"],
        "learning_rate": resolved["bc"]["learning_rate"],
        "evaluation_episodes": resolved["eval_episodes"],
        "evaluation_seed": resolved["eval_seed"],
        "evaluation_mode": resolved["evaluation_mode"],
        "task_set_seed": resolved["task_set_seed"],
        "reward_function_version": "v2",
        "objective": resolved["bc"]["objective"],
        "optimizer": "Adam; fresh per run; no gradient clipping",
        "only_experimental_variable": "initial log_std_head.bias",
        "baseline_reference": str(baseline),
    }
    for key in ("training_seed", "epochs", "batch_size", "learning_rate", "evaluation_episodes", "evaluation_seed", "evaluation_mode", "task_set_seed"):
        if config[key] != baseline_config[key]:
            raise ValueError(f"Ablation configuration differs from baseline for {key}.")

    reference = initialized_policy(low, high, config["training_seed"], -3)
    candidates = {
        "initial_log_std_minus1": (-1.0, initialized_policy(low, high, config["training_seed"], -1)),
        "initial_log_std_zero": (0.0, initialized_policy(low, high, config["training_seed"], 0)),
    }
    initialization = {
        name: compare_initializations(reference, policy, bias)
        for name, (bias, policy) in candidates.items()
    }

    sequence = TaskSequence.from_names([TASK])
    run_outputs = {}
    for name, (bias, policy) in candidates.items():
        history, best_nll, best_mse, final_state = train_stochastic(
            policy, train, validation, config
        )
        best_nll_policy = stochastic_from_state(low, high, best_nll[2])
        best_mse_policy = stochastic_from_state(low, high, best_mse[2])
        evaluations = {}
        for selection_name, selected_policy in (
            ("best_nll", best_nll_policy),
            ("best_mse", best_mse_policy),
        ):
            evaluations[selection_name] = evaluate_task(
                selected_policy,
                sequence.task(0),
                episodes=config["evaluation_episodes"],
                seed=config["evaluation_seed"],
                evaluation_mode=config["evaluation_mode"],
                task_set_seed=config["task_set_seed"],
                reward_function_version=config["reward_function_version"],
                backend=config["backend"],
                progress=args.progress,
            )
        run_outputs[name] = {
            "bias": bias,
            "history": history,
            "best_nll": best_nll,
            "best_mse": best_mse,
            "final_state": final_state,
            "best_nll_policy": best_nll_policy,
            "best_mse_policy": best_mse_policy,
            "evaluations": evaluations,
            "saturation": saturation_summary(history),
        }

    baseline_hashes = baseline_evaluations["stochastic_nll_best"]["ordered_task_hashes"]
    for run in run_outputs.values():
        for evaluation in run["evaluations"].values():
            if evaluation["ordered_task_hashes"] != baseline_hashes:
                raise RuntimeError("Evaluation task bank differs from the -3 baseline.")

    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", {**config, "initialization_comparisons": initialization})
    comparison = {
        "initial_log_std_minus3": {
            "source": str(baseline),
            "best_nll": baseline_comparison["B1_stochastic_nll_best"],
            "best_mse": baseline_comparison["B2_stochastic_mse_best"],
            "initial_gradients": baseline_history[0]["gradients"],
            "saturation": saturation_summary(baseline_history),
        }
    }
    for name, run in run_outputs.items():
        directory = output / name
        directory.mkdir()
        write_json(directory / "training_history.json", run["history"])
        run["best_nll_policy"].save(
            directory / "best_nll_policy.pt",
            metadata={"initial_log_std_bias": run["bias"], "epoch": run["best_nll"][1], "selection": "validation_nll"},
        )
        run["best_mse_policy"].save(
            directory / "best_mse_policy.pt",
            metadata={"initial_log_std_bias": run["bias"], "epoch": run["best_mse"][1], "selection": "validation_physical_action_mse"},
        )
        evaluation_report = {
            "best_nll": selected_result(run["history"], run["best_nll"], run["evaluations"]["best_nll"]),
            "best_mse": selected_result(run["history"], run["best_mse"], run["evaluations"]["best_mse"]),
            "saturation": run["saturation"],
            "initialization": initialization[name],
        }
        write_json(directory / "evaluation.json", evaluation_report)
        comparison[name] = {
            **evaluation_report,
            "initial_gradients": run["history"][0]["gradients"],
        }
    write_json(output / "comparison_report.json", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
