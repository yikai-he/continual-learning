"""Deterministic stage evaluation or independently trained expert cross-task audit.

R[k,j] is success after training stage k on evaluation task j. Unmeasured cells
are JSON null. Cross-task expert rows are NOT training stages or CL results.
"""

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from src.continual.evaluation import (
    EvaluationMatrix,
    evaluate_stage,
    evaluate_task,
)
from src.continual.task_sequence import PILOT_SEQUENCE, TaskSequence
from src.support.io import write_json
from src.support.reproducibility import file_hash


def load_policy(kind, path):
    """Load a BC checkpoint or wrap an SB3 SAC checkpoint for evaluation."""
    from src.continual.bc_policy import GeneralPolicy
    from src.continual.policy_adapter import SB3SACPolicy

    if kind == "bc":
        return GeneralPolicy.load(path)
    from stable_baselines3 import SAC

    return SB3SACPolicy(SAC.load(path, device="cpu"))


def main():
    """Evaluate one continual stage or build an expert cross-task diagnostic.

    Stage mode evaluates one continual ``R[k,j]`` row in a new or loaded matrix
    and writes the result to a new output file. Expert mode evaluates one
    independent model per task; its rows are a cross-task diagnostic, not
    sequential training stages or a continual-learning matrix.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=PILOT_SEQUENCE.task_names)
    parser.add_argument("--mode", choices=["stage", "experts"], default="stage")
    parser.add_argument("--policy-kind", choices=["sac", "bc"], default="sac")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--stage", type=int)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--include-future", action="store_true")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--evaluation-mode", choices=["fixed-tasks", "sampled"], default="fixed-tasks"
    )
    parser.add_argument("--task-set-seed", type=int, default=10000)
    parser.add_argument("--reward-function-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new file.")
    if args.episodes is not None and args.episodes < 1:
        parser.error("Need positive episodes.")
    sequence = TaskSequence.from_names(args.tasks)
    if args.mode == "stage" and (len(args.models) != 1 or args.stage is None):
        parser.error("Stage mode needs one model and --stage.")
    if args.mode == "experts" and (
        len(args.models) != sequence.num_tasks or args.matrix is not None
    ):
        parser.error("Expert mode needs one model per task and no stage matrix.")
    torch.set_num_threads(1)
    provenance = {str(p.resolve()): file_hash(p) for p in args.models}
    if args.mode == "stage":
        matrix = (
            EvaluationMatrix.load(args.matrix)
            if args.matrix
            else EvaluationMatrix(sequence)
        )
        if matrix.sequence != sequence:
            raise ValueError("Sequence differs from saved matrix.")
        evaluate_stage(
            load_policy(args.policy_kind, args.models[0]),
            matrix,
            args.stage,
            include_future=args.include_future,
            episodes=args.episodes,
            seed=args.seed,
            evaluation_mode=args.evaluation_mode,
            task_set_seed=args.task_set_seed,
            reward_function_version=args.reward_function_version,
        )
        matrix.stage_provenance[str(args.stage)] = {
            "models_sha256": provenance,
            "policy_kind": args.policy_kind,
            "evaluation_mode": args.evaluation_mode,
            "task_set_seed": args.task_set_seed,
        }
        report = matrix.to_dict()
    else:
        cells = []
        for path in args.models:
            policy = load_policy(args.policy_kind, path)
            cells.append(
                [
                    evaluate_task(
                        policy,
                        t,
                        episodes=args.episodes,
                        seed=args.seed,
                        evaluation_mode=args.evaluation_mode,
                        task_set_seed=args.task_set_seed,
                        reward_function_version=args.reward_function_version,
                    )
                    for t in sequence.tasks
                ]
            )
        report = {
            "kind": "independent_expert_cross_task_diagnostic_NOT_continual_learning",
            "tasks": sequence.task_names,
            "row_models": [str(p.resolve()) for p in args.models],
            "success_matrix": [[c["success_rate"] for c in row] for row in cells],
            "cells": cells,
        }
    report["model_sha256"] = provenance
    task_selection = (
        f"fixed task bank seed={args.task_set_seed}"
        if args.evaluation_mode == "fixed-tasks"
        else "task sampled by MetaWorld on reset"
    )
    report["protocol"] = (
        f"{args.evaluation_mode}; {task_selection}; reset(seed+i); deterministic; "
        "any-step success; horizon from env; return std ddof=0."
    )
    for path, digest in provenance.items():
        if file_hash(path) != digest:
            raise RuntimeError("Model input changed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report.get("R", report.get("success_matrix"))))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
