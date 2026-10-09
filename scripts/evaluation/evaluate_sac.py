"""Evaluate a saved SAC policy on a MetaWorld MT1 task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stable_baselines3 import SAC

from src.continual.evaluation import evaluate_task
from src.continual.policy_adapter import SB3SACPolicy
from src.continual.task_sequence import TaskSpec
from src.support.expert_manifest import qualification_record
from src.support.reproducibility import file_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--env-backend",
        choices=("metaworld-v3", "kuka-v2"),
        default="metaworld-v3",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--evaluation-mode", choices=("fixed-tasks", "sampled"), default="fixed-tasks"
    )
    parser.add_argument("--task-set-seed", type=int, default=10_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward-function-version", choices=("v1", "v2"), default="v2")
    return parser.parse_args()


def resolve_model_path(path: Path) -> Path:
    candidates = [path]
    if path.suffix != ".zip":
        candidates.append(path.with_suffix(".zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"SAC model not found: {path} (also checked .zip suffix)")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    model_path = resolve_model_path(args.model)
    if args.env_backend != "metaworld-v3" and args.evaluation_mode == "fixed-tasks":
        raise ValueError("fixed-tasks evaluation is specific to MetaWorld-v3.")
    try:
        model = SAC.load(model_path, device="auto")
    except Exception as exc:
        raise RuntimeError(f"Could not load SAC model {model_path}: {exc}") from exc
    result = evaluate_task(
        SB3SACPolicy(model),
        TaskSpec(args.task, args.task),
        episodes=args.episodes,
        seed=args.seed,
        evaluation_mode=args.evaluation_mode,
        task_set_seed=args.task_set_seed,
        reward_function_version=args.reward_function_version,
        backend=args.env_backend,
        render_mode="human" if args.render else None,
    )
    task_hashes = [
        item["task_data_sha256"] for item in result.get("episode_provenance") or []
    ]
    qualification = qualification_record(
        success_rate=result["success_rate"],
        mean_return=result["mean_return"],
        return_std=result["std_return"],
        episodes=result["episodes"],
        task_set_seed=args.task_set_seed,
        evaluation_seed=args.seed,
        deterministic=True,
        task_hashes=task_hashes,
    )
    report = {
        "kind": "expert_qualification_v1",
        "task_name": args.task,
        "checkpoint": str(model_path),
        "checkpoint_sha256": file_hash(model_path),
        "backend": args.env_backend,
        "algorithm": "SAC",
        "reward_function_version": args.reward_function_version,
        "qualification": qualification,
    }
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(f"Task: {args.task}")
    print(f"Protocol: {result['evaluation_mode']}")
    print(f"Task-set seed: {result['task_set_seed']}")
    print(f"Episodes: {result['episodes']}")
    print(f"Mean return: {result['mean_return']:.6f}")
    print(f"Return standard deviation: {result['std_return']:.6f}")
    print(f"Success rate: {result['success_rate']:.6f}")
    print(f"Mean episode length: {result['mean_episode_length']:.2f}")


if __name__ == "__main__":
    main()
