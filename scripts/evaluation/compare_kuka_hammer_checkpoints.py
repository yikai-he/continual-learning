#!/usr/bin/env python3
"""Compare KUKA Hammer SAC checkpoints on one identical fixed task bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stable_baselines3 import SAC

from src.envs import make_env
from src.hammer_diagnostics import HammerEpisodeTracker, summarize_hammer_episodes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--episodes", type=int, choices=range(1, 51), default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--task-set-seed", type=int, default=10_000)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("original", "nail_progress"),
        help="One reward variant per checkpoint (default: original for all).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="One nail-progress weight per checkpoint (default: zero for all).",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args(argv)


def find_task_wrapper(env):
    current = env
    while current is not None:
        if hasattr(current, "tasks") and hasattr(
            current, "toggle_sample_tasks_on_reset"
        ):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("KUKA environment has no task-selection wrapper.")


def task_identity(task) -> str:
    return hashlib.sha256(task.data).hexdigest()


def evaluate_checkpoint(model, env, tasks, seeds) -> dict:
    returns, successes, episodes = [], [], []
    wrapper = find_task_wrapper(env)
    wrapper.toggle_sample_tasks_on_reset(False)
    for task, seed in zip(tasks, seeds):
        env.unwrapped.set_task(task)
        observation, _ = env.reset(seed=seed)
        tracker = HammerEpisodeTracker(env)
        tracker.sample(0)
        episode_return = 0.0
        success = False
        for timestep in range(1, env.spec.max_episode_steps + 1):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            success |= bool(info.get("success", info.get("is_success", False)))
            tracker.sample(timestep)
            if terminated or truncated:
                break
        returns.append(episode_return)
        successes.append(success)
        episodes.append(tracker.finish(success).to_dict())
    summary = summarize_hammer_episodes(episodes)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        **summary,
        "episode_returns": returns,
        "episodes": episodes,
    }


def table_row(label: str, result: dict) -> dict:
    return {
        "checkpoint": label,
        "success_rate": result["success_rate"],
        "mean_return": result["mean_return"],
        "mean_final_nail_qpos": result["hammer_mean_final_nail_qpos"],
        "mean_max_nail_qpos": result["hammer_mean_max_nail_qpos"],
        "median_max_nail_qpos": result["hammer_median_max_nail_qpos"],
        "best_max_nail_qpos": result["hammer_best_max_nail_qpos"],
        "fraction_ge_005": result["hammer_fraction_ge_005"],
        "fraction_ge_008": result["hammer_fraction_ge_008"],
        "fraction_gt_009": result["hammer_fraction_success_threshold"],
        "head_nail_contact_fraction": result[
            "hammer_head_nail_contact_fraction"
        ],
    }


def print_table(rows: list[dict]) -> None:
    headings = (
        "Checkpoint", "Success", "Return", "Final q", "Mean max", "Median max",
        "Best max", ">=.05", ">=.08", ">.09", "Contact",
    )
    print(" | ".join(headings))
    print(" | ".join("---" for _ in headings))
    for row in rows:
        print(
            f"{row['checkpoint']} | {row['success_rate']:.1%} | "
            f"{row['mean_return']:.2f} | {row['mean_final_nail_qpos']:.4f} | "
            f"{row['mean_max_nail_qpos']:.4f} | "
            f"{row['median_max_nail_qpos']:.4f} | "
            f"{row['best_max_nail_qpos']:.4f} | "
            f"{row['fraction_ge_005']:.1%} | {row['fraction_ge_008']:.1%} | "
            f"{row['fraction_gt_009']:.1%} | "
            f"{row['head_nail_contact_fraction']:.1%}"
        )


def main(argv=None) -> None:
    args = parse_args(argv)
    variants = args.variants or ["original"] * len(args.checkpoints)
    weights = args.weights or [0.0] * len(args.checkpoints)
    if len(variants) != len(args.checkpoints) or len(weights) != len(args.checkpoints):
        raise ValueError("--variants and --weights must match the checkpoint count.")
    if any(weight < 0 for weight in weights):
        raise ValueError("--weights must be nonnegative.")
    bank_env = make_env("kuka-v3", "kuka-hammer-v3", args.task_set_seed)
    try:
        wrapper = find_task_wrapper(bank_env)
        tasks = tuple(wrapper.tasks[: args.episodes])
        seeds = tuple(range(args.seed, args.seed + args.episodes))
        identities = tuple(task_identity(task) for task in tasks)
        rows, details = [], []
        for checkpoint, variant, weight in zip(args.checkpoints, variants, weights):
            path = checkpoint.expanduser().resolve()
            model = SAC.load(path, device="auto")
            env = make_env(
                "kuka-v3",
                "kuka-hammer-v3",
                args.task_set_seed,
                hammer_reward_variant=variant,
                hammer_nail_progress_weight=weight,
            )
            try:
                if model.observation_space != env.observation_space or model.action_space != env.action_space:
                    raise ValueError(f"Checkpoint spaces do not match KUKA Hammer: {path}")
                result = evaluate_checkpoint(model, env, tasks, seeds)
            finally:
                env.close()
            label = f"{path.name} [{variant}, w={weight:g}]"
            rows.append(table_row(label, result))
            details.append(
                {
                    "checkpoint": str(path),
                    "reward_variant": variant,
                    "nail_progress_weight": weight,
                    **result,
                }
            )
        print_table(rows)
        report = {
            "task": "kuka-hammer-v3",
            "deterministic": True,
            "task_set_seed": args.task_set_seed,
            "evaluation_seeds": list(seeds),
            "ordered_task_sha256": list(identities),
            "rows": rows,
            "details": details,
        }
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.csv:
            with args.csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
    finally:
        bank_env.close()


if __name__ == "__main__":
    main()
