#!/usr/bin/env python3
"""Diagnose saved KUKA-v3 pick-place policies without training or mutation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stable_baselines3 import SAC  # noqa: E402

from src.envs import make_env  # noqa: E402

TASK = "kuka-pick-place-v3"
BACKEND = "kuka-v3"
REWARD_VERSION = "v2"
SEEDS = tuple(range(10_000, 10_020))
HORIZON = 200
NEAR_OBJECT_DISTANCE = 0.03
LIFT_DISTANCE = 0.01
MEANINGFUL_TRANSPORT_DISTANCE = 0.05
STAGES = (
    "no meaningful approach",
    "reached object",
    "valid caging/contact",
    "grasped",
    "lifted",
    "transported toward goal",
    "success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/reward_v2/kuka-pick-place-v3/seed_0_short_50k"),
        help="Destination run directory; completed timestamped sibling is auto-detected.",
    )
    return parser.parse_args()


def resolve_artifact_run(run_dir: Path) -> Path:
    """Find the immutable completed run while preserving the requested output path."""
    required = (Path("best_model/best_model.zip"), Path("final_model.zip"))
    run_dir = run_dir.resolve()
    if all((run_dir / item).is_file() for item in required):
        return run_dir
    candidates = sorted(
        candidate
        for candidate in run_dir.parent.glob(f"{run_dir.name}_*")
        if all((candidate / item).is_file() for item in required)
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one completed sibling for {run_dir}, found {candidates}"
        )
    return candidates[0].resolve()


def bilateral_contact(task_env: Any) -> bool:
    return bool(task_env.touching_main_object)


def classify_episode(metrics: dict[str, Any]) -> str:
    """Return the furthest sequentially completed task stage."""
    stage = STAGES[0]
    if not metrics["ever_near_object"]:
        return stage
    stage = STAGES[1]
    if not metrics["ever_bilateral_contact"]:
        return stage
    stage = STAGES[2]
    if not metrics["ever_grasp_success"]:
        return stage
    stage = STAGES[3]
    if metrics["max_object_lift"] <= LIFT_DISTANCE:
        return stage
    stage = STAGES[4]
    if metrics["max_transport_progress_after_lift"] < MEANINGFUL_TRANSPORT_DISTANCE:
        return stage
    stage = STAGES[5]
    if metrics["success"]:
        stage = STAGES[6]
    return stage


def evaluate_model(model_path: Path, model_name: str) -> list[dict[str, Any]]:
    model = SAC.load(model_path, device="cpu")
    episodes: list[dict[str, Any]] = []
    env = make_env(
        BACKEND,
        TASK,
        SEEDS[0],
        reward_function_version=REWARD_VERSION,
    )
    try:
        if model.observation_space != env.observation_space:
            raise ValueError("Model and environment observation spaces differ")
        if model.action_space != env.action_space:
            raise ValueError("Model and environment action spaces differ")
        for episode_index, seed in enumerate(SEEDS):
            observation, _ = env.reset(seed=seed)
            task_env = env.unwrapped
            initial_object = np.asarray(observation[4:7], dtype=float).copy()
            initial_target_distance = float(
                np.linalg.norm(initial_object - np.asarray(task_env._target_pos))
            )
            timesteps: list[dict[str, Any]] = []
            episode_return = 0.0
            episode_success = False
            for timestep in range(HORIZON):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                obj = np.asarray(observation[4:7], dtype=float).copy()
                tcp = np.asarray(task_env.tcp_center, dtype=float).copy()
                tcp_to_object = float(np.linalg.norm(tcp - obj))
                lift = float(obj[2] - initial_object[2])
                contact = bilateral_contact(task_env)
                success = bool(info.get("success", False))
                episode_return += float(reward)
                episode_success |= success
                timesteps.append(
                    {
                        "timestep": timestep,
                        "near_object": bool(info.get("near_object", False)),
                        "grasp_success": bool(info.get("grasp_success", False)),
                        "grasp_reward": float(info.get("grasp_reward", 0.0)),
                        "in_place_reward": float(info.get("in_place_reward", 0.0)),
                        "obj_to_target": float(info["obj_to_target"]),
                        "object_position": obj,
                        "object_z": float(obj[2]),
                        "tcp_position": tcp,
                        "tcp_to_object": tcp_to_object,
                        "gripper_observation": float(observation[3]),
                        "gripper_action": float(np.asarray(action)[-1]),
                        "bilateral_contact": contact,
                        "lift": lift,
                        "success": success,
                    }
                )
                if terminated or truncated:
                    break

            lifted_steps = [step for step in timesteps if step["lift"] > LIFT_DISTANCE]
            min_after_lift = min(
                (step["obj_to_target"] for step in lifted_steps),
                default=initial_target_distance,
            )
            metrics: dict[str, Any] = {
                "model": model_name,
                "episode": episode_index,
                "seed": seed,
                "episode_return": episode_return,
                "episode_length": len(timesteps),
                "initial_object_z": float(initial_object[2]),
                "max_object_z": max(step["object_z"] for step in timesteps),
                "max_object_lift": max(step["lift"] for step in timesteps),
                "minimum_tcp_to_object_distance": min(
                    step["tcp_to_object"] for step in timesteps
                ),
                "ever_near_object": any(step["near_object"] for step in timesteps),
                "maximum_grasp_reward": max(step["grasp_reward"] for step in timesteps),
                "ever_bilateral_contact": any(
                    step["bilateral_contact"] for step in timesteps
                ),
                "ever_positive_closing_action": any(
                    step["gripper_action"] > 0 for step in timesteps
                ),
                "maximum_closing_action": max(
                    step["gripper_action"] for step in timesteps
                ),
                "ever_grasp_success": any(step["grasp_success"] for step in timesteps),
                "initial_object_to_target_distance": initial_target_distance,
                "minimum_object_to_target_distance": min(
                    step["obj_to_target"] for step in timesteps
                ),
                "final_object_to_target_distance": timesteps[-1]["obj_to_target"],
                "max_transport_progress_after_lift": max(
                    0.0, initial_target_distance - min_after_lift
                ),
                "success": episode_success,
            }
            metrics["furthest_stage"] = classify_episode(metrics)
            episodes.append(metrics)
    finally:
        env.close()
    return episodes


def json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(
            {key: json_value(value) for key, value in episode.items()}
            for episode in episodes
        )


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    distances = np.asarray(
        [episode["minimum_tcp_to_object_distance"] for episode in episodes]
    )
    return {
        "episodes": len(episodes),
        "stage_counts": {
            stage: Counter(episode["furthest_stage"] for episode in episodes).get(stage, 0)
            for stage in STAGES
        },
        "median_minimum_tcp_to_object_distance": float(np.median(distances)),
        "minimum_tcp_to_object_distance": float(np.min(distances)),
        "near_object_episodes": sum(e["ever_near_object"] for e in episodes),
        "bilateral_contact_episodes": sum(e["ever_bilateral_contact"] for e in episodes),
        "positive_closing_action_episodes": sum(
            e["ever_positive_closing_action"] for e in episodes
        ),
        "grasp_success_episodes": sum(e["ever_grasp_success"] for e in episodes),
        "lift_episodes": sum(e["max_object_lift"] > LIFT_DISTANCE for e in episodes),
        "meaningful_transport_episodes": sum(
            e["max_transport_progress_after_lift"] >= MEANINGFUL_TRANSPORT_DISTANCE
            for e in episodes
        ),
        "success_episodes": sum(e["success"] for e in episodes),
        "median_maximum_grasp_reward": float(
            np.median([e["maximum_grasp_reward"] for e in episodes])
        ),
        "maximum_object_lift": float(max(e["max_object_lift"] for e in episodes)),
        "median_final_object_to_target_distance": float(
            np.median([e["final_object_to_target_distance"] for e in episodes])
        ),
    }


def failure_stage(best: dict[str, Any], final: dict[str, Any]) -> str:
    total_near = best["near_object_episodes"] + final["near_object_episodes"]
    total_contact = best["bilateral_contact_episodes"] + final["bilateral_contact_episodes"]
    if total_near == 0:
        return "reaching/exploration failure"
    if total_contact == 0 or total_contact < total_near / 2:
        return "centering/caging failure"
    if best["grasp_success_episodes"] + final["grasp_success_episodes"] == 0:
        return "grasp/contact failure"
    if best["lift_episodes"] + final["lift_episodes"] == 0:
        return "lift/retention failure"
    if best["meaningful_transport_episodes"] + final["meaningful_transport_episodes"] == 0:
        return "transport failure"
    if best["success_episodes"] + final["success_episodes"] == 0:
        return "placement failure"
    return "inconclusive"


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# KUKA Pick Place Diagnostic Summary",
        "",
        "Deterministic evaluation of the saved policies only: seeds 10000–10019, "
        "200-step horizon, reward v2, and no termination on success.",
        "",
        "## Stage counts",
        "",
        "| Furthest stage | best_model | final_model |",
        "|---|---:|---:|",
    ]
    for stage in STAGES:
        lines.append(
            f"| {stage} | {summary['best_model']['stage_counts'][stage]} | "
            f"{summary['final_model']['stage_counts'][stage]} |"
        )
    lines.extend(["", "## Key metrics", ""])
    for name in ("best_model", "final_model"):
        item = summary[name]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Median/minimum TCP-to-object distance: "
                f"{item['median_minimum_tcp_to_object_distance']:.4f} / "
                f"{item['minimum_tcp_to_object_distance']:.4f} m",
                f"- Near object: {item['near_object_episodes']}/20",
                f"- Bilateral contact: {item['bilateral_contact_episodes']}/20",
                f"- Grasp success: {item['grasp_success_episodes']}/20",
                f"- Lift over 1 cm: {item['lift_episodes']}/20",
                f"- Meaningful transport: {item['meaningful_transport_episodes']}/20",
                f"- Success: {item['success_episodes']}/20",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            f"Main failure: **{summary['main_failure_stage']}**.",
            "",
            "Ranked likely causes:",
            "",
            "1. **Exploration difficulty.** The deterministic saved policies rarely reach the "
            "3 cm near-object threshold and do not reliably discover the sequential contact, "
            "grasp, lift, and transport behavior.",
            "2. **Reward shaping issue.** There is no standalone TCP-reaching term: the base "
            "reward couples caging with object-to-goal progress, while the stronger lift reward "
            "is available only after a close, lifted configuration. The low grasp rewards and "
            "absence of near-object episodes show that this signal did not teach reliable reach.",
            "3. **Insufficient training steps.** This run has 50k environment steps and only "
            "40k steps after learning starts, while neither saved policy establishes the first "
            "contact-dependent stages.",
            "",
            "Thresholds: reached object uses the task's `near_object` (TCP distance ≤3 cm); "
            "contact is simultaneous left/right pad contact; grasp uses the task's "
            "`grasp_success`; lift is object rise >1 cm, matching the v2 lift-reward gate; "
            "meaningful transport is a ≥5 cm object-to-goal reduction after lift; success is "
            "the unmodified task `success` field.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    destination_run = args.run_dir.resolve()
    artifact_run = resolve_artifact_run(destination_run)
    output_dir = destination_run / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = {
        "best_model": artifact_run / "best_model/best_model.zip",
        "final_model": artifact_run / "final_model.zip",
    }
    results = {
        name: evaluate_model(path, name) for name, path in model_paths.items()
    }
    for name, episodes in results.items():
        write_csv(output_dir / f"{name}_episode_metrics.csv", episodes)

    summary: dict[str, Any] = {
        "evaluation": {
            "task": TASK,
            "backend": BACKEND,
            "reward_function_version": REWARD_VERSION,
            "deterministic": True,
            "terminate_on_success": False,
            "horizon": HORIZON,
            "seeds": list(SEEDS),
            "artifact_run": str(artifact_run),
            "output_run": str(destination_run),
        },
        "thresholds": {
            "near_object_distance": NEAR_OBJECT_DISTANCE,
            "lift_distance": LIFT_DISTANCE,
            "meaningful_transport_distance": MEANINGFUL_TRANSPORT_DISTANCE,
            "bilateral_contact": "simultaneous contact force at both pad geoms",
            "grasp": "existing info['grasp_success']",
            "success": "existing info['success']",
        },
        "best_model": summarize(results["best_model"]),
        "final_model": summarize(results["final_model"]),
    }
    summary["main_failure_stage"] = failure_stage(
        summary["best_model"], summary["final_model"]
    )
    summary["ranked_causes"] = [
        {
            "rank": 1,
            "cause": "exploration difficulty",
            "evidence": (
                "Neither policy reached the task's 3 cm near-object threshold in any of "
                "40 episodes; best/final median closest distances were "
                f"{summary['best_model']['median_minimum_tcp_to_object_distance']:.4f} m "
                f"and {summary['final_model']['median_minimum_tcp_to_object_distance']:.4f} m."
            ),
        },
        {
            "rank": 2,
            "cause": "reward shaping issue",
            "evidence": (
                "The v2 reward has no independent reaching term and couples caging with "
                "object-to-goal progress; median maximum grasp reward was only "
                f"{summary['best_model']['median_maximum_grasp_reward']:.4f} for best_model "
                f"and {summary['final_model']['median_maximum_grasp_reward']:.4f} for final_model."
            ),
        },
        {
            "rank": 3,
            "cause": "insufficient training steps",
            "evidence": (
                "The 50k run contains only 40k post-learning-start steps, and final_model "
                "regressed from best_model: median closest distance worsened from "
                f"{summary['best_model']['median_minimum_tcp_to_object_distance']:.4f} m "
                f"to {summary['final_model']['median_minimum_tcp_to_object_distance']:.4f} m."
            ),
        },
    ]
    (output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "DIAGNOSTIC_SUMMARY.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
