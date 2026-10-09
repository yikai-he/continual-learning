#!/usr/bin/env python3
"""Probe KUKA-v3 pick-place reaching with a deterministic scripted controller."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.envs import make_env  # noqa: E402

TASK = "kuka-pick-place-v3"
BACKEND = "kuka-v3"
REWARD_VERSION = "v2"
SEEDS = tuple(range(10_000, 10_020))
HORIZON = 200
NEAR_OBJECT_DISTANCE = 0.03
HOVER_HEIGHT = 0.08
APPROACH_HEIGHT = 0.025
POSITION_TOLERANCE = 0.002
OBJECT_MOTION_TOLERANCE = 0.001
OPEN_GRIPPER_ACTION = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/reward_v2/kuka-pick-place-v3/seed_0_short_50k/"
            "diagnostics/reaching"
        ),
    )
    return parser.parse_args()


def state_record(
    seed: int,
    step: int,
    phase: str,
    observation: np.ndarray,
    reward: float,
    info: dict[str, Any],
    action: np.ndarray,
    task_env: Any,
    initial_object: np.ndarray,
) -> dict[str, Any]:
    tcp = np.asarray(task_env.tcp_center, dtype=float).copy()
    obj = np.asarray(observation[4:7], dtype=float).copy()
    return {
        "seed": seed,
        "step": step,
        "phase": phase,
        "tcp_x": tcp[0],
        "tcp_y": tcp[1],
        "tcp_z": tcp[2],
        "object_x": obj[0],
        "object_y": obj[1],
        "object_z": obj[2],
        "tcp_to_object": float(np.linalg.norm(tcp - obj)),
        "reward": float(reward),
        "near_object": bool(info["near_object"]),
        "grasp_reward": float(info["grasp_reward"]),
        "in_place_reward": float(info["in_place_reward"]),
        "obj_to_target": float(info["obj_to_target"]),
        "success": bool(info["success"]),
        "action_x": float(action[0]),
        "action_y": float(action[1]),
        "action_z": float(action[2]),
        "action_gripper": float(action[3]),
        "object_displacement": float(np.linalg.norm(obj - initial_object)),
        "object_contact": bool(task_env.touching_main_object),
    }


def current_reward_info(
    task_env: Any, observation: np.ndarray, action: np.ndarray
) -> tuple[float, dict[str, Any]]:
    reward, info = task_env.evaluate_state(observation, action)
    return float(reward), info


def controller_action(task_env: Any, obj: np.ndarray, phase: str) -> np.ndarray:
    tcp = np.asarray(task_env.tcp_center, dtype=float)
    if phase == "hover":
        target = obj + np.array([0.0, 0.0, HOVER_HEIGHT])
    else:
        target = obj + np.array([0.0, 0.0, APPROACH_HEIGHT])
    # set_xyz_action applies action * action_scale; compensate only for that
    # public action interpretation, while retaining normal clipping and dynamics.
    xyz = np.clip((target - tcp) / float(task_env.action_scale), -1.0, 1.0)
    return np.asarray([xyz[0], xyz[1], xyz[2], OPEN_GRIPPER_ACTION], dtype=np.float32)


def run_episode(env: Any, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observation, _ = env.reset(seed=seed)
    task_env = env.unwrapped
    initial_object = np.asarray(observation[4:7], dtype=float).copy()
    open_action = np.array([0.0, 0.0, 0.0, OPEN_GRIPPER_ACTION], dtype=np.float32)
    initial_reward, initial_info = current_reward_info(task_env, observation, open_action)
    records = [
        state_record(
            seed, -1, "initial", observation, initial_reward, initial_info,
            open_action, task_env, initial_object
        )
    ]
    phase = "hover"
    threshold_step: int | None = None

    for step in range(HORIZON):
        obj = np.asarray(observation[4:7], dtype=float)
        tcp = np.asarray(task_env.tcp_center, dtype=float)
        hover_target = obj + np.array([0.0, 0.0, HOVER_HEIGHT])
        if phase == "hover" and np.linalg.norm(tcp - hover_target) <= POSITION_TOLERANCE:
            phase = "approach"
        action = controller_action(task_env, obj, phase)
        observation, reward, terminated, truncated, info = env.step(action)
        record = state_record(
            seed, step, phase, observation, reward, info, action, task_env,
            initial_object
        )
        records.append(record)
        if threshold_step is None and record["tcp_to_object"] <= NEAR_OBJECT_DISTANCE:
            threshold_step = step + 1
            break
        if terminated or truncated:
            break

    closest = min(records, key=lambda row: row["tcp_to_object"])
    approach_records = records[1:] or records
    unexpected_motion = max(row["object_displacement"] for row in records) > OBJECT_MOTION_TOLERANCE
    unexpected_contact = any(row["object_contact"] for row in records)
    distance_decrease_pairs = [
        (previous, current)
        for previous, current in zip(records, records[1:])
        if current["tcp_to_object"] < previous["tcp_to_object"]
    ]
    summary = {
        "seed": seed,
        "initial_tcp_to_object_distance": records[0]["tcp_to_object"],
        "minimum_tcp_to_object_distance": closest["tcp_to_object"],
        "reached_le_3cm": threshold_step is not None,
        "steps_to_le_3cm": "" if threshold_step is None else threshold_step,
        "reward_at_initial_state": initial_reward,
        "maximum_reward_before_contact_grasp": max(
            row["reward"]
            for row in approach_records
            if not row["object_contact"] and row["grasp_reward"] < 0.97
        ),
        "reward_at_closest_point": closest["reward"],
        "reward_change_initial_to_closest": closest["reward"] - initial_reward,
        "reward_distance_spearman": spearman_correlation(
            [row["tcp_to_object"] for row in records],
            [row["reward"] for row in records],
        ),
        "fraction_closer_steps_with_non_decreasing_reward": sum(
            current["reward"] >= previous["reward"]
            for previous, current in distance_decrease_pairs
        ) / len(distance_decrease_pairs),
        "maximum_grasp_reward": max(row["grasp_reward"] for row in records),
        "in_place_reward_range": max(row["in_place_reward"] for row in records)
        - min(row["in_place_reward"] for row in records),
        "maximum_object_displacement": max(row["object_displacement"] for row in records),
        "unexpected_object_motion": unexpected_motion,
        "unexpected_contact": unexpected_contact,
        "failure_reason": "" if threshold_step is not None else failure_reason(records, task_env),
    }
    return summary, records


def failure_reason(records: list[dict[str, Any]], task_env: Any) -> str:
    last = records[-1]
    tcp = np.array([last["tcp_x"], last["tcp_y"], last["tcp_z"]])
    obj = np.array([last["object_x"], last["object_y"], last["object_z"]])
    desired = obj + np.array([0.0, 0.0, APPROACH_HEIGHT])
    clipped_axes = [
        axis for axis, value, low, high in zip("xyz", desired, task_env.hand_low, task_env.hand_high)
        if value < low or value > high
    ]
    if clipped_axes:
        return f"desired approach outside hand bounds on {','.join(clipped_axes)}"
    return f"controller did not converge within horizon; final target error={np.linalg.norm(tcp - desired):.4f} m"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spearman_correlation(x: list[float], y: list[float]) -> float:
    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=float)
        result[order] = np.arange(len(values), dtype=float)
        return result

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def report(
    episodes: list[dict[str, Any]], records: list[dict[str, Any]], policy_rows: dict[str, list[dict[str, str]]]
) -> str:
    reached = [row for row in episodes if row["reached_le_3cm"]]
    reward_changes = [float(row["reward_change_initial_to_closest"]) for row in episodes]
    median_correlation = float(
        np.median([row["reward_distance_spearman"] for row in episodes])
    )
    median_monotonic_fraction = float(
        np.median(
            [row["fraction_closer_steps_with_non_decreasing_reward"] for row in episodes]
        )
    )
    median_delta = float(np.median(reward_changes))
    useful_gradient = median_delta >= 0.05 and median_correlation <= -0.5
    reliable = len(reached) == len(episodes)
    classification = (
        "SAC exploration failure likely" if reliable and useful_gradient
        else "weak reaching reward signal likely" if reliable
        else "workspace/control issue likely" if not reached
        else "mixed cause"
    )
    failed = [row for row in episodes if not row["reached_le_3cm"]]

    lines = [
        "# KUKA Pick Place Scripted Reaching Diagnostic",
        "",
        "Deterministic scripted control on seeds 10000–10019, horizon 200, reward v2, "
        "and `terminate_on_success=False`. The controller uses only normal environment "
        "actions, keeps the gripper open, and stops at the official 3 cm near-object threshold.",
        "",
        "## Result",
        "",
        f"- Success reaching ≤3 cm: **{len(reached)}/20**",
        f"- Median initial distance: **{np.median([r['initial_tcp_to_object_distance'] for r in episodes]):.4f} m**",
        f"- Median minimum distance: **{np.median([r['minimum_tcp_to_object_distance'] for r in episodes]):.4f} m**",
        f"- Median steps to ≤3 cm: **{np.median([r['steps_to_le_3cm'] for r in reached]):.1f}**" if reached else "- Median steps to ≤3 cm: **N/A**",
        f"- Median reward increase during approach: **{median_delta:.6f}**",
        f"- Median per-episode reward/distance Spearman correlation: **{median_correlation:.4f}** (negative means reward rises as distance falls)",
        f"- Median fraction of closer steps with non-decreasing reward: **{median_monotonic_fraction:.1%}**",
        f"- Maximum grasp reward across episodes: **{max(r['maximum_grasp_reward'] for r in episodes):.6f}**",
        f"- Maximum in-place reward range within an episode: **{max(r['in_place_reward_range'] for r in episodes):.6g}**",
        f"- Unexpected contact episodes: **{sum(r['unexpected_contact'] for r in episodes)}/20**",
        f"- Unexpected object-motion episodes (>1 mm): **{sum(r['unexpected_object_motion'] for r in episodes)}/20**",
        "",
        "## Interpretation",
        "",
        f"The reward **{'does' if useful_gradient else 'does not'} provide a useful reaching gradient** by the diagnostic criterion (median increase ≥0.05 and median per-episode correlation ≤−0.5). Its relationship to distance is {'directionally consistent' if median_monotonic_fraction >= 0.8 else 'not reliably monotonic'} during approach.",
        f"Workspace/action scaling **{'does not prevent reliable reaching' if reliable else 'prevents reliable reaching on some configurations'}**.",
        "`in_place_reward` tracks object-to-target distance, so it is effectively independent of TCP approach while the object remains stationary.",
        "`grasp_reward` is the only pre-grasp component that changes with the TCP; the total reward is its Hamacher product with `in_place_reward`.",
        "",
        "## Saved-policy comparison",
        "",
    ]
    for name in ("best_model", "final_model"):
        rows = policy_rows[name]
        lines.append(
            f"- {name}: near object in {sum(r['ever_near_object'] == 'True' for r in rows)}/20 episodes; "
            f"median closest distance {np.median([float(r['minimum_tcp_to_object_distance']) for r in rows]):.4f} m."
        )
    lines.extend(["", "## Failed seeds", ""])
    if failed:
        lines.extend(f"- {row['seed']}: {row['failure_reason']}" for row in failed)
    else:
        lines.append("None.")
    lines.extend(["", "## Classification", "", f"**{classification}**", ""])
    return "\n".join(lines)


def read_policy_metrics(diagnostics_dir: Path) -> dict[str, list[dict[str, str]]]:
    result = {}
    for name in ("best_model", "final_model"):
        with (diagnostics_dir / f"{name}_episode_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as source:
            result[name] = list(csv.DictReader(source))
    return result


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(BACKEND, TASK, SEEDS[0], reward_function_version=REWARD_VERSION)
    episodes: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            episode, steps = run_episode(env, seed)
            episodes.append(episode)
            records.extend(steps)
    finally:
        env.close()

    write_csv(output_dir / "scripted_reaching_metrics.csv", episodes)
    write_csv(output_dir / "reward_vs_distance.csv", records)
    policy_rows = read_policy_metrics(output_dir.parent)
    (output_dir / "REACHING_DIAGNOSTIC_SUMMARY.md").write_text(
        report(episodes, records, policy_rows), encoding="utf-8"
    )
    print(report(episodes, records, policy_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
