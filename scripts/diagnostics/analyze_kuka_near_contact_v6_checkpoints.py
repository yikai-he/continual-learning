#!/usr/bin/env python3
"""Read-only fixed-bank diagnosis of KUKA Near-Contact v6 SAC checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
import torch
from stable_baselines3 import SAC

from metaworld_kuka.envs.kuka_push_cube_handoff import (
    extract_handoff_state,
    handoff_violations,
)
from src.continual.evaluation import disable_task_sampling
from src.env_adapters.kuka_v3 import make_kuka_v3_env, kuka_v3_tasks

TASK_SET_SEED = 10_000
TASK_COUNT = 50
HORIZON = 200
REWARD_KEYS = ("reward_contact", "reward_push", "reward_progress", "reward_first_success")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def make_env():
    env = make_kuka_v3_env(
        "kuka-push-cube-v3", TASK_SET_SEED,
        reward_function_version="v6_rear_contact_push_progress",
        cube_push_curriculum_level=2,
        cube_push_initialization="near_contact",
        cube_y_min=0.502,
        near_contact_gap_m=0.005,
    )
    disable_task_sampling(env)
    return env


def cube_tilt(base: Any) -> float:
    matrix = np.asarray(base.data.geom("objGeom").xmat).reshape(3, 3)
    return math.acos(float(np.clip(matrix[2, 2], -1.0, 1.0)))


def max_robot_cube_penetration(base: Any) -> float:
    obj = base.model.geom("objGeom").id
    gripper = set(base._gripper_collision_geom_ids())
    return max(
        [max(0.0, -float(c.dist)) for c in base.data.contact
         if obj in (c.geom1, c.geom2) and bool({c.geom1, c.geom2} & gripper)],
        default=0.0,
    )


def gripper_table_contact(base: Any) -> bool:
    floor = base.model.geom("floor").id
    gripper = set(base._gripper_collision_geom_ids())
    return any(
        floor in (c.geom1, c.geom2) and bool({c.geom1, c.geom2} & gripper)
        for c in base.data.contact
    )


def goal_frame(cube: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = np.asarray(goal[:2] - cube[:2], dtype=float)
    unit /= max(float(np.linalg.norm(unit)), np.finfo(float).eps)
    return unit, np.array([unit[1], -unit[0]])


def episode(
    env: Any, task: Any, task_index: int,
    action_fn: Callable[[np.ndarray], np.ndarray], label: str,
) -> dict[str, Any]:
    base = env.unwrapped
    base.set_task(task)
    observation, _ = env.reset(seed=TASK_SET_SEED + task_index)
    cube0 = np.asarray(observation[4:7], dtype=float).copy()
    goal = np.asarray(observation[36:39], dtype=float).copy()
    tcp0 = np.asarray(observation[:3], dtype=float).copy()
    target0 = base._stage1_approach_target(cube0, goal)
    initial_goal_distance = float(np.linalg.norm(cube0 - goal))
    initial_tcp_cube = float(np.linalg.norm(tcp0 - cube0))
    initial_tcp_target = float(np.linalg.norm(tcp0 - target0))
    apertures = [float(base.gripper_aperture)]
    actions = []
    tcp_cube_distances = [initial_tcp_cube]
    tcp_target_distances = [initial_tcp_target]
    goal_distances = [initial_goal_distance]
    max_displacement = max_tilt = max_penetration = max_cube_angular_speed = 0.0
    max_tcp_speed = 0.0
    physical_steps = pad_steps = valid_steps = handoff_steps = 0
    first_physical = first_pad = first_valid = None
    loss_events = reacquisition_events = 0
    was_valid = False
    ever_valid = False
    clips = np.zeros(3, dtype=int)
    clipped_action_steps = 0
    table_collision_any = False
    component_totals = {key: 0.0 for key in REWARD_KEYS}
    observed_return = 0.0
    any_success = False
    previous_tcp = tcp0.copy()
    previous_valid = False
    dt = float(base.dt)
    offset_samples = []
    for step in range(1, HORIZON + 1):
        action = np.asarray(action_fn(observation), dtype=np.float32).reshape(4)
        actions.append(action.copy())
        proposed = np.asarray(base.data.mocap_pos[0]) + np.clip(action[:3], -1, 1) * base.action_scale
        clipped_axes = (proposed < base.mocap_low) | (proposed > base.mocap_high)
        clips += clipped_axes.astype(int)
        clipped_action_steps += int(np.any(clipped_axes))
        observation, reward, terminated, truncated, info = env.step(action)
        observed_return += float(reward)
        for key in REWARD_KEYS:
            component_totals[key] += float(info.get(key, 0.0))
        cube = np.asarray(observation[4:7], dtype=float)
        tcp = np.asarray(observation[:3], dtype=float)
        target = base._stage1_approach_target(cube, goal)
        tcp_cube_distances.append(float(np.linalg.norm(tcp - cube)))
        tcp_target_distances.append(float(np.linalg.norm(tcp - target)))
        goal_distances.append(float(np.linalg.norm(cube - goal)))
        apertures.append(float(base.gripper_aperture))
        max_displacement = max(max_displacement, float(np.linalg.norm(cube - cube0)))
        max_tilt = max(max_tilt, cube_tilt(base))
        max_penetration = max(max_penetration, max_robot_cube_penetration(base))
        table_collision_any |= gripper_table_contact(base)
        max_cube_angular_speed = max(
            max_cube_angular_speed,
            float(np.linalg.norm(base.data.joint("objjoint").qvel[3:])),
        )
        max_tcp_speed = max(max_tcp_speed, float(np.linalg.norm(tcp - previous_tcp)) / dt)
        previous_tcp = tcp.copy()
        physical, named_pad, _ = base.robot_object_contact_diagnostics(base._get_id_main_object())
        valid = bool(base._rear_contact_diagnostics()[0])
        # Handoff checks are meaningful only after valid rear contact and are
        # comparatively expensive; skip them for the overwhelmingly common
        # no-contact steps without changing the classification.
        ready = False
        if valid:
            handoff = extract_handoff_state(base, cube0)
            ready = not handoff_violations(handoff, base.handoff_thresholds)
        physical_steps += int(physical); pad_steps += int(named_pad)
        valid_steps += int(valid); handoff_steps += int(ready)
        if physical and first_physical is None: first_physical = step
        if named_pad and first_pad is None: first_pad = step
        if valid and first_valid is None: first_valid = step
        if previous_valid and not valid: loss_events += 1
        if valid and not previous_valid and ever_valid: reacquisition_events += 1
        ever_valid |= valid; previous_valid = valid; was_valid = valid
        unit, lateral = goal_frame(cube, goal)
        rel = tcp - cube
        offset_samples.append([
            float(np.dot(rel[:2], unit)), float(np.dot(rel[:2], lateral)), float(rel[2])
        ])
        any_success |= bool(info.get("success", info.get("is_success", False)))
        if terminated or truncated:
            break
    action_array = np.asarray(actions)
    offsets = np.asarray(offset_samples)
    component_sum = sum(component_totals.values())
    final_goal_distance = goal_distances[-1]
    goal_displacement = initial_goal_distance - final_goal_distance
    if not physical_steps:
        failure = "never_physical_contact"
    elif not pad_steps:
        failure = "wrong_geometry_contact"
    elif not valid_steps:
        failure = "pad_contact_not_v6_valid"
    elif valid_steps <= 2:
        failure = "valid_contact_immediately_lost"
    elif goal_displacement <= -1e-4:
        failure = "push_wrong_direction"
    elif goal_displacement > 1e-4 and not any_success:
        failure = "goal_directed_push_incomplete"
    elif not any_success:
        failure = "compatible_contact_without_push"
    else:
        failure = "success"
    return {
        "policy": label, "task_index": task_index, "reset_seed": TASK_SET_SEED + task_index,
        "steps": len(actions), "return": observed_return,
        **component_totals, "component_sum": component_sum,
        "reward_sum_error": observed_return - component_sum,
        "any_step_success": any_success,
        "final_step_success": bool(goal_distances[-1] <= base.TARGET_RADIUS),
        "initial_goal_distance_m": initial_goal_distance,
        "minimum_goal_distance_m": min(goal_distances),
        "final_goal_distance_m": final_goal_distance,
        "goal_directed_displacement_m": goal_displacement,
        "any_physical_contact": bool(physical_steps), "any_pad_contact": bool(pad_steps),
        "any_valid_rear_contact": bool(valid_steps), "any_handoff_ready": bool(handoff_steps),
        "first_physical_contact_step": first_physical,
        "first_pad_contact_step": first_pad, "first_valid_contact_step": first_valid,
        "physical_contact_steps": physical_steps, "pad_contact_steps": pad_steps,
        "valid_contact_steps": valid_steps, "handoff_ready_steps": handoff_steps,
        "contact_loss_events": loss_events, "contact_reacquisition_events": reacquisition_events,
        "initial_tcp_cube_distance_m": initial_tcp_cube,
        "minimum_tcp_cube_distance_m": min(tcp_cube_distances),
        "final_tcp_cube_distance_m": tcp_cube_distances[-1],
        "initial_tcp_target_distance_m": initial_tcp_target,
        "minimum_tcp_target_distance_m": min(tcp_target_distances),
        "final_tcp_target_distance_m": tcp_target_distances[-1],
        "mean_forward_offset_m": float(np.mean(offsets[:, 0])),
        "mean_lateral_offset_m": float(np.mean(offsets[:, 1])),
        "mean_vertical_offset_m": float(np.mean(offsets[:, 2])),
        "maximum_tcp_speed_m_s": max_tcp_speed,
        "initial_aperture_m": apertures[0], "mean_aperture_m": float(np.mean(apertures[1:])),
        "final_aperture_m": apertures[-1],
        "aperture_in_handoff_range_fraction": float(np.mean((np.asarray(apertures[1:]) >= .0835) & (np.asarray(apertures[1:]) <= .0845))),
        "mean_gripper_action": float(np.mean(action_array[:, 3])),
        "opening_action_fraction": float(np.mean(action_array[:, 3] < 0)),
        "closing_action_fraction": float(np.mean(action_array[:, 3] > 0)),
        "maximum_cube_displacement_m": max_displacement,
        "maximum_cube_tilt_rad": max_tilt,
        "maximum_cube_angular_speed_rad_s": max_cube_angular_speed,
        "maximum_penetration_m": max_penetration,
        "gripper_table_collision": table_collision_any,
        "clipped_action_steps": clipped_action_steps,
        "x_clip_events": int(clips[0]), "y_clip_events": int(clips[1]), "z_clip_events": int(clips[2]),
        "failure_category": failure,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))
    return {
        "episodes": len(rows),
        "any_physical_contact_rate": mean("any_physical_contact"),
        "pad_contact_rate": mean("any_pad_contact"),
        "valid_rear_contact_rate": mean("any_valid_rear_contact"),
        "first_contact_bonus_rate": float(np.mean([r["reward_contact"] > 0 for r in rows])),
        "handoff_ready_rate": mean("any_handoff_ready"),
        "any_step_goal_success_rate": mean("any_step_success"),
        "final_step_goal_success_rate": mean("final_step_success"),
        "mean_return": mean("return"), "return_std": float(np.std([r["return"] for r in rows])),
        "mean_minimum_goal_distance_m": mean("minimum_goal_distance_m"),
        "mean_final_goal_distance_m": mean("final_goal_distance_m"),
        "mean_minimum_tcp_target_distance_m": mean("minimum_tcp_target_distance_m"),
        "mean_final_tcp_target_distance_m": mean("final_tcp_target_distance_m"),
        "mean_minimum_tcp_cube_distance_m": mean("minimum_tcp_cube_distance_m"),
        "mean_final_tcp_cube_distance_m": mean("final_tcp_cube_distance_m"),
        "mean_aperture_m": mean("mean_aperture_m"),
        "aperture_in_handoff_range_fraction": mean("aperture_in_handoff_range_fraction"),
        "mean_goal_directed_displacement_m": mean("goal_directed_displacement_m"),
        "maximum_cube_tilt_rad": max(r["maximum_cube_tilt_rad"] for r in rows),
        "workspace_clip_episode_rate": float(np.mean([(r["x_clip_events"] + r["y_clip_events"] + r["z_clip_events"]) > 0 for r in rows])),
        "reward_contact": sum(r["reward_contact"] for r in rows),
        "reward_push": sum(r["reward_push"] for r in rows),
        "reward_progress": sum(r["reward_progress"] for r in rows),
        "reward_success": sum(r["reward_first_success"] for r in rows),
        "maximum_reward_sum_error": max(abs(r["reward_sum_error"]) for r in rows),
        "failure_categories": {name: sum(r["failure_category"] == name for r in rows)
                               for name in sorted({r["failure_category"] for r in rows})},
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--only-random", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run.resolve()
    checkpoints = sorted((run / "checkpoints").glob("sac_*_steps.zip"))
    tasks = kuka_v3_tasks("kuka-push-cube-v3", TASK_SET_SEED, 2)
    all_rows = []
    checkpoint_metadata = []
    for checkpoint in ([] if args.only_random else checkpoints):
        model = SAC.load(checkpoint, device="cpu")
        step = int(model.num_timesteps)
        if step not in (20_000, 40_000, 60_000):
            continue
        checkpoint_metadata.append({
            "path": str(checkpoint), "sha256": sha256(checkpoint),
            "embedded_num_timesteps": step,
            "observation_shape": list(model.observation_space.shape),
            "action_shape": list(model.action_space.shape),
            "policy_kwargs": model.policy_kwargs,
        })
        env = make_env()
        try:
            for index, task in enumerate(tasks[:TASK_COUNT]):
                row = episode(env, task, index,
                              lambda obs, m=model: m.predict(obs, deterministic=True)[0],
                              f"sac_{step}")
                all_rows.append(row)
        finally:
            env.close()
    if args.random_policy or args.only_random:
        rng = np.random.default_rng(0)
        env = make_env()
        try:
            for index, task in enumerate(tasks[:TASK_COUNT]):
                all_rows.append(episode(
                    env, task, index,
                    lambda obs, random=rng: random.uniform(-1, 1, 4).astype(np.float32),
                    "random_seed_0",
                ))
        finally:
            env.close()
    labels = sorted({row["policy"] for row in all_rows})
    summaries = {label: summarize([row for row in all_rows if row["policy"] == label])
                 for label in labels}
    args.output.mkdir(parents=True)
    (args.output / "episode_metrics.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (args.output / "checkpoint_comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output / "checkpoint_comparison.csv").open("w", newline="") as stream:
        flat = ["policy"] + [key for key, value in next(iter(summaries.values())).items()
                             if not isinstance(value, dict)]
        writer = csv.DictWriter(stream, fieldnames=flat); writer.writeheader()
        for label in labels:
            writer.writerow({"policy": label, **{key: summaries[label][key] for key in flat[1:]}})
    failures = {label: {category: [r["task_index"] for r in all_rows
                                   if r["policy"] == label and r["failure_category"] == category]
                        for category in summaries[label]["failure_categories"]}
                for label in labels}
    (args.output / "failure_cases.json").write_text(json.dumps(failures, indent=2) + "\n")
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "run": str(run), "checkpoints": checkpoint_metadata,
        "task_set_seed": TASK_SET_SEED, "task_indices": [0, 49],
        "reset_seeds": [10_000, 10_049], "deterministic_sac": True,
        "horizon": HORIZON, "environment": {
            "name": "kuka-push-cube-v3", "reward": "v6_rear_contact_push_progress",
            "curriculum_level": 2, "initialization": "near_contact",
            "cube_y_min": 0.502, "near_contact_gap_m": 0.005,
        },
        "continual_learning_commit": git_head(root),
        "metaworld_kuka_commit": git_head(root.parent / "metaworld-kuka-v3"),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
