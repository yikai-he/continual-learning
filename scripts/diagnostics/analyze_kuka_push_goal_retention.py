#!/usr/bin/env python3
"""Deterministic post-success counterfactuals for KUKA Cube Push.

This script never trains or mutates an environment directly. Interventions use
only the existing four-dimensional action interface after the first successful
environment step.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stable_baselines3 import SAC

from src.envs import make_env


DEFAULT_CHECKPOINT = REPOSITORY_ROOT / (
    "runs/reward_ablations/kuka-push-cube-v3/sawyer_push_v2_adapted/"
    "seed_0_200k/best_success_model/best_success_model.zip"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "experiments/diagnostics/kuka_push_cube_goal_retention"
)
MODES = ("original", "stop", "retract", "terminate")
TARGET_RADIUS = 0.05
RETRACT_XY_DISTANCE = 0.08
RETRACT_Z_CLEARANCE = 0.06


@dataclass
class EpisodeResult:
    mode: str
    task_index: int
    reset_seed: int
    episode_length: int
    initial_distance: float
    minimum_distance: float
    final_distance: float
    first_success_step: int | None
    last_success_step: int | None
    successful_steps: int
    ever_success: bool
    final_success: bool
    continuous_retention: bool
    entries: int
    exits: int
    exit_step: int | None
    post_success_displacement: float
    post_success_contact_steps: int
    first_success_cube_speed: float | None
    first_success_goal_velocity: float | None
    workspace_clipping_events: int
    contact_at_success: bool | None
    contact_released: bool
    contact_release_step: int | None
    retraction_completed: bool
    retraction_moved_cube: float
    total_reward: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-set-seed", type=int, default=10_000)
    parser.add_argument("--reset-seed-base", type=int, default=10_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def unique_output_directory(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.exists():
        return root
    return root.parent / f"{root.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def find_task_wrapper(env: Any) -> Any:
    current = env
    while current is not None:
        if hasattr(current, "tasks") and hasattr(current, "toggle_sample_tasks_on_reset"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("KUKA fixed-task wrapper not found")


def held_gripper_action(base: Any) -> float:
    """Invert the Robotiq policy-to-actuator mapping for the current command."""
    return float(np.clip(float(base.data.ctrl[0]) / 127.5 - 1.0, -1.0, 1.0))


def workspace_would_clip(base: Any, action: np.ndarray) -> bool:
    delta = np.clip(np.asarray(action[:3], dtype=float), -1.0, 1.0)
    requested = np.asarray(base.data.mocap_pos[0], dtype=float) + delta * base.action_scale
    return bool(np.any(requested < base.mocap_low) or np.any(requested > base.mocap_high))


def contact_flags(base: Any) -> dict[str, bool]:
    object_id = base._get_id_main_object()
    physical, named_pad, follower = base.robot_object_contact_diagnostics(object_id)
    pad_names = {
        "left_pad": ("left_pad1", "left_pad2"),
        "right_pad": ("right_pad1", "right_pad2"),
    }
    result = {
        "physical_contact": physical,
        "named_pad_contact": named_pad,
        "follower_contact": follower,
        "left_pad_contact": False,
        "right_pad_contact": False,
        "left_follower_contact": False,
        "right_follower_contact": False,
    }
    pad_ids = {
        side: {base.model.geom(name).id for name in names}
        for side, names in pad_names.items()
    }
    follower_ids = {
        side: base.model.body(f"{side}_follower").id for side in ("left", "right")
    }
    for contact in base.data.contact:
        if object_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == object_id else contact.geom1
        result["left_pad_contact"] |= other in pad_ids["left_pad"]
        result["right_pad_contact"] |= other in pad_ids["right_pad"]
        body_id = int(base.model.geom_bodyid[other])
        result["left_follower_contact"] |= body_id == follower_ids["left"]
        result["right_follower_contact"] |= body_id == follower_ids["right"]
    return {key: bool(value) for key, value in result.items()}


def make_retraction_target(base: Any, cube: np.ndarray) -> np.ndarray:
    tcp = np.asarray(base.tcp_center, dtype=float)
    away = tcp[:2] - np.asarray(cube[:2], dtype=float)
    norm = float(np.linalg.norm(away))
    if norm <= np.finfo(float).eps:
        raise RuntimeError("Cannot define retraction: TCP and cube XY positions coincide")
    away /= norm
    target = tcp.copy()
    target[:2] += RETRACT_XY_DISTANCE * away
    target[2] += RETRACT_Z_CLEARANCE
    return np.clip(target, base.mocap_low, base.mocap_high)


def retraction_action(base: Any, target: np.ndarray) -> tuple[np.ndarray, bool]:
    error = np.asarray(target, dtype=float) - np.asarray(base.tcp_center, dtype=float)
    completed = bool(np.linalg.norm(error) <= 0.015)
    xyz = np.zeros(3) if completed else np.clip(error / base.action_scale, -1.0, 1.0)
    return np.r_[xyz, -1.0].astype(np.float32), completed


def aggregate(results: list[EpisodeResult]) -> dict[str, Any]:
    successful = [row for row in results if row.ever_success]
    post_success = successful or []
    return {
        "episodes": len(results),
        "ever_success_count": sum(row.ever_success for row in results),
        "ever_success_rate": float(np.mean([row.ever_success for row in results])),
        "final_success_count": sum(row.final_success for row in results),
        "final_success_rate": float(np.mean([row.final_success for row in results])),
        "goal_retention_rate": (
            float(np.mean([row.final_success for row in successful])) if successful else None
        ),
        "entered_and_exited_count": sum(row.exits > 0 for row in results),
        "continuous_retention_count": sum(row.continuous_retention for row in results),
        "mean_final_distance": float(np.mean([row.final_distance for row in results])),
        "mean_post_success_displacement": (
            float(np.mean([row.post_success_displacement for row in post_success]))
            if post_success else None
        ),
        "mean_successful_steps": (
            float(np.mean([row.successful_steps for row in successful]))
            if successful else None
        ),
        "mean_post_success_contact_steps": (
            float(np.mean([row.post_success_contact_steps for row in post_success]))
            if post_success else None
        ),
        "mean_cube_speed_at_first_success": (
            float(np.mean([row.first_success_cube_speed for row in successful]))
            if successful else None
        ),
        "workspace_clipping_events": sum(row.workspace_clipping_events for row in results),
        "successful_contact_releases": sum(row.contact_released for row in results),
        "retraction_completions": sum(row.retraction_completed for row in results),
    }


def run_episode(
    *, mode: str, task_index: int, task: Any, seed: int, model: SAC, env: Any
) -> tuple[EpisodeResult, list[dict[str, Any]], list[list[float]]]:
    base = env.unwrapped
    base.set_task(task)
    observation, _ = env.reset(seed=seed)
    initial_cube = np.asarray(observation[4:7], dtype=float).copy()
    goal = np.asarray(observation[36:39], dtype=float).copy()
    goal_direction = goal - initial_cube
    goal_direction /= np.linalg.norm(goal_direction)
    initial_distance = float(np.linalg.norm(initial_cube - goal))
    minimum_distance = initial_distance
    previous_tcp = np.asarray(base.tcp_center, dtype=float).copy()
    dt = float(base.model.opt.timestep * base.frame_skip)
    first_success_step = last_success_step = exit_step = None
    first_success_cube = None
    first_success_cube_speed = first_success_goal_velocity = None
    contact_at_success = None
    held_gripper = None
    retract_target = None
    retract_start_cube = None
    retraction_phase_cube_displacement = 0.0
    retraction_completed = False
    contact_released = False
    contact_release_step = None
    entries = exits = successful_steps = clipping_events = 0
    post_success_contact_steps = 0
    total_reward = 0.0
    previous_success = initial_distance <= TARGET_RADIUS
    traces: list[dict[str, Any]] = []
    proposed_actions: list[list[float]] = []

    for step in range(1, 201):
        proposed, _ = model.predict(observation, deterministic=True)
        proposed = np.asarray(proposed, dtype=np.float32)
        proposed_actions.append(proposed.astype(float).tolist())
        intervention_active = first_success_step is not None
        if not intervention_active or mode == "original":
            executed = proposed.copy()
        elif mode == "stop":
            executed = np.array([0.0, 0.0, 0.0, held_gripper], dtype=np.float32)
        elif mode == "retract":
            executed, completed_now = retraction_action(base, retract_target)
            retraction_completed |= completed_now
        elif mode == "terminate":
            break
        else:
            raise ValueError(mode)
        clipped = workspace_would_clip(base, executed)
        clipping_events += int(clipped)
        old_tcp = np.asarray(base.tcp_center, dtype=float).copy()
        observation, reward, terminated, truncated, info = env.step(executed)
        cube = np.asarray(observation[4:7], dtype=float).copy()
        tcp = np.asarray(base.tcp_center, dtype=float).copy()
        cube_velocity = np.asarray(base.data.joint("objjoint").qvel[:3], dtype=float).copy()
        tcp_velocity = (tcp - old_tcp) / dt
        distance = float(np.linalg.norm(cube - goal))
        success = bool(info["success"])
        flags = contact_flags(base)
        if (
            mode == "retract"
            and retract_target is not None
            and not retraction_completed
            and np.linalg.norm(np.asarray(base.tcp_center) - retract_target) <= 0.015
        ):
            retraction_completed = True
            retraction_phase_cube_displacement = float(
                np.linalg.norm(cube - retract_start_cube)
            )
        if success and not previous_success:
            entries += 1
        if previous_success and not success:
            exits += 1
            if exit_step is None:
                exit_step = step
        previous_success = success
        successful_steps += int(success)
        total_reward += float(reward)
        minimum_distance = min(minimum_distance, distance)
        if success:
            last_success_step = step
        if success and first_success_step is None:
            first_success_step = step
            first_success_cube = cube.copy()
            first_success_cube_speed = float(np.linalg.norm(cube_velocity))
            first_success_goal_velocity = float(np.dot(cube_velocity, goal_direction))
            contact_at_success = flags["physical_contact"]
            held_gripper = held_gripper_action(base)
            retract_target = make_retraction_target(base, cube)
            retract_start_cube = cube.copy()
        elif first_success_step is not None:
            post_success_contact_steps += int(flags["physical_contact"])
            if contact_at_success and not flags["physical_contact"] and not contact_released:
                contact_released = True
                contact_release_step = step
        traces.append(
            {
                "mode": mode,
                "task_index": task_index,
                "reset_seed": seed,
                "step": step,
                "cube_x": float(cube[0]), "cube_y": float(cube[1]), "cube_z": float(cube[2]),
                "cube_vx": float(cube_velocity[0]), "cube_vy": float(cube_velocity[1]), "cube_vz": float(cube_velocity[2]),
                "tcp_x": float(tcp[0]), "tcp_y": float(tcp[1]), "tcp_z": float(tcp[2]),
                "tcp_vx": float(tcp_velocity[0]), "tcp_vy": float(tcp_velocity[1]), "tcp_vz": float(tcp_velocity[2]),
                "goal_x": float(goal[0]), "goal_y": float(goal[1]), "goal_z": float(goal[2]),
                "cube_goal_distance": distance,
                **{f"proposed_action_{i}": float(proposed[i]) for i in range(4)},
                **{f"executed_action_{i}": float(executed[i]) for i in range(4)},
                **flags,
                "workspace_clipping": clipped,
                "success": success,
                "reward": float(reward),
                "first_success_step": first_success_step,
                "goal_entry": bool(success and (len(traces) == 0 or not traces[-1]["success"])),
                "goal_exit": bool(not success and len(traces) > 0 and traces[-1]["success"]),
                "intervention_active": intervention_active,
                "retraction_completed": retraction_completed,
            }
        )
        previous_tcp = tcp
        if terminated or truncated or (mode == "terminate" and first_success_step is not None):
            break

    final_cube = np.asarray(observation[4:7], dtype=float)
    final_distance = float(np.linalg.norm(final_cube - goal))
    ever_success = first_success_step is not None
    final_success = bool(final_distance <= TARGET_RADIUS)
    continuous_retention = bool(ever_success and exits == 0 and final_success and len(traces) == 200)
    post_displacement = (
        float(np.linalg.norm(final_cube - first_success_cube)) if ever_success else 0.0
    )
    retract_moved_cube = retraction_phase_cube_displacement
    result = EpisodeResult(
        mode=mode, task_index=task_index, reset_seed=seed,
        episode_length=len(traces), initial_distance=initial_distance,
        minimum_distance=minimum_distance, final_distance=final_distance,
        first_success_step=first_success_step, last_success_step=last_success_step,
        successful_steps=successful_steps, ever_success=ever_success,
        final_success=final_success, continuous_retention=continuous_retention,
        entries=entries, exits=exits, exit_step=exit_step,
        post_success_displacement=post_displacement,
        post_success_contact_steps=post_success_contact_steps,
        first_success_cube_speed=first_success_cube_speed,
        first_success_goal_velocity=first_success_goal_velocity,
        workspace_clipping_events=clipping_events,
        contact_at_success=contact_at_success, contact_released=contact_released,
        contact_release_step=contact_release_step,
        retraction_completed=retraction_completed,
        retraction_moved_cube=retract_moved_cube, total_reward=total_reward,
    )
    return result, traces, proposed_actions


def validate_matched_actions(
    action_sets: dict[str, dict[int, list[list[float]]]],
    trace_sets: dict[str, dict[int, list[dict[str, Any]]]],
    results: dict[str, list[EpisodeResult]],
) -> dict[str, Any]:
    baseline = {row.task_index: row for row in results["original"]}
    checks = {
        "pre_success_actions_identical": True,
        "pre_success_states_identical": True,
        "never_success_trajectories_identical": True,
        "first_success_steps_identical": True,
        "stop_has_zero_translation": True,
    }
    for mode in MODES[1:]:
        rows = {row.task_index: row for row in results[mode]}
        for task, original in baseline.items():
            compare_steps = original.first_success_step or 200
            left = np.asarray(action_sets["original"][task][:compare_steps])
            right = np.asarray(action_sets[mode][task][:compare_steps])
            equal = left.shape == right.shape and np.array_equal(left, right)
            checks["pre_success_actions_identical"] &= bool(equal)
            state_columns = (
                "cube_x", "cube_y", "cube_z", "tcp_x", "tcp_y", "tcp_z",
                "cube_goal_distance",
            )
            baseline_states = np.asarray([
                [sample[name] for name in state_columns]
                for sample in trace_sets["original"][task][:compare_steps]
            ])
            mode_states = np.asarray([
                [sample[name] for name in state_columns]
                for sample in trace_sets[mode][task][:compare_steps]
            ])
            states_equal = (
                baseline_states.shape == mode_states.shape
                and np.array_equal(baseline_states, mode_states)
            )
            checks["pre_success_states_identical"] &= bool(states_equal)
            if original.first_success_step is None:
                checks["never_success_trajectories_identical"] &= bool(
                    equal and states_equal
                )
            checks["first_success_steps_identical"] &= rows[task].first_success_step == original.first_success_step
    for task, original in baseline.items():
        if original.first_success_step is None:
            continue
        for sample in trace_sets["stop"][task][original.first_success_step:]:
            translation = [sample[f"executed_action_{i}"] for i in range(3)]
            checks["stop_has_zero_translation"] &= bool(
                np.array_equal(translation, np.zeros(3))
            )
    checks["baseline_ever_success_18_percent"] = aggregate(results["original"])["ever_success_count"] == 9
    checks["baseline_final_success_4_percent"] = aggregate(results["original"])["final_success_count"] == 2
    checks["success_threshold_unchanged"] = TARGET_RADIUS == 0.05
    return checks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = unique_output_directory(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    model = SAC.load(checkpoint, device=args.device)
    env = make_env(
        "kuka-v3", "kuka-push-cube-v3", args.task_set_seed,
        reward_function_version="sawyer_push_v2_adapted",
    )
    wrapper = find_task_wrapper(env)
    tasks = list(wrapper.tasks)
    wrapper.toggle_sample_tasks_on_reset(False)
    results: dict[str, list[EpisodeResult]] = {mode: [] for mode in MODES}
    action_sets: dict[str, dict[int, list[list[float]]]] = {mode: {} for mode in MODES}
    trace_sets: dict[str, dict[int, list[dict[str, Any]]]] = {mode: {} for mode in MODES}
    telemetry: list[dict[str, Any]] = []
    try:
        for mode in MODES:
            for task_index, task in enumerate(tasks[:50]):
                result, trace, actions = run_episode(
                    mode=mode, task_index=task_index, task=task,
                    seed=args.reset_seed_base + task_index, model=model, env=env,
                )
                results[mode].append(result)
                action_sets[mode][task_index] = actions
                trace_sets[mode][task_index] = trace
                telemetry.extend(trace)
    finally:
        env.close()
    checks = validate_matched_actions(action_sets, trace_sets, results)
    if not all(checks.values()):
        raise RuntimeError(f"Counterfactual validation failed: {checks}")
    episode_rows = [asdict(row) for mode in MODES for row in results[mode]]
    summary = {
        "checkpoint": str(checkpoint),
        "task_set_seed": args.task_set_seed,
        "reset_seed_base": args.reset_seed_base,
        "target_radius": TARGET_RADIUS,
        "velocity_frame": "MuJoCo world frame; TCP velocity is finite-difference over one control step",
        "stop_gripper": "holds the actuator command present at first success",
        "retraction": {
            "xy_distance": RETRACT_XY_DISTANCE,
            "z_clearance": RETRACT_Z_CLEARANCE,
            "direction": "world-frame XY from cube toward TCP at first success",
            "gripper_action": -1.0,
        },
        "validation": checks,
        "modes": {mode: aggregate(results[mode]) for mode in MODES},
    }
    write_csv(output / "episodes.csv", episode_rows)
    write_csv(output / "telemetry.csv", telemetry)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# KUKA Cube Push Goal-Retention Counterfactual", "", f"Checkpoint: `{checkpoint}`", "", "```json", json.dumps(summary, indent=2), "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
