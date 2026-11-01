#!/usr/bin/env python3
"""Evaluate KUKA Cube Push Stage 2 from controlled contact perturbations.

This is deliberately an external diagnostic: it does not alter environment,
reward, policy, or training code. Results are written one condition at a time
so an interrupted matrix can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from stable_baselines3 import SAC

from src.continual.evaluation import disable_task_sampling
from src.env_adapters.kuka_v3 import make_kuka_v3_env, kuka_v3_tasks
from src.support.io import atomic_replace_json
from src.support.reproducibility import file_hash


DEFAULT_MODEL = Path(
    "runs/exploratory/kuka-push-cube-v3/contact_start/"
    "reward_v6_rear_contact_push_progress/mass_100g/lr_1e4_resume_400k/"
    "seed_0_resume_200k_to_400k/best_success_model/best_success_model.zip"
)
EXPECTED_MODEL_SHA256 = (
    "5cdcbb5882dafed3c741b3d9d6d24648a46e27f3150c8d2bb75cbed2186f850a"
)
TASK_SET_SEED = 10_000
TASK_COUNT = 50
HORIZON = 200
MAX_SAFE_PENETRATION = 0.0005
# Passive Robotiq coupling joints exceed their nominal zero endpoint by up to
# 2.7 mrad in the canonical compliant Contact-Start reset itself.
JOINT_LIMIT_TOLERANCE = 0.005


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    kind: str
    value: float = 0.0
    direction: int = 0


def conditions() -> list[Condition]:
    result = [Condition("baseline", "baseline", "baseline")]
    for magnitude in (0.010, 0.020, 0.030, 0.050):
        for direction, label in ((1, "toward"), (-1, "away")):
            result.append(Condition(f"tcp_velocity_{label}_{magnitude*1000:.0f}mms", "tcp_velocity", "tcp_velocity", magnitude, direction))
    for magnitude in (0.005, 0.010, 0.020, 0.040):
        for direction, label in ((1, "toward"), (-1, "away")):
            result.append(Condition(f"cube_velocity_{label}_{magnitude*1000:.0f}mms", "cube_velocity", "cube_velocity", magnitude, direction))
    for magnitude in (0.1, 0.25, 0.5, 1.0):
        for direction, label in ((1, "positive"), (-1, "negative")):
            result.append(Condition(f"cube_angular_{label}_{magnitude:g}rads", "cube_angular", "cube_angular", magnitude, direction))
    for millimetres in (-5, -2, 2, 5):
        result.append(Condition(f"tcp_lateral_{millimetres:+d}mm", "tcp_lateral", "tcp_lateral", millimetres / 1000.0))
    for millimetres in (-5, -2, 2, 5):
        result.append(Condition(f"tcp_forward_{millimetres:+d}mm", "tcp_forward", "tcp_forward", millimetres / 1000.0))
    for millimetres in (85, 82, 80, 75):
        result.append(Condition(f"aperture_{millimetres}mm", "aperture", "aperture", millimetres / 1000.0))
    # Existing reset applies 0.75 mm after first valid contact. Values below are
    # desired changes relative to that established reference, not teleport depth.
    for delta_mm in (-0.75, -0.50, -0.25, 0.25):
        result.append(Condition(f"preload_delta_{delta_mm:+.2f}mm", "preload", "preload", delta_mm / 1000.0))
    result.extend(
        [
            Condition("history_short_approach", "history", "history_approach"),
            Condition("history_contact_settle", "history", "history_settle"),
        ]
    )
    return result


def atomic_json(path: Path, value: Any) -> None:
    atomic_replace_json(path, value, sort_keys=True)


def cube_tilt(quaternion_xyzw: np.ndarray) -> float:
    x, y, z, w = quaternion_xyzw
    local_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(local_z_world_z, -1.0, 1.0)))


def goal_frame(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cube = np.asarray(observation[4:7], dtype=float)
    goal = np.asarray(observation[36:39], dtype=float)
    unit = goal[:2] - cube[:2]
    unit /= np.linalg.norm(unit)
    return unit, np.array([unit[1], -unit[0]])


def move_mocap(base: Any, planar_delta: np.ndarray) -> None:
    distance = float(np.linalg.norm(planar_delta))
    steps = max(1, int(math.ceil(distance / 0.00025)))
    increment = np.asarray(planar_delta, dtype=float) / steps
    for _ in range(steps):
        base.data.mocap_pos[0, :2] += increment
        base.data.mocap_pos[0] = np.clip(base.data.mocap_pos[0], base.mocap_low, base.mocap_high)
        base.data.mocap_quat[0] = base.MOCAP_QUATERNION
        base.do_simulation([-1, 1], base.frame_skip)


def tcp_linear_jacobian(base: Any) -> np.ndarray:
    jacobian = np.zeros((3, base.model.nv))
    for site_name in ("rightEndEffector", "leftEndEffector"):
        jacp = np.zeros((3, base.model.nv))
        jacr = np.zeros((3, base.model.nv))
        mujoco.mj_jacSite(base.model, base.data, jacp, jacr, base.model.site(site_name).id)
        jacobian += 0.5 * jacp
    return jacobian


def robot_dofs(base: Any) -> list[int]:
    result = []
    for joint_id in range(base.model.njnt):
        name = base.model.joint(joint_id).name
        if name and name.startswith("kuka_joint_") and name.removeprefix("kuka_joint_").isdigit():
            result.append(int(base.model.jnt_dofadr[joint_id]))
    if len(result) != 7:
        raise RuntimeError(f"Expected seven KUKA arm joints, found {result}")
    return result


def achieve_aperture(base: Any, target: float) -> None:
    # The actuator command is position-like but the aperture/command mapping is
    # nonlinear under pad contact. Approach the target with a continuously
    # scaled command instead of a full close/open bang-bang step, which can
    # overshoot a requested aperture by more than a millimetre.
    for _ in range(500):
        error = target - float(base.gripper_aperture)
        if abs(error) <= 0.00025:
            break
        # -1 is fully open; increase effort gradually when aperture is too wide.
        effort = float(np.clip(-1.0 - 50.0 * error, -1.0, 1.0))
        base.do_simulation([effort, -effort], base.frame_skip)


def refresh_observation(base: Any) -> np.ndarray:
    return np.asarray(base._get_obs(), dtype=np.float32)


def apply_condition(base: Any, observation: np.ndarray, condition: Condition) -> np.ndarray:
    unit, lateral = goal_frame(observation)
    if condition.kind == "baseline":
        return observation
    if condition.kind in ("tcp_lateral", "tcp_forward", "preload"):
        direction = lateral if condition.kind == "tcp_lateral" else unit
        move_mocap(base, direction * condition.value)
        base.data.qvel[:] = 0.0
        mujoco.mj_forward(base.model, base.data)
        return refresh_observation(base)
    if condition.kind == "aperture":
        achieve_aperture(base, condition.value)
        base.data.qvel[:] = 0.0
        mujoco.mj_forward(base.model, base.data)
        return refresh_observation(base)
    if condition.kind == "cube_velocity":
        base.data.joint("objjoint").qvel[:] = 0.0
        base.data.joint("objjoint").qvel[:2] = condition.direction * condition.value * unit
        mujoco.mj_forward(base.model, base.data)
        return refresh_observation(base)
    if condition.kind == "cube_angular":
        base.data.joint("objjoint").qvel[:] = 0.0
        # Goal direction is +Y in Level 2; its lateral axis is the physically
        # relevant horizontal tipping axis.
        base.data.joint("objjoint").qvel[3:5] = condition.direction * condition.value * lateral
        mujoco.mj_forward(base.model, base.data)
        return refresh_observation(base)
    if condition.kind == "tcp_velocity":
        jacobian = tcp_linear_jacobian(base)
        dofs = robot_dofs(base)
        desired = np.r_[condition.direction * condition.value * unit, 0.0]
        base.data.qvel[:] = 0.0
        base.data.qvel[dofs] = np.linalg.pinv(jacobian[:, dofs], rcond=1e-6) @ desired
        mujoco.mj_forward(base.model, base.data)
        return refresh_observation(base)
    if condition.kind == "history_approach":
        move_mocap(base, -0.001 * unit)
        refresh_observation(base)
        move_mocap(base, 0.001 * unit)
        return refresh_observation(base)
    if condition.kind == "history_settle":
        refresh_observation(base)
        base.do_simulation([-1, 1], base.frame_skip)
        return refresh_observation(base)
    raise ValueError(condition)


def contact_snapshot(base: Any, observation: np.ndarray) -> dict[str, Any]:
    cube = np.asarray(observation[4:7], dtype=float)
    goal = np.asarray(observation[36:39], dtype=float)
    unit, lateral = goal_frame(observation)
    tcp = np.asarray(observation[:3], dtype=float)
    relative = tcp - cube
    valid, height, alignment = base._rear_contact_diagnostics(cube, goal)
    obj_id = base.model.geom("objGeom").id
    floor_id = base.model.geom("floor").id
    gripper_root = base.model.body("base").id
    gripper_bodies = {gripper_root}
    for body_id in range(base.model.nbody):
        parent = body_id
        while parent > 0:
            parent = int(base.model.body_parentid[parent])
            if parent == gripper_root:
                gripper_bodies.add(body_id)
                break
    gripper_geoms = {
        geom_id for geom_id in range(base.model.ngeom)
        if int(base.model.geom_bodyid[geom_id]) in gripper_bodies
        and int(base.model.geom_contype[geom_id]) != 0
    }
    max_penetration = 0.0
    cube_table = False
    gripper_table = False
    for contact in base.data.contact:
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        pair = {int(contact.geom1), int(contact.geom2)}
        cube_table |= pair == {obj_id, floor_id}
        gripper_table |= floor_id in pair and bool(pair & gripper_geoms)
    jacobian = tcp_linear_jacobian(base)
    tcp_velocity = jacobian @ np.asarray(base.data.qvel)
    cube_velocity = np.asarray(base.data.joint("objjoint").qvel, dtype=float).copy()
    joint_limits_valid = True
    joint_limit_violations = []
    for joint_id in range(base.model.njnt):
        if not bool(base.model.jnt_limited[joint_id]):
            continue
        address = int(base.model.jnt_qposadr[joint_id])
        low, high = base.model.jnt_range[joint_id]
        in_range = bool(
            low - JOINT_LIMIT_TOLERANCE
            <= base.data.qpos[address]
            <= high + JOINT_LIMIT_TOLERANCE
        )
        joint_limits_valid &= in_range
        if not in_range:
            joint_limit_violations.append(
                {"joint": base.model.joint(joint_id).name,
                 "position": float(base.data.qpos[address]),
                 "range": [float(low), float(high)]}
            )
    finite = bool(
        np.isfinite(observation).all()
        and np.isfinite(base.data.qpos).all()
        and np.isfinite(base.data.qvel).all()
    )
    return {
        "valid_rear_contact": bool(valid),
        "contact_height": float(height),
        "contact_alignment": float(alignment),
        "tcp_forward_offset": float(np.dot(relative[:2], unit)),
        "tcp_lateral_offset": float(np.dot(relative[:2], lateral)),
        "tcp_vertical_offset": float(relative[2]),
        "tcp_world_z": float(tcp[2]),
        "tcp_velocity": tcp_velocity.tolist(),
        "tcp_speed": float(np.linalg.norm(tcp_velocity)),
        "cube_linear_velocity": cube_velocity[:3].tolist(),
        "cube_planar_speed": float(np.linalg.norm(cube_velocity[:2])),
        "cube_angular_velocity": cube_velocity[3:].tolist(),
        "cube_angular_speed": float(np.linalg.norm(cube_velocity[3:])),
        "aperture": float(base.gripper_aperture),
        "cube_tilt": cube_tilt(np.asarray(observation[7:11], dtype=float)),
        "tcp_mocap_error": float(np.linalg.norm(tcp - base.data.mocap_pos[0])),
        "max_penetration": float(max_penetration),
        "cube_table_contact": bool(cube_table),
        "gripper_table_contact": bool(gripper_table),
        "joint_limits_valid": bool(joint_limits_valid),
        "joint_limit_violations": joint_limit_violations,
        "finite": finite,
        "history_max_abs_difference": float(np.max(np.abs(observation[:18] - observation[18:36]))),
    }


def physical_validity(snapshot: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if not snapshot["finite"]:
        reasons.append("nonfinite")
    if not snapshot["cube_table_contact"]:
        reasons.append("cube_not_on_table")
    if snapshot["gripper_table_contact"]:
        reasons.append("gripper_table_contact")
    if snapshot["max_penetration"] > MAX_SAFE_PENETRATION:
        reasons.append("penetration_above_0.5mm")
    if not snapshot["joint_limits_valid"]:
        reasons.append("joint_limit")
    return not reasons, reasons


def workspace_clipped(base: Any, action: np.ndarray) -> bool:
    position = np.asarray(base.data.mocap_pos[0])
    return bool(np.any((position <= base.mocap_low + 1e-7) & (action[:3] < 0)) or np.any((position >= base.mocap_high - 1e-7) & (action[:3] > 0)))


def evaluate_episode(env: Any, model: SAC, observation: np.ndarray, initial_cube: np.ndarray) -> dict[str, Any]:
    base = env.unwrapped
    initial_distance = float(np.linalg.norm(observation[4:7] - observation[36:39]))
    total_reward = 0.0
    any_success = False
    final_success = False
    valid_steps = 0
    first_loss = None
    reacquisitions = 0
    previous_valid = bool(base._rear_contact_diagnostics()[0])
    max_tilt = cube_tilt(np.asarray(observation[7:11], dtype=float))
    clipped_steps = 0
    apertures = []
    minimum_distance = initial_distance
    for step in range(HORIZON):
        action, _ = model.predict(observation, deterministic=True)
        clipped_steps += int(workspace_clipped(base, np.asarray(action)))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        current_valid = bool(base._rear_contact_diagnostics()[0])
        valid_steps += int(current_valid)
        if previous_valid and not current_valid and first_loss is None:
            first_loss = step + 1
        if not previous_valid and current_valid:
            reacquisitions += 1
        previous_valid = current_valid
        success = bool(info.get("success", False))
        any_success |= success
        final_success = success
        distance = float(np.linalg.norm(observation[4:7] - observation[36:39]))
        minimum_distance = min(minimum_distance, distance)
        max_tilt = max(max_tilt, cube_tilt(np.asarray(observation[7:11], dtype=float)))
        apertures.append(float(base.gripper_aperture))
        if terminated or truncated:
            break
    final_distance = float(np.linalg.norm(observation[4:7] - observation[36:39]))
    return {
        "any_step_success": any_success,
        "final_step_success": final_success,
        "return": total_reward,
        "initial_goal_distance": initial_distance,
        "minimum_goal_distance": minimum_distance,
        "final_goal_distance": final_distance,
        "final_goal_progress": initial_distance - final_distance,
        "maximum_goal_progress": initial_distance - minimum_distance,
        "first_contact_loss_step": first_loss,
        "rear_contact_fraction": valid_steps / HORIZON,
        "rear_contact_reacquisitions": reacquisitions,
        "maximum_cube_tilt": max_tilt,
        "tipped_over_5deg": max_tilt > math.radians(5.0),
        "workspace_clipped_steps": clipped_steps,
        "mean_aperture": float(np.mean(apertures)),
        "final_cube_displacement": float(np.linalg.norm(np.asarray(observation[4:7])[:2] - initial_cube[:2])),
    }


def summarize(condition: Condition, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [episode for episode in episodes if episode["physical_valid"]]
    rolled = [episode for episode in valid if "rollout" in episode]
    def mean(key: str) -> float | None:
        return None if not rolled else float(np.mean([episode["rollout"][key] for episode in rolled]))
    loss_steps = [episode["rollout"]["first_contact_loss_step"] for episode in rolled if episode["rollout"]["first_contact_loss_step"] is not None]
    return {
        "condition": asdict(condition),
        "tasks": len(episodes),
        "valid_initial_states": len(valid),
        "invalid_reasons": {
            reason: sum(reason in episode["invalid_reasons"] for episode in episodes)
            for reason in sorted({reason for episode in episodes for reason in episode["invalid_reasons"]})
        },
        "initial_valid_rear_contact_rate": None if not valid else float(np.mean([episode["initial"]["valid_rear_contact"] for episode in valid])),
        "any_step_success": mean("any_step_success"),
        "final_step_success": mean("final_step_success"),
        "mean_return": mean("return"),
        "mean_final_goal_progress": mean("final_goal_progress"),
        "mean_maximum_goal_progress": mean("maximum_goal_progress"),
        "mean_final_goal_distance": mean("final_goal_distance"),
        "mean_first_contact_loss_step": None if not loss_steps else float(np.mean(loss_steps)),
        "contact_loss_rate": None if not rolled else float(np.mean([episode["rollout"]["first_contact_loss_step"] is not None for episode in rolled])),
        "mean_rear_contact_fraction": mean("rear_contact_fraction"),
        "mean_reacquisitions": mean("rear_contact_reacquisitions"),
        "tipping_rate": mean("tipped_over_5deg"),
        "mean_workspace_clipped_steps": mean("workspace_clipped_steps"),
        "mean_initial_aperture": None if not valid else float(np.mean([episode["initial"]["aperture"] for episode in valid])),
        "mean_initial_tcp_speed": None if not valid else float(np.mean([episode["initial"]["tcp_speed"] for episode in valid])),
        "mean_initial_cube_planar_speed": None if not valid else float(np.mean([episode["initial"]["cube_planar_speed"] for episode in valid])),
        "mean_initial_cube_angular_speed": None if not valid else float(np.mean([episode["initial"]["cube_angular_speed"] for episode in valid])),
        "mean_initial_forward_offset": None if not valid else float(np.mean([episode["initial"]["tcp_forward_offset"] for episode in valid])),
        "mean_initial_lateral_offset": None if not valid else float(np.mean([episode["initial"]["tcp_lateral_offset"] for episode in valid])),
        "mean_initial_penetration": None if not valid else float(np.mean([episode["initial"]["max_penetration"] for episode in valid])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    checkpoint_hash = file_hash(model_path)
    if checkpoint_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(f"Unexpected Stage 2 checkpoint SHA-256: {checkpoint_hash}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SAC.load(model_path, device="cpu")
    env = make_kuka_v3_env(
        "kuka-push-cube-v3", 0,
        reward_function_version="v6_rear_contact_push_progress",
        cube_push_curriculum_level=2,
        cube_push_initialization="contact",
    )
    tasks = kuka_v3_tasks("kuka-push-cube-v3", TASK_SET_SEED, 2)
    disable_task_sampling(env)
    selected = conditions()
    if args.families:
        families = set(args.families)
        selected = [condition for condition in selected if condition.family in families or condition.name == "baseline"]
    provenance = {
        "checkpoint": str(model_path),
        "checkpoint_sha256": checkpoint_hash,
        "task_set_seed": TASK_SET_SEED,
        "task_count": TASK_COUNT,
        "reset_seeds": list(range(TASK_SET_SEED, TASK_SET_SEED + TASK_COUNT)),
        "horizon": HORIZON,
        "deterministic": True,
        "max_safe_penetration": MAX_SAFE_PENETRATION,
        "passive_joint_limit_tolerance": JOINT_LIMIT_TOLERANCE,
        "conditions": [asdict(condition) for condition in selected],
    }
    atomic_json(output_dir / "protocol.json", provenance)
    summaries = []
    try:
        for condition in selected:
            result_path = output_dir / "conditions" / f"{condition.name}.json"
            if result_path.exists() and not args.overwrite:
                result = json.loads(result_path.read_text())
                summaries.append(result["summary"])
                print(f"SKIP {condition.name}: existing result", flush=True)
                continue
            episodes = []
            for task_index in range(TASK_COUNT):
                env.unwrapped.set_task(tasks[task_index])
                observation, _ = env.reset(seed=TASK_SET_SEED + task_index)
                initial_cube = np.asarray(observation[4:7], dtype=float).copy()
                observation = apply_condition(env.unwrapped, np.asarray(observation), condition)
                snapshot = contact_snapshot(env.unwrapped, observation)
                valid, reasons = physical_validity(snapshot)
                episode = {
                    "task_index": task_index,
                    "reset_seed": TASK_SET_SEED + task_index,
                    "physical_valid": valid,
                    "invalid_reasons": reasons,
                    "initial": snapshot,
                }
                if valid:
                    episode["rollout"] = evaluate_episode(env, model, observation, initial_cube)
                episodes.append(episode)
            summary = summarize(condition, episodes)
            result = {"summary": summary, "episodes": episodes}
            atomic_json(result_path, result)
            summaries.append(summary)
            print(
                f"DONE {condition.name}: valid={summary['valid_initial_states']}/50 "
                f"contact={summary['initial_valid_rear_contact_rate']} "
                f"any={summary['any_step_success']} final={summary['final_step_success']}",
                flush=True,
            )
            if condition.name == "baseline" and summary["any_step_success"] != 0.84:
                raise RuntimeError(
                    "Baseline did not reproduce 84% any-step success; stopping before perturbations"
                )
        baseline = next(summary for summary in summaries if summary["condition"]["name"] == "baseline")
        for summary in summaries:
            for metric in ("any_step_success", "final_step_success", "mean_return"):
                value, reference = summary[metric], baseline[metric]
                summary[f"delta_{metric}"] = None if value is None or reference is None else value - reference
        atomic_json(output_dir / "summary.json", {"protocol": provenance, "results": summaries})
    finally:
        env.close()


if __name__ == "__main__":
    main()
