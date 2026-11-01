#!/usr/bin/env python3
"""Measure feasible physical Near-Contact gaps without changing the environment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from src.continual.evaluation import disable_task_sampling
from src.env_adapters.kuka_v3 import make_kuka_v3_env, kuka_v3_tasks

TASK_SET_SEED = 10_000
TASK_COUNT = 50
CANDIDATE_GAPS_M = (0.001, 0.002, 0.003, 0.005, 0.007, 0.010)
GAP_TOLERANCE_M = 2.5e-5
SEARCH_ITERATIONS = 14
SETTLE_STEPS = 100
APPROACH_ACTION_Y = 0.025  # 0.25 mm commanded motion per policy step
APPROACH_HORIZON = 100
JOINT_LIMIT_TOLERANCE = 0.005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def gripper_geom_ids(base: Any) -> list[int]:
    root = base.model.body("base").id
    bodies = {root}
    for body_id in range(base.model.nbody):
        parent = body_id
        while parent > 0:
            parent = int(base.model.body_parentid[parent])
            if parent == root:
                bodies.add(body_id)
                break
    return [
        geom_id for geom_id in range(base.model.ngeom)
        if int(base.model.geom_bodyid[geom_id]) in bodies
        and int(base.model.geom_contype[geom_id]) != 0
    ]


def physical_gap(base: Any, gripper_geoms: list[int]) -> float:
    obj_id = base.model.geom("objGeom").id
    return min(
        float(mujoco.mj_geomDistance(
            base.model, base.data, geom_id, obj_id, 1.0, None
        ))
        for geom_id in gripper_geoms
    )


def cube_tilt(base: Any) -> float:
    matrix = np.asarray(base.data.geom("objGeom").xmat).reshape(3, 3)
    return math.acos(float(np.clip(matrix[2, 2], -1.0, 1.0)))


def contacts(base: Any, gripper_geoms: list[int]) -> tuple[bool, bool, bool]:
    obj_id = base.model.geom("objGeom").id
    table_id = base.model.geom("floor").id
    robot_cube = gripper_table = cube_table = False
    for contact in base.data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        robot_cube |= obj_id in pair and bool(pair & set(gripper_geoms))
        gripper_table |= table_id in pair and bool(pair & set(gripper_geoms))
        cube_table |= pair == {obj_id, table_id}
    return bool(robot_cube), bool(gripper_table), bool(cube_table)


def joint_status(base: Any) -> tuple[bool, float]:
    margin = math.inf
    valid = True
    for joint_id in range(base.model.njnt):
        name = base.model.joint(joint_id).name or ""
        if not name.startswith("kuka_joint_"):
            continue
        qpos = float(base.data.qpos[int(base.model.jnt_qposadr[joint_id])])
        low, high = map(float, base.model.jnt_range[joint_id])
        margin = min(margin, qpos - low, high - qpos)
        valid &= low - JOINT_LIMIT_TOLERANCE <= qpos <= high + JOINT_LIMIT_TOLERANCE
    return bool(valid), float(margin)


def restore_cube(base: Any, cube_qpos: np.ndarray) -> None:
    joint = base.data.joint("objjoint")
    joint.qpos[:] = cube_qpos
    joint.qvel[:] = 0.0
    mujoco.mj_forward(base.model, base.data)


def settle_at(base: Any, command: np.ndarray, cube_qpos: np.ndarray) -> None:
    base.hand_init_pos = command.copy()
    restore_cube(base, cube_qpos)
    base._reset_hand(SETTLE_STEPS)
    restore_cube(base, cube_qpos)
    base.data.qvel[:] = 0.0
    mujoco.mj_forward(base.model, base.data)


def search_gap(
    base: Any, gripper_geoms: list[int], contact_command: np.ndarray,
    cube_qpos: np.ndarray, target_gap: float, max_retreat: float,
) -> tuple[np.ndarray, float, float] | None:
    boundary = contact_command.copy()
    boundary[1] -= max_retreat
    settle_at(base, boundary, cube_qpos)
    if physical_gap(base, gripper_geoms) + GAP_TOLERANCE_M < target_gap:
        return None
    low, high = 0.0, max_retreat
    for _ in range(SEARCH_ITERATIONS):
        retreat = 0.5 * (low + high)
        command = contact_command.copy()
        command[1] -= retreat
        settle_at(base, command, cube_qpos)
        if physical_gap(base, gripper_geoms) < target_gap:
            low = retreat
        else:
            high = retreat
    command = contact_command.copy()
    command[1] -= high
    settle_at(base, command, cube_qpos)
    return command, high, physical_gap(base, gripper_geoms)


def approach(base: Any, initial_cube: np.ndarray) -> dict[str, Any]:
    first = None
    max_displacement = 0.0
    max_tilt = cube_tilt(base)
    clipped = False
    for step in range(1, APPROACH_HORIZON + 1):
        action = np.array([0.0, APPROACH_ACTION_Y, 0.0, -1.0], dtype=np.float32)
        proposed = base.data.mocap_pos[0] + action[:3] * base.action_scale
        clipped |= bool(np.any(proposed < base.mocap_low) or np.any(proposed > base.mocap_high))
        _, _, terminated, truncated, _ = base.step(action)
        cube = np.asarray(base.data.geom("objGeom").xpos, dtype=float)
        max_displacement = max(max_displacement, float(np.linalg.norm(cube - initial_cube)))
        max_tilt = max(max_tilt, cube_tilt(base))
        details = base._rear_contact_details()
        if details["valid"]:
            first = {
                "step": step,
                "alignment": float(details["alignment"]),
                "height_m": float(details["height"]),
                "penetration_m": float(details["penetration"]),
            }
            break
        if terminated or truncated:
            break
    return {
        "acquired": first is not None,
        "first_contact": first,
        "maximum_cube_displacement_m": max_displacement,
        "maximum_cube_tilt_rad": max_tilt,
        "workspace_clipped": clipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    env_repo = root.parent / "metaworld-kuka-v3"
    env = make_kuka_v3_env(
        "kuka-push-cube-v3", TASK_SET_SEED,
        reward_function_version="v6_rear_contact_push_progress",
        cube_push_curriculum_level=2, cube_push_initialization="contact",
    )
    disable_task_sampling(env)
    base = env.unwrapped
    tasks = kuka_v3_tasks("kuka-push-cube-v3", TASK_SET_SEED, 2)
    gripper_geoms = gripper_geom_ids(base)
    rows: list[dict[str, Any]] = []
    try:
        for task_index, task in enumerate(tasks[:TASK_COUNT]):
            base.set_task(task)
            observation, _ = env.reset(seed=TASK_SET_SEED + task_index)
            cube_qpos = np.asarray(base.data.joint("objjoint").qpos).copy()
            cube = np.asarray(base.data.geom("objGeom").xpos).copy()
            contact_command = np.asarray(base.hand_init_pos).copy()
            contact_reference = {
                "cube": cube.tolist(), "goal": np.asarray(base._target_pos).tolist(),
                "cube_quaternion_wxyz": cube_qpos[3:].tolist(),
                "tcp": np.asarray(base.tcp_center).tolist(),
                "mocap": np.asarray(base.data.mocap_pos[0]).tolist(),
                "command": contact_command.tolist(),
                "arm_qpos": [float(base.data.joint(f"kuka_joint_{i}").qpos[0]) for i in range(1, 8)],
                "aperture_m": float(base.gripper_aperture),
                "physical_gap_m": physical_gap(base, gripper_geoms),
            }
            max_retreat = max(0.0, contact_command[1] - float(base.mocap_low[1]))
            boundary = contact_command.copy(); boundary[1] = float(base.mocap_low[1])
            settle_at(base, boundary, cube_qpos)
            maximum_gap = physical_gap(base, gripper_geoms)
            maximum_joint_valid, maximum_joint_margin = joint_status(base)
            for requested_gap in CANDIDATE_GAPS_M:
                result = search_gap(
                    base, gripper_geoms, contact_command, cube_qpos,
                    requested_gap, max_retreat,
                )
                if result is None:
                    rows.append({
                        "task_index": task_index, "requested_gap_m": requested_gap,
                        "feasible": False, "failure_causes": ["y_workspace_violation"],
                        "maximum_gap_m": maximum_gap,
                    })
                    continue
                command, retreat, achieved_gap = result
                robot_cube, gripper_table, cube_table = contacts(base, gripper_geoms)
                joints_valid, joint_margin = joint_status(base)
                finite = bool(np.isfinite(base.data.qpos).all() and np.isfinite(base.data.qvel).all())
                causes = []
                if command[1] < base.mocap_low[1] - 1e-12: causes.append("y_workspace_violation")
                if command[2] < base.mocap_low[2] - 1e-12: causes.append("inherited_z_workspace_violation")
                if robot_cube: causes.append("robot_cube_contact")
                if gripper_table: causes.append("gripper_table_collision")
                if not cube_table: causes.append("cube_table_instability")
                if not joints_valid: causes.append("joint_limit_violation")
                if not finite: causes.append("invalid_physical_state")
                if abs(achieved_gap - requested_gap) > GAP_TOLERANCE_M: causes.append("gap_tolerance")
                scripted = approach(base, cube)
                rows.append({
                    "task_index": task_index, "requested_gap_m": requested_gap,
                    "feasible": not [c for c in causes if c != "inherited_z_workspace_violation"],
                    "failure_causes": causes, "achieved_gap_m": achieved_gap,
                    "gap_error_m": achieved_gap - requested_gap,
                    "commanded_retreat_m": retreat, "command_y_m": float(command[1]),
                    "y_workspace_margin_m": float(command[1] - base.mocap_low[1]),
                    "z_workspace_margin_m": float(command[2] - base.mocap_low[2]),
                    "tcp_mocap_tracking_error_m": float(np.linalg.norm(base.tcp_center - base.data.mocap_pos[0])),
                    "joint_limits_valid": joints_valid, "minimum_joint_margin_rad": joint_margin,
                    "gripper_table_collision": gripper_table, "cube_table_contact": cube_table,
                    "finite_state": finite, "maximum_gap_m": maximum_gap,
                    "scripted_approach": scripted, "contact_reference": contact_reference,
                })
            rows[-1]["maximum_boundary_joint_valid"] = maximum_joint_valid
            rows[-1]["maximum_boundary_joint_margin_rad"] = maximum_joint_margin
    finally:
        env.close()

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "task_set_seed": TASK_SET_SEED, "task_indices": [0, 49],
        "reset_seed_formula": "10000 + task_index", "candidate_gaps_m": CANDIDATE_GAPS_M,
        "gap_tolerance_m": GAP_TOLERANCE_M, "search_iterations": SEARCH_ITERATIONS,
        "settle_steps": SETTLE_STEPS, "approach_action": [0.0, APPROACH_ACTION_Y, 0.0, -1.0],
        "approach_horizon": APPROACH_HORIZON,
        "gap_definition": "minimum mj_geomDistance(objGeom, collision-enabled Robotiq descendant geom)",
        "training_commit": git_head(root), "environment_commit": git_head(env_repo),
        "environment_sha256": sha256(env_repo / "metaworld_kuka/envs/kuka_push_cube_v3.py"),
        "diagnostic_sha256": sha256(Path(__file__)),
        "note": "Historical Contact-Start command z=0.039 m is 1 mm below mocap_low z=0.040 m; recorded separately.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    flat_fields = ["task_index", "requested_gap_m", "feasible", "achieved_gap_m", "gap_error_m", "commanded_retreat_m", "command_y_m", "y_workspace_margin_m", "z_workspace_margin_m", "maximum_gap_m", "failure_causes"]
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key)) if key == "failure_causes" else row.get(key) for key in flat_fields})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
