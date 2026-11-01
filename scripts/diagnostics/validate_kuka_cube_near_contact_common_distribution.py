#!/usr/bin/env python3
"""Validate the fixed-50 common-distribution Contact/Near-Contact experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from src.continual.evaluation import disable_task_sampling
from src.env_adapters.kuka_v3 import make_kuka_v3_env, kuka_v3_tasks

TASK_SEED = 10_000
TASK_COUNT = 50
GAP_M = 0.005
GAP_TOLERANCE_M = 0.000025
CUBE_Y_MIN = 0.502
ACTION = np.array([0.0, 0.025, 0.0, -1.0], dtype=np.float32)


def joint_limits_valid(base) -> bool:
    for joint_id in range(base.model.njnt):
        name = base.model.joint(joint_id).name or ""
        if name.startswith("kuka_joint_"):
            q = float(base.data.qpos[int(base.model.jnt_qposadr[joint_id])])
            low, high = base.model.jnt_range[joint_id]
            if not float(low) - 0.005 <= q <= float(high) + 0.005:
                return False
    return True


def cube_tilt(base) -> float:
    matrix = np.asarray(base.data.geom("objGeom").xmat).reshape(3, 3)
    return math.acos(float(np.clip(matrix[2, 2], -1.0, 1.0)))


def gripper_table_collision(base) -> bool:
    gripper = set(base._gripper_collision_geom_ids())
    floor = base.model.geom("floor").id
    return any(floor in (c.geom1, c.geom2) and bool({c.geom1, c.geom2} & gripper)
               for c in base.data.contact)


def make(mode: str):
    env = make_kuka_v3_env(
        "kuka-push-cube-v3", TASK_SEED,
        reward_function_version="v6_rear_contact_push_progress",
        cube_push_curriculum_level=2,
        cube_push_initialization=mode,
        cube_y_min=CUBE_Y_MIN,
        near_contact_gap_m=GAP_M,
    )
    disable_task_sampling(env)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contact, near = make("contact"), make("near_contact")
    tasks = kuka_v3_tasks("kuka-push-cube-v3", TASK_SEED, 2)
    rows = []
    try:
        for index, task in enumerate(tasks[:TASK_COUNT]):
            print(f"validating task {index}/49", flush=True)
            seed = TASK_SEED + index
            contact.unwrapped.set_task(task)
            near.unwrapped.set_task(task)
            contact_obs, _ = contact.reset(seed=seed)
            near_obs, _ = near.reset(seed=seed)
            cb, nb = contact.unwrapped, near.unwrapped
            initial_cube = np.asarray(near_obs[4:7], dtype=float).copy()
            gap = nb.cube_gripper_physical_gap()
            initial_contact = nb.robot_object_contact_diagnostics(nb._get_id_main_object())[0]
            row = {
                "task_index": index, "reset_seed": seed,
                "cube": initial_cube.tolist(), "goal": near_obs[36:39].tolist(),
                "cube_y_m": float(initial_cube[1]), "physical_gap_m": gap,
                "gap_error_m": gap - GAP_M,
                "cube_position_difference_m": float(np.linalg.norm(contact_obs[4:7] - near_obs[4:7])),
                "cube_quaternion_difference": float(np.linalg.norm(contact_obs[7:11] - near_obs[7:11])),
                "goal_difference_m": float(np.linalg.norm(contact_obs[36:39] - near_obs[36:39])),
                "contact_start_valid_rear_contact": bool(cb._rear_contact_diagnostics()[0]),
                "contact_start_latch": bool(cb._reward_v6_contact_latched),
                "near_initial_robot_cube_contact": bool(initial_contact),
                "near_initial_latch": bool(nb._reward_v6_contact_latched),
                "near_workspace_y_margin_m": float(nb.hand_init_pos[1] - nb.mocap_low[1]),
                "near_workspace_z_margin_m": float(nb.hand_init_pos[2] - nb.mocap_low[2]),
                "near_aperture_m": float(nb.gripper_aperture),
                "contact_tcp": np.asarray(cb.tcp_center).tolist(),
                "near_tcp": np.asarray(nb.tcp_center).tolist(),
                "tcp_difference_m": float(np.linalg.norm(nb.tcp_center - cb.tcp_center)),
                "tcp_vertical_difference_m": float(nb.tcp_center[2] - cb.tcp_center[2]),
                "near_joint_limits_valid": joint_limits_valid(nb),
                "near_gripper_table_collision": gripper_table_collision(nb),
                "near_finite": bool(np.isfinite(near_obs).all() and np.isfinite(nb.data.qpos).all()),
                "near_initial_cube_speed": float(np.linalg.norm(nb.data.joint("objjoint").qvel)),
            }
            first = None
            max_displacement = 0.0
            max_tilt = cube_tilt(nb)
            clipped = False
            contact_bonus_count = 0
            for step in range(1, 101):
                proposed = nb.data.mocap_pos[0] + ACTION[:3] * nb.action_scale
                clipped |= bool(np.any(proposed < nb.mocap_low) or np.any(proposed > nb.mocap_high))
                observation, _, terminated, truncated, info = near.step(ACTION)
                contact_bonus_count += int(info.get("reward_contact", 0.0) > 0)
                cube = np.asarray(observation[4:7], dtype=float)
                max_displacement = max(max_displacement, float(np.linalg.norm(cube - initial_cube)))
                max_tilt = max(max_tilt, cube_tilt(nb))
                details = nb._rear_contact_details()
                if first is None and details["valid"]:
                    first = {
                        "step": step, "alignment": float(details["alignment"]),
                        "height_m": float(details["height"]),
                        "penetration_m": float(details["penetration"]),
                        "tcp_mocap_error_m": float(np.linalg.norm(nb.tcp_center - nb.data.mocap_pos[0])),
                    }
                    # One additional step proves the latch does not pay again.
                    _, _, _, _, next_info = near.step(np.array([0, 0, 0, -1], dtype=np.float32))
                    contact_bonus_count += int(next_info.get("reward_contact", 0.0) > 0)
                    break
                if terminated or truncated:
                    break
            row.update({
                "contact_acquired": first is not None,
                "first_contact": first,
                "contact_bonus_count": contact_bonus_count,
                "maximum_cube_displacement_m": max_displacement,
                "maximum_cube_tilt_rad": max_tilt,
                "workspace_clipped": clipped,
            })
            row["valid"] = bool(
                abs(gap - GAP_M) <= GAP_TOLERANCE_M and not initial_contact
                and row["cube_position_difference_m"] <= 1e-7
                and row["cube_quaternion_difference"] <= 1e-7
                and row["goal_difference_m"] <= 1e-7
                and row["contact_start_valid_rear_contact"]
                and row["contact_start_latch"] and not row["near_initial_latch"]
                and row["near_workspace_y_margin_m"] >= 0
                and row["near_workspace_z_margin_m"] >= 0
                and row["near_joint_limits_valid"]
                and not row["near_gripper_table_collision"]
                and row["near_finite"] and row["contact_acquired"]
                and row["contact_bonus_count"] == 1 and not clipped
            )
            rows.append(row)
    finally:
        contact.close(); near.close()
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = {
        "tasks": len(rows), "valid": sum(row["valid"] for row in rows),
        "gap_m": {"min": min(r["physical_gap_m"] for r in rows),
                  "mean": float(np.mean([r["physical_gap_m"] for r in rows])),
                  "max": max(r["physical_gap_m"] for r in rows)},
        "contact_acquisition": sum(r["contact_acquired"] for r in rows),
        "mean_steps_to_contact": float(np.mean([r["first_contact"]["step"] for r in rows if r["first_contact"]])),
        "maximum_cube_displacement_m": max(r["maximum_cube_displacement_m"] for r in rows),
        "maximum_cube_tilt_rad": max(r["maximum_cube_tilt_rad"] for r in rows),
        "workspace_clips": sum(r["workspace_clipped"] for r in rows),
        "minimum_y_margin_m": min(r["near_workspace_y_margin_m"] for r in rows),
        "minimum_z_margin_m": min(r["near_workspace_z_margin_m"] for r in rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
