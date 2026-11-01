"""Deterministically evaluate Cube Push on one fixed initialization distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continual.evaluation import disable_task_sampling
from src.env_adapters.kuka_v3 import kuka_v3_tasks
from src.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cube-push-initialization", choices=("default", "contact", "near_contact"), required=True)
    parser.add_argument("--cube-y-min", type=float, default=0.490)
    parser.add_argument("--near-contact-gap-m", type=float, default=0.005)
    parser.add_argument(
        "--reward-version",
        choices=("v5_contact_start_progress", "v6_rear_contact_push_progress"),
        default="v5_contact_start_progress",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task-set-seed", type=int, default=10_000)
    parser.add_argument("--reset-seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.episodes <= 50:
        raise ValueError("--episodes must be in [1, 50]")
    model = SAC.load(args.model.expanduser().resolve(), device=args.device)
    env = make_env(
        "kuka-v3", "kuka-push-cube-v3", args.task_set_seed,
        reward_function_version=args.reward_version,
        cube_push_curriculum_level=2,
        cube_push_initialization=args.cube_push_initialization,
        cube_y_min=args.cube_y_min,
        near_contact_gap_m=args.near_contact_gap_m,
    )
    tasks = kuka_v3_tasks("kuka-push-cube-v3", args.task_set_seed, 2)
    disable_task_sampling(env)
    base = env.unwrapped
    rows = []
    try:
        for index in range(args.episodes):
            base.set_task(tasks[index])
            observation, _ = env.reset(seed=args.reset_seed + index)
            initial = np.asarray(observation[4:7], dtype=float).copy()
            goal = np.asarray(observation[36:39], dtype=float).copy()
            initial_distance = float(np.linalg.norm(initial - goal))
            previous_distance = initial_distance
            any_success = False
            contact_steps = wrong_steps = 0
            max_displacement = max_tilt = 0.0
            clipped = False
            for _ in range(200):
                action, _ = model.predict(observation, deterministic=True)
                proposed = np.asarray(base.data.mocap_pos[0]) + np.clip(action[:3], -1, 1) * base.action_scale
                clipped |= bool(np.any(proposed < base.mocap_low) or np.any(proposed > base.mocap_high))
                observation, _, terminated, truncated, info = env.step(action)
                obj = np.asarray(observation[4:7], dtype=float)
                distance = float(np.linalg.norm(obj - goal))
                wrong_steps += int(distance > previous_distance + 1e-5)
                previous_distance = distance
                contact_steps += int(base.robot_object_contact_diagnostics(base._get_id_main_object())[0])
                max_displacement = max(max_displacement, float(np.linalg.norm(obj - initial)))
                matrix = base.data.geom("objGeom").xmat.reshape(3, 3)
                tilt = float(np.degrees(np.arccos(np.clip(np.max(np.abs(matrix[:, 2])), 0, 1))))
                max_tilt = max(max_tilt, tilt)
                any_success |= bool(info["success"])
                if terminated or truncated:
                    break
            rows.append({
                "task_index": index,
                "any_step_success": any_success,
                "final_step_success": bool(info["success"]),
                "contact_rate": contact_steps / 200,
                "signed_goal_progress": initial_distance - previous_distance,
                "maximum_cube_displacement": max_displacement,
                "wrong_direction_step_rate": wrong_steps / 200,
                "max_tilt_deg": max_tilt,
                "tipped": max_tilt >= 15.0,
                "workspace_clipped": clipped,
            })
    finally:
        env.close()
    keys = ("any_step_success", "final_step_success", "contact_rate",
            "signed_goal_progress", "maximum_cube_displacement",
            "wrong_direction_step_rate", "tipped", "workspace_clipped")
    report = {
        "model": str(args.model.expanduser().resolve()),
        "initialization": args.cube_push_initialization,
        "reward": args.reward_version,
        "task_set_seed": args.task_set_seed,
        "reset_seeds": [args.reset_seed, args.reset_seed + args.episodes - 1],
        "episodes": args.episodes,
        "deterministic": True,
        "summary": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "episodes_detail": rows,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.open("x", encoding="utf-8").write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
