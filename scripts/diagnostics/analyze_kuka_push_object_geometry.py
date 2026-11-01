#!/usr/bin/env python3
"""Isolated KUKA Cube Push geometry/contact diagnostic (never trains)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kuka_push_scripted_controller import CalibratedPushController
from src.env_adapters.kuka_v3 import kuka_v3_tasks
from src.envs import make_env
from src.support.reproducibility import file_hash

OUTPUT_ROOT = ROOT / "experiments/diagnostics/kuka_push_cube_geometry"
ENV_REPO = ROOT.parent / "metaworld-kuka-v3"
XML = ENV_REPO / "metaworld_kuka/assets/kuka_xyz/kuka_push_cube_v3.xml"
SUCCESS_RADIUS = 0.05
HORIZON = 200
VARIANTS = {
    "cube_40": (0.040, 0.040, 0.040),
    "cube_60": (0.060, 0.060, 0.060),
    "cube_80": (0.080, 0.080, 0.080),
    "box_60x60x30": (0.060, 0.060, 0.030),
}


def box_inertia(mass: float, dimensions: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = dimensions
    return (mass * (y*y + z*z) / 12, mass * (x*x + z*z) / 12, mass * (x*x + y*y) / 12)


def variant_mass(name: str, strategy: str) -> float:
    dims = VARIANTS[name]
    return 0.75 if strategy == "constant_mass" else 0.75 * math.prod(dims) / 0.040**3


def controller_parameters(name: str, protocol: str) -> tuple[float, float]:
    if protocol == "fixed":
        return 0.060, 0.042
    x, y, z = VARIANTS[name]
    return 0.040 + max(x, y) / 2, z / 2 + 0.022


class GeometryAwareController(CalibratedPushController):
    def __init__(self, rho: float, tcp_z: float) -> None:
        super().__init__()
        self.rho = rho
        self.tcp_z = tcp_z


@dataclass(frozen=True)
class Configuration:
    key: str
    geometry: str
    dimensions_m: tuple[float, float, float]
    mass_strategy: str
    mass_kg: float
    inertia_kg_m2: tuple[float, float, float]
    controller_protocol: str
    controller_rho_m: float
    controller_tcp_z_m: float
    center_height_adjustment_m: float


def configurations() -> list[Configuration]:
    rows: list[Configuration] = []
    for geometry, dims in VARIANTS.items():
        for mass_strategy in ("constant_mass", "constant_density"):
            for protocol in ("fixed", "geometry_aware"):
                # Baseline physics and formulas are identical, so run it once.
                if geometry == "cube_40" and (mass_strategy != "constant_mass" or protocol != "fixed"):
                    continue
                mass = variant_mass(geometry, mass_strategy)
                rho, tcp_z = controller_parameters(geometry, protocol)
                rows.append(Configuration(
                    key=f"{geometry}__{mass_strategy}__{protocol}", geometry=geometry,
                    dimensions_m=dims, mass_strategy=mass_strategy, mass_kg=mass,
                    inertia_kg_m2=box_inertia(mass, dims), controller_protocol=protocol,
                    controller_rho_m=rho, controller_tcp_z_m=tcp_z,
                    center_height_adjustment_m=dims[2] / 2 - 0.020,
                ))
    return rows


def find_wrapper(env: Any) -> Any:
    node = env
    while node is not None:
        if hasattr(node, "tasks") and hasattr(node, "toggle_sample_tasks_on_reset"):
            return node
        node = getattr(node, "env", None)
    raise RuntimeError("fixed-task wrapper not found")


def apply_geometry(base: Any, cfg: Configuration) -> np.ndarray:
    """Modify only this compiled model/data instance and return refreshed obs."""
    geom_id = base.model.geom("objGeom").id
    body_id = base.model.body("obj").id
    base.model.geom_size[geom_id, :3] = np.asarray(cfg.dimensions_m) / 2
    base.model.body_mass[body_id] = cfg.mass_kg
    base.model.body_inertia[body_id, :3] = cfg.inertia_kg_m2
    z = cfg.dimensions_m[2] / 2
    base.data.joint("objjoint").qpos[2] = z
    base.data.joint("objjoint").qvel[:] = 0
    base.obj_init_pos = np.r_[base.obj_init_pos[:2], z]
    base._target_pos = np.r_[base._target_pos[:2], z]
    base.model.site("goal").pos[2] = z
    mujoco.mj_forward(base.model, base.data)
    return np.asarray(base._get_obs(), dtype=np.float32)


def workspace_would_clip(base: Any, action: np.ndarray) -> bool:
    proposed = np.asarray(base.data.mocap_pos[0]) + np.clip(action[:3], -1, 1) * base.action_scale
    return bool(np.any(proposed < base.mocap_low) or np.any(proposed > base.mocap_high))


def contact_sample(base: Any) -> dict[str, Any]:
    obj = base.model.geom("objGeom").id
    pad_ids = {
        "left_pad": {base.model.geom(n).id for n in ("left_pad1", "left_pad2")},
        "right_pad": {base.model.geom(n).id for n in ("right_pad1", "right_pad2")},
    }
    followers = {side: base.model.body(f"{side}_follower").id for side in ("left", "right")}
    physical, named_pad, follower = base.robot_object_contact_diagnostics(obj)
    out: dict[str, Any] = {"physical": physical, "named_pad": named_pad, "follower": follower,
                           "left_pad": False, "right_pad": False, "other_gripper": False,
                           "point_z": None, "relative_z": None}
    points = []
    gripper_root = base.model.body("base").id
    for c in base.data.contact:
        if obj not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == obj else c.geom1
        out["left_pad"] |= other in pad_ids["left_pad"]
        out["right_pad"] |= other in pad_ids["right_pad"]
        body = int(base.model.geom_bodyid[other])
        ancestry = body
        is_gripper = body == gripper_root
        while ancestry > 0 and not is_gripper:
            ancestry = int(base.model.body_parentid[ancestry])
            is_gripper = ancestry == gripper_root
        if is_gripper:
            known = out["left_pad"] or out["right_pad"] or body in followers.values()
            out["other_gripper"] |= not known
            points.append(np.asarray(c.pos).copy())
    if points:
        out["point_z"] = float(np.mean([p[2] for p in points]))
        out["relative_z"] = out["point_z"] - float(base.data.body("obj").xpos[2])
    return out


def tilt_deg(base: Any) -> float:
    rotation = np.asarray(base.data.geom("objGeom").xmat).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(np.max(np.abs(rotation[2, :])), 0, 1))))


def run_episode(env: Any, task: Any, task_index: int, cfg: Configuration, telemetry: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = env.unwrapped
    base.set_task(task)
    obs, _ = env.reset(seed=10_000 + task_index)
    obs = apply_geometry(base, cfg)
    obj0 = obs[4:7].copy(); goal = obs[36:39].copy()
    direction = goal[:2] - obj0[:2]; direction /= np.linalg.norm(direction)
    initial_distance = float(np.linalg.norm(obj0 - goal))
    controller = GeometryAwareController(cfg.controller_rho_m, cfg.controller_tcp_z_m)
    controller.reset(obj0, goal)
    first_contact = first_success = None
    contact_steps = contact_segments = losses = reacquisitions = 0
    was_contact = False; ever_contact = False; ever_success = False
    left = right = follower = other = clip = success_steps = exits = 0
    prior_success = False; effective_steps = 0
    contact_displacement = 0.0; prior_obj = obj0.copy()
    contact_z: list[float] = []; relative_z: list[float] = []
    max_tilt = max_ang = max_speed = 0.0; speed_at_success = None
    forward_progress = 0.0; lateral = 0.0
    traces: list[dict[str, Any]] = []
    for step in range(1, HORIZON + 1):
        action, _ = controller.predict(obs, base)
        clipped = workspace_would_clip(base, action); clip += clipped
        obs, reward, terminated, truncated, info = env.step(action)
        obj = np.asarray(obs[4:7]); vel = np.asarray(base.data.joint("objjoint").qvel)
        contact = contact_sample(base); touching = bool(contact["physical"])
        displacement = float(np.linalg.norm(obj - prior_obj))
        if touching:
            contact_steps += 1; contact_displacement += displacement
            effective_steps += int(np.dot(obj[:2] - prior_obj[:2], direction) > 1e-5)
            if first_contact is None: first_contact = step
            if contact["point_z"] is not None: contact_z.append(contact["point_z"]); relative_z.append(contact["relative_z"])
        if touching and not was_contact:
            contact_segments += 1
            if ever_contact: reacquisitions += 1
        if was_contact and not touching: losses += 1
        ever_contact |= touching; was_contact = touching
        left += contact["left_pad"]; right += contact["right_pad"]
        follower += contact["follower"]; other += contact["other_gripper"]
        distance = float(np.linalg.norm(obj - goal)); success = distance <= SUCCESS_RADIUS
        if success and first_success is None:
            first_success = step; speed_at_success = float(np.linalg.norm(vel[:3]))
        if prior_success and not success: exits += 1
        prior_success = success; ever_success |= success; success_steps += success
        max_tilt = max(max_tilt, tilt_deg(base)); max_ang = max(max_ang, float(np.linalg.norm(vel[3:])))
        max_speed = max(max_speed, float(np.linalg.norm(vel[:3])))
        delta = obj[:2] - obj0[:2]
        forward_progress = float(np.dot(delta, direction))
        lateral = float(abs(direction[0] * delta[1] - direction[1] * delta[0]))
        if telemetry:
            traces.append({"configuration": cfg.key, "task_index": task_index, "step": step,
                "object_x": float(obj[0]), "object_y": float(obj[1]), "object_z": float(obj[2]),
                "goal_distance": distance, "linear_speed": float(np.linalg.norm(vel[:3])),
                "angular_speed": float(np.linalg.norm(vel[3:])), "tilt_deg": tilt_deg(base),
                "physical_contact": touching, "left_pad": contact["left_pad"],
                "right_pad": contact["right_pad"], "follower": contact["follower"],
                "workspace_clipping": clipped, "success": success, "reward": float(reward)})
        prior_obj = obj.copy()
        if terminated or truncated: break
    final_distance = float(np.linalg.norm(prior_obj - goal))
    row = {"configuration": cfg.key, "geometry": cfg.geometry, "mass_strategy": cfg.mass_strategy,
        "controller_protocol": cfg.controller_protocol, "task_index": task_index, "reset_seed": 10_000 + task_index,
        "initial_distance": initial_distance, "first_contact_step": first_contact, "contact_steps": contact_steps,
        "contact_segments": contact_segments, "contact_losses": losses, "contact_reacquisitions": reacquisitions,
        "physical_contact": ever_contact, "left_pad_steps": left, "right_pad_steps": right,
        "follower_steps": follower, "other_gripper_steps": other,
        "mean_contact_point_z": float(np.mean(contact_z)) if contact_z else None,
        "mean_contact_relative_z": float(np.mean(relative_z)) if relative_z else None,
        "object_displacement_during_contact": contact_displacement,
        "effective_push": effective_steps > 0, "effective_push_steps": effective_steps,
        "workspace_clipping_events": clip, "forward_progress": forward_progress,
        "lateral_displacement": lateral, "final_distance": final_distance,
        "first_success_step": first_success, "ever_success": ever_success,
        "final_success": final_distance <= SUCCESS_RADIUS, "goal_exits": exits,
        "goal_retention": bool(ever_success and final_distance <= SUCCESS_RADIUS),
        "speed_at_first_success": speed_at_success, "max_linear_speed": max_speed,
        "max_angular_speed": max_ang, "max_tilt_deg": max_tilt,
        "tipped": max_tilt >= 15.0, "rolled": max_tilt >= 45.0 and max_ang >= 1.0,
        "tipped_and_succeeded": max_tilt >= 15.0 and ever_success,
        "successful_steps": success_steps}
    return row, traces


def mean(rows: list[dict[str, Any]], field: str, predicate=lambda x: True) -> float | None:
    values = [float(r[field]) for r in rows if r[field] is not None and predicate(r)]
    return float(np.mean(values)) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {"episodes": n,
        "ever_success_rate": mean(rows, "ever_success"), "final_success_rate": mean(rows, "final_success"),
        "goal_retention_rate": mean(rows, "goal_retention", lambda r: r["ever_success"]),
        "physical_contact_rate": mean(rows, "physical_contact"), "effective_push_rate": mean(rows, "effective_push"),
        "mean_first_contact_step": mean(rows, "first_contact_step"), "mean_contact_duration_steps": mean(rows, "contact_steps"),
        "left_pad_contact_episode_rate": float(np.mean([r["left_pad_steps"] > 0 for r in rows])),
        "right_pad_contact_episode_rate": float(np.mean([r["right_pad_steps"] > 0 for r in rows])),
        "mean_left_pad_contact_steps": mean(rows, "left_pad_steps"),
        "mean_right_pad_contact_steps": mean(rows, "right_pad_steps"),
        "follower_contact_episode_rate": float(np.mean([r["follower_steps"] > 0 for r in rows])),
        "other_gripper_contact_episode_rate": float(np.mean([r["other_gripper_steps"] > 0 for r in rows])),
        "mean_contact_losses": mean(rows, "contact_losses"), "mean_contact_reacquisitions": mean(rows, "contact_reacquisitions"),
        "mean_contact_point_z": mean(rows, "mean_contact_point_z"), "mean_contact_relative_z": mean(rows, "mean_contact_relative_z"),
        "mean_contact_object_displacement": mean(rows, "object_displacement_during_contact"),
        "mean_forward_progress": mean(rows, "forward_progress"), "mean_lateral_displacement": mean(rows, "lateral_displacement"),
        "mean_final_goal_distance": mean(rows, "final_distance"),
        "mean_object_speed_at_first_success": mean(rows, "speed_at_first_success"),
        "mean_max_angular_velocity": mean(rows, "max_angular_speed"), "tipping_frequency": mean(rows, "tipped"),
        "rolling_frequency": mean(rows, "rolled"), "workspace_clipping_episode_rate": float(np.mean([r["workspace_clipping_events"] > 0 for r in rows])),
        "goal_exit_episode_rate": float(np.mean([r["goal_exits"] > 0 for r in rows])),
        "tipped_but_succeeded_count": sum(r["tipped_and_succeeded"] for r in rows)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=int, default=50, choices=range(1, 51))
    p.add_argument("--smoke", action="store_true", help="run tasks 0, 19, 49 only")
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return p.parse_args()


def main() -> None:
    args = parse_args(); before_hash = file_hash(XML)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = args.output_root / stamp; out.mkdir(parents=True, exist_ok=False)
    cfgs = configurations(); task_bank = kuka_v3_tasks("kuka-push-cube-v3", 10_000)
    task_indices = [0, 19, 49] if args.smoke else list(range(args.tasks))
    episode_rows: list[dict[str, Any]] = []; traces: list[dict[str, Any]] = []
    started = time.monotonic()
    for ci, cfg in enumerate(cfgs, 1):
        print(f"[{ci}/{len(cfgs)}] {cfg.key}", flush=True)
        env = make_env("kuka-v3", "kuka-push-cube-v3", 10_000, render_mode=None,
                       reward_function_version="sawyer_push_v2_adapted")
        try:
            for index in task_indices:
                row, trace = run_episode(env, task_bank[index], index, cfg, index in (0, 19, 49))
                episode_rows.append(row); traces.extend(trace)
        finally: env.close()
    grouped = {cfg.key: aggregate([r for r in episode_rows if r["configuration"] == cfg.key]) for cfg in cfgs}
    config_rows = []
    for cfg in cfgs:
        row = asdict(cfg); row.update(grouped[cfg.key]); config_rows.append(row)
    feasibility = {"raw_original_center_z_m": 0.02,
        "variants": {name: {"dimensions_m": dims, "raw_bottom_z_m": 0.02-dims[2]/2,
            "raw_placement_feasible": abs(0.02-dims[2]/2) < 1e-12,
            "evaluation_center_and_goal_z_m": dims[2]/2,
            "explicit_vertical_adjustment_m": dims[2]/2-0.02,
            "initial_success_precluded_by_sampling_minimum_m": 0.15,
            "tcp_workspace_m": {"low": [-0.25, 0.42, 0.04], "high": [0.25, 0.75, 0.50]}}
            for name, dims in VARIANTS.items()}}
    summary = {"schema_version": 1, "created": stamp, "training_performed": False,
        "reward_version": "sawyer_push_v2_adapted", "success_definition": "3D object-center distance <= 0.05 m",
        "task_set_seed": 10000, "task_indices": task_indices, "reset_seed_formula": "10000 + task_index",
        "horizon": HORIZON, "elapsed_seconds": time.monotonic()-started,
        "baseline_xml_sha256_before": before_hash, "baseline_xml_sha256_after": file_hash(XML),
        "geometry_mutation": "in-memory compiled MuJoCo model only; explicit center/goal z=half-height",
        "tipping_definition": "max nearest-face tilt >= 15 degrees",
        "rolling_definition": "max nearest-face tilt >= 45 degrees and max angular speed >= 1 rad/s",
        "configurations": {cfg.key: {**asdict(cfg), **grouped[cfg.key]} for cfg in cfgs}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "geometry_feasibility.json").write_text(json.dumps(feasibility, indent=2) + "\n")
    contact_keys = ("physical_contact_rate", "mean_first_contact_step", "mean_contact_duration_steps",
        "left_pad_contact_episode_rate", "right_pad_contact_episode_rate", "mean_left_pad_contact_steps",
        "mean_right_pad_contact_steps", "follower_contact_episode_rate", "other_gripper_contact_episode_rate",
        "mean_contact_losses", "mean_contact_reacquisitions", "mean_contact_point_z",
        "mean_contact_relative_z", "mean_contact_object_displacement", "effective_push_rate")
    motion_keys = ("ever_success_rate", "final_success_rate", "goal_retention_rate", "mean_forward_progress",
        "mean_lateral_displacement", "mean_final_goal_distance", "mean_object_speed_at_first_success",
        "mean_max_angular_velocity", "tipping_frequency", "rolling_frequency",
        "workspace_clipping_episode_rate", "goal_exit_episode_rate", "tipped_but_succeeded_count")
    (out / "contact_analysis.json").write_text(json.dumps({key: {k: grouped[key][k] for k in contact_keys} for key in grouped}, indent=2) + "\n")
    (out / "object_motion_analysis.json").write_text(json.dumps({key: {k: grouped[key][k] for k in motion_keys} for key in grouped}, indent=2) + "\n")
    write_csv(out / "configurations.csv", config_rows); write_csv(out / "episodes.csv", episode_rows)
    write_csv(out / "selected_step_telemetry.csv", traces)
    report = ["# KUKA Cube Push Object-Geometry Diagnostic", "", f"Generated: `{stamp}`", "",
        "No training or policy update occurred. All geometry changes existed only in isolated compiled MuJoCo model instances.", "",
        "## Protocol", "", "- Reward: `sawyer_push_v2_adapted`; official success unchanged (3D center distance <= 0.05 m).",
        "- Fixed controller: rho=0.060 m, TCP z=0.042 m.",
        "- Geometry-aware formulas: rho=0.040+max(width,depth)/2; TCP z=height/2+0.022 m.",
        "- Larger objects would penetrate the table under the raw z=0.020 m reset. Evaluation explicitly sets object and goal center z to half-height.",
        "- Tipping: >=15 deg. Rolling diagnostic: >=45 deg plus >=1 rad/s angular speed.", "", "## Results", "",
        "| Configuration | Ever | Final | Contact | Effective push | Tip | Roll | Final distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cfg in cfgs:
        a = grouped[cfg.key]
        report.append(f"| {cfg.key} | {a['ever_success_rate']:.1%} | {a['final_success_rate']:.1%} | {a['physical_contact_rate']:.1%} | {a['effective_push_rate']:.1%} | {a['tipping_frequency']:.1%} | {a['rolling_frequency']:.1%} | {a['mean_final_goal_distance']:.4f} |")
    report += ["", "## Interpretation boundary", "",
        "This report separates geometry, mass strategy, and controller protocol. Constant-density variants change both mass and inertia; geometry-aware runs also change controller targets. The larger-object runs include a documented vertical placement correction and therefore are not literal raw-reset comparisons.", "",
        "See `summary.json`, `configurations.csv`, `episodes.csv`, `selected_step_telemetry.csv`, `contact_analysis.json`, `object_motion_analysis.json`, and `geometry_feasibility.json` for measurements."]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"OUTPUT={out}", flush=True)


if __name__ == "__main__":
    main()
