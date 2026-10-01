"""Read-only loading, diagnostics, and export for stored physical trajectories."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.continual.diffusion_data import (
    PHYSICAL_ACTION_SLICE,
    PHYSICAL_TRAJECTORY_FEATURES,
)
from src.envs import (
    ACTION_SHAPE,
    CURRENT_STATE_DIM,
    CURRENT_STATE_SLICE,
    GOAL_DIM,
    GOAL_SLICE,
    OBSERVATION_SHAPE,
    PREVIOUS_STATE_SLICE,
)

DEFAULT_TOLERANCE = 1e-6
PERCENTILES = (1, 5, 50, 90, 95, 99)
ACTION_THRESHOLDS = (0.90, 0.95, 0.99, 1.00)
CONSTANT_STATE_STD_TOL = 1e-6
CONSTANT_STATE_ABS_TOL = 1e-6


@dataclass(frozen=True)
class LoadedTrajectories:
    """Physical trajectories and optional metadata loaded without modification."""

    trajectories: tuple[np.ndarray, ...]
    metadata: tuple[dict[str, object], ...]
    source_format: str


def _physical(observations: np.ndarray, actions: np.ndarray, label: str) -> np.ndarray:
    if observations.ndim != 2 or observations.shape[1:] != OBSERVATION_SHAPE:
        raise ValueError(f"{label} observations must have shape (T, 39).")
    if actions.ndim != 2 or actions.shape != (len(observations), *ACTION_SHAPE):
        raise ValueError(f"{label} actions must have shape (T, 4).")
    if len(observations) < 1:
        raise ValueError(f"{label} must contain at least one timestep.")
    return np.concatenate((observations, actions), axis=-1)


def load_trajectories(path: str | Path) -> LoadedTrajectories:
    """Load canonical ``.npy`` groups or collected ``real_trajectories.npz``."""
    source = Path(path)
    if source.suffix == ".npy":
        values = np.load(source, allow_pickle=False)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3 or values.shape[-1] != PHYSICAL_TRAJECTORY_FEATURES:
            raise ValueError("Expected .npy data with shape (N,T,43) or (T,43).")
        if len(values) < 1 or values.shape[1] < 1:
            raise ValueError("Trajectory data must be non-empty.")
        return LoadedTrajectories(
            tuple(values[index] for index in range(len(values))),
            tuple({} for _ in range(len(values))),
            "packed_npy",
        )
    if source.suffix != ".npz":
        raise ValueError("Supported inputs are .npy and .npz trajectory files.")
    with np.load(source, allow_pickle=False) as archive:
        pattern = re.compile(r"episode_(\d+)_observations$")
        indices = sorted(
            int(match.group(1))
            for key in archive.files
            if (match := pattern.fullmatch(key))
        )
        if not indices:
            raise ValueError("No episode_<index>_observations entries found in .npz.")
        trajectories, metadata = [], []
        for index in indices:
            prefix = f"episode_{index}_"
            observation_key, action_key = prefix + "observations", prefix + "actions"
            if action_key not in archive:
                raise ValueError(f"Missing {action_key}.")
            trajectories.append(
                _physical(archive[observation_key], archive[action_key], prefix[:-1])
            )
            metadata.append(
                {
                    key[len(prefix) :]: archive[key].copy()
                    for key in archive.files
                    if key.startswith(prefix)
                    and key not in (observation_key, action_key)
                }
            )
    return LoadedTrajectories(tuple(trajectories), tuple(metadata), "real_npz")


def split_fields(trajectory: np.ndarray) -> dict[str, np.ndarray]:
    """Return views into one physical ``(T,43)`` trajectory."""
    values = np.asarray(trajectory)
    if values.ndim != 2 or values.shape[-1] != PHYSICAL_TRAJECTORY_FEATURES:
        raise ValueError("Expected one trajectory with shape (T,43).")
    if len(values) < 1:
        raise ValueError("Trajectory must be non-empty.")
    return {
        "current_state": values[:, CURRENT_STATE_SLICE],
        "previous_state": values[:, PREVIOUS_STATE_SLICE],
        "goal": values[:, GOAL_SLICE],
        "action": values[:, PHYSICAL_ACTION_SLICE],
    }


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {key: float("nan") for key in ("min", "max", "mean", "std")}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _norm_rows(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)



def distribution_statistics(values: np.ndarray) -> dict[str, float | int | None]:
    """Return JSON-safe statistics for finite scalar values."""
    finite = np.asarray(values)[np.isfinite(values)]
    keys = ("mean", "std", "median", "p90", "p95", "p99", "max")
    if not finite.size:
        return {"count": 0, **{key: None for key in keys}}
    return {
        "count": int(finite.size), "mean": float(np.mean(finite)),
        "std": float(np.std(finite)), "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)), "max": float(np.max(finite)),
    }


def _dimension_statistics(values: np.ndarray, prefix: str) -> dict[str, dict[str, object]]:
    result = {}
    for dimension in range(values.shape[1]):
        finite = values[:, dimension][np.isfinite(values[:, dimension])]
        name = f"{prefix}_{dimension:02d}"
        keys = ("mean", "std", "min", "max", "p01", "p05", "p50", "p95", "p99")
        if not finite.size:
            result[name] = {"count": 0, **{key: None for key in keys}}
            continue
        percentiles = np.percentile(finite, PERCENTILES)
        result[name] = {
            "count": int(finite.size), "mean": float(np.mean(finite)),
            "std": float(np.std(finite)), "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            **{f"p{q:02d}": float(value) for q, value in zip(PERCENTILES, percentiles)},
        }
    return result


def action_saturation_statistics(actions: np.ndarray) -> dict[str, dict[str, float]]:
    """Report component and timestep fractions beyond absolute thresholds."""
    absolute = np.abs(actions)
    return {
        "component_fraction": {f">{q:.2f}": float(np.mean(absolute > q)) for q in ACTION_THRESHOLDS},
        "timestep_fraction": {f">{q:.2f}": float(np.mean(np.any(absolute > q, axis=1))) for q in ACTION_THRESHOLDS},
    }


def action_overshoot_statistics(trajectories, *, top_k: int = 5) -> dict[str, object]:
    offenders = []
    for trajectory_index, trajectory in enumerate(trajectories):
        actions = split_fields(trajectory)["action"]
        for timestep, dimension in np.argwhere(np.isfinite(actions) & (np.abs(actions) > 1)):
            value = float(actions[timestep, dimension])
            offenders.append({"trajectory_index": trajectory_index, "timestep": int(timestep),
                "action_dimension": int(dimension), "action_value": value,
                "overshoot": abs(value) - 1})
    offenders.sort(key=lambda item: item["overshoot"], reverse=True)
    stats = distribution_statistics(np.asarray([item["overshoot"] for item in offenders]))
    return {key: stats[key] for key in ("count", "mean", "median", "p90", "p95", "p99", "max")} | {"top_components": offenders[:top_k]}


def diagnose_trajectory(
    trajectory: np.ndarray, *, tolerance: float = DEFAULT_TOLERANCE, top_k: int = 5
) -> dict[str, object]:
    """Compute descriptive physical-trajectory diagnostics without mutation."""
    if tolerance < 0 or top_k < 1:
        raise ValueError("tolerance must be nonnegative and top_k must be positive.")
    values = np.asarray(trajectory)
    fields = split_fields(values)
    current, previous = fields["current_state"], fields["previous_state"]
    goal, action = fields["goal"], fields["action"]
    previous_errors = _norm_rows(previous[1:] - current[:-1])
    goal_drifts = _norm_rows(goal - goal[:1])
    state_deltas = _norm_rows(current[1:] - current[:-1])
    component_oob = (action < -1) | (action > 1)
    timestep_oob = component_oob.any(axis=1)
    overshoot = np.maximum(np.abs(action) - 1, 0)

    def finite_reduction(data, operation, default=0.0):
        finite = data[np.isfinite(data)]
        return float(operation(finite)) if finite.size else float(default)

    jump_order = np.argsort(np.nan_to_num(state_deltas, nan=np.inf))[::-1][:top_k]
    previous_max = finite_reduction(previous_errors, np.max)
    max_positions = np.flatnonzero(previous_errors == previous_max)
    continuity = distribution_statistics(state_deltas)
    return {
        "shape": tuple(values.shape),
        "length": len(values),
        "dtype": str(values.dtype),
        "values": _safe_stats(values),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "action_min": _safe_stats(action)["min"],
        "action_max": _safe_stats(action)["max"],
        "action_oob_count": int(component_oob.sum()),
        "action_oob_fraction": float(component_oob.mean()),
        "action_oob_timestep_fraction": float(timestep_oob.mean()),
        "action_max_overshoot": finite_reduction(overshoot, np.max),
        "action_oob_timesteps": np.flatnonzero(timestep_oob).tolist(),
        "previous_alignment": {
            "mean": finite_reduction(previous_errors, np.mean),
            "median": finite_reduction(previous_errors, np.median),
            "max": previous_max,
            "above_tolerance": int(np.sum(previous_errors > tolerance)),
            "max_timestep": int(max_positions[0] + 1) if max_positions.size else None,
            "errors": previous_errors,
        },
        "goal_drift": {
            "mean": finite_reduction(goal_drifts, np.mean),
            "max": finite_reduction(goal_drifts, np.max),
            "above_tolerance": int(np.sum(goal_drifts > tolerance)),
            "errors": goal_drifts,
        },
        "state_continuity": {
            **continuity,
            "top_jumps": [
                {"timestep": int(index + 1), "delta": float(state_deltas[index])}
                for index in jump_order
            ],
            "deltas": state_deltas,
        },
    }


def column_names() -> list[str]:
    return (
        ["timestep"]
        + [f"current_state_{i:02d}" for i in range(CURRENT_STATE_DIM)]
        + [f"previous_state_{i:02d}" for i in range(CURRENT_STATE_DIM)]
        + [f"goal_{i:02d}" for i in range(GOAL_DIM)]
        + [f"action_{i:02d}" for i in range(ACTION_SHAPE[0])]
    )


def export_csv(trajectory: np.ndarray, path: str | Path) -> None:
    fields = split_fields(trajectory)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(column_names())
        for timestep in range(len(trajectory)):
            writer.writerow(
                [timestep]
                + fields["current_state"][timestep].tolist()
                + fields["previous_state"][timestep].tolist()
                + fields["goal"][timestep].tolist()
                + fields["action"][timestep].tolist()
            )


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.9g}" for value in values) + "]"


def format_report(
    trajectory: np.ndarray,
    *,
    trajectory_index: int = 0,
    tolerance: float = DEFAULT_TOLERANCE,
    top_k: int = 5,
) -> str:
    fields = split_fields(trajectory)
    report = diagnose_trajectory(trajectory, tolerance=tolerance, top_k=top_k)
    previous = report["previous_alignment"]
    drift = report["goal_drift"]
    continuity = report["state_continuity"]
    lines = [
        f"trajectory_index: {trajectory_index}",
        f"shape: {report['shape']}",
        f"length: {report['length']}",
        f"dtype: {report['dtype']}",
        "values: " + ", ".join(f"{k}={v:.9g}" for k, v in report["values"].items()),
        f"nan_count: {report['nan_count']}",
        f"inf_count: {report['inf_count']}",
        f"action_range: [{report['action_min']:.9g}, {report['action_max']:.9g}]",
        f"action_oob: count={report['action_oob_count']}, component_fraction={report['action_oob_fraction']:.9g}, timestep_fraction={report['action_oob_timestep_fraction']:.9g}, max_overshoot={report['action_max_overshoot']:.9g}",
        f"action_oob_timesteps: {report['action_oob_timesteps']}",
        f"previous_alignment: mean={previous['mean']:.9g}, median={previous['median']:.9g}, max={previous['max']:.9g}, above_tolerance={previous['above_tolerance']}, max_timestep={previous['max_timestep']}",
        f"goal_drift: mean={drift['mean']:.9g}, max={drift['max']:.9g}, above_tolerance={drift['above_tolerance']}",
        f"state_continuity: count={continuity['count']}, mean={continuity['mean']:.9g}, std={continuity['std']:.9g}, median={continuity['median']:.9g}, p90={continuity['p90']:.9g}, p95={continuity['p95']:.9g}, p99={continuity['p99']:.9g}, max={continuity['max']:.9g}",
        f"largest_state_jumps: {continuity['top_jumps']}",
        "",
        "timesteps:",
    ]
    for timestep in range(len(trajectory)):
        lines.extend(
            [
                f"timestep: {timestep}",
                f"  current_state[18]: {_format_vector(fields['current_state'][timestep])}",
                f"  previous_state[18]: {_format_vector(fields['previous_state'][timestep])}",
                f"  goal[3]: {_format_vector(fields['goal'][timestep])}",
                f"  action[4]: {_format_vector(fields['action'][timestep])}",
            ]
        )
    return "\n".join(lines) + "\n"


def export_txt(trajectory: np.ndarray, path: str | Path, **kwargs) -> None:
    Path(path).write_text(format_report(trajectory, **kwargs), encoding="utf-8")


def scan_trajectories(
    trajectories: tuple[np.ndarray, ...], *, tolerance=DEFAULT_TOLERANCE, top_k=5
) -> dict[str, list[tuple[int, float]] | list[int]]:
    reports = [diagnose_trajectory(item, tolerance=tolerance, top_k=top_k) for item in trajectories]

    def ranking(metric):
        return sorted(enumerate(metric(report) for report in reports), key=lambda x: x[1], reverse=True)[:top_k]

    return {
        "largest_previous_state_mismatch": ranking(lambda r: r["previous_alignment"]["max"]),
        "largest_state_jump": ranking(lambda r: r["state_continuity"]["max"]),
        "highest_action_oob_fraction": ranking(lambda r: r["action_oob_fraction"]),
        "largest_goal_drift": ranking(lambda r: r["goal_drift"]["max"]),
        "nan_or_inf_trajectories": [
            index for index, report in enumerate(reports) if report["nan_count"] or report["inf_count"]
        ],
    }


def aggregate_diagnostics(trajectories: tuple[np.ndarray, ...]) -> dict[str, object]:
    """Aggregate all valid timestep deltas rather than trajectory-level means."""
    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    fields = [split_fields(item) for item in trajectories]
    reports = [diagnose_trajectory(item) for item in trajectories]
    state = np.concatenate([item["current_state"] for item in fields])
    action = np.concatenate([item["action"] for item in fields])
    deltas = np.concatenate([item["state_continuity"]["deltas"] for item in reports])
    saturation = action_saturation_statistics(action)
    return {
        "trajectory_count": len(trajectories),
        "timestep_count": int(sum(len(item) for item in trajectories)),
        "nan_count": int(sum(np.isnan(item).sum() for item in trajectories)),
        "inf_count": int(sum(np.isinf(item).sum() for item in trajectories)),
        "state_delta": distribution_statistics(deltas),
        "state_mean": [item["mean"] for item in _dimension_statistics(state, "state").values()],
        "state_std": [item["std"] for item in _dimension_statistics(state, "state").values()],
        "action_mean": [item["mean"] for item in _dimension_statistics(action, "action").values()],
        "action_std": [item["std"] for item in _dimension_statistics(action, "action").values()],
        "action_saturation": saturation,
        "action_oob_fraction": saturation["component_fraction"][">1.00"],
        "action_oob_timestep_fraction": saturation["timestep_fraction"][">1.00"],
        "previous_alignment_mean": float(np.mean([r["previous_alignment"]["mean"] for r in reports])),
        "previous_alignment_max": float(np.max([r["previous_alignment"]["max"] for r in reports])),
        "goal_drift_mean": float(np.mean([r["goal_drift"]["mean"] for r in reports])),
        "goal_drift_max": float(np.max([r["goal_drift"]["max"] for r in reports])),
        "state_dimensions": _dimension_statistics(state, "state"),
        "action_dimensions": _dimension_statistics(action, "action"),
    }


def action_oob_by_timestep(trajectories, *, top_k: int = 5) -> dict[str, object]:
    """Measure action-bound violations at each available trajectory timestep."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    actions = [split_fields(item)["action"] for item in trajectories]
    horizon = max((len(item) for item in actions), default=0)
    per_timestep = []
    for timestep in range(horizon):
        available = [item[timestep] for item in actions if timestep < len(item)]
        values = np.stack(available)
        oob = np.isfinite(values) & (np.abs(values) > 1)
        component_count = int(oob.sum())
        trajectory_count = int(np.any(oob, axis=1).sum())
        valid_count = len(available)
        per_timestep.append({
            "timestep": timestep,
            "valid_trajectory_count": valid_count,
            "component_count": component_count,
            "component_fraction": float(component_count / (valid_count * ACTION_SHAPE[0])),
            "trajectory_count": trajectory_count,
            "trajectory_fraction": float(trajectory_count / valid_count),
        })

    def highest(key):
        return sorted(per_timestep, key=lambda item: (-item[key], item["timestep"]))[:top_k]

    return {
        "per_timestep": per_timestep,
        "top_trajectory_fraction_timesteps": highest("trajectory_fraction"),
        "top_component_fraction_timesteps": highest("component_fraction"),
    }


def state_reference_diagnostics(
    real_state: np.ndarray,
    generated_state: np.ndarray,
    *,
    std_tolerance: float = CONSTANT_STATE_STD_TOL,
    absolute_tolerance: float = CONSTANT_STATE_ABS_TOL,
) -> tuple[dict[str, object], dict[str, object]]:
    """Separate constant-reference deviations from percentile support checks."""
    if std_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Constant-state tolerances must be nonnegative.")
    constant, percentile = {}, {}
    for dimension in range(CURRENT_STATE_DIM):
        rv = real_state[:, dimension]
        rv = rv[np.isfinite(rv)]
        gv = generated_state[:, dimension]
        gv = gv[np.isfinite(gv)]
        name = f"state_{dimension:02d}"
        real_std = float(np.std(rv))
        if real_std < std_tolerance:
            reference = float(np.median(rv))
            deviations = np.abs(gv - reference)
            stats = distribution_statistics(deviations)
            constant[name] = {
                "constant_reference": True,
                "real_reference_value": reference,
                "real_std": real_std,
                "generated_abs_deviation": {
                    key: stats[key] for key in ("count", "mean", "median", "p90", "p95", "p99", "max")
                },
                "fraction_above_tolerance": float(np.mean(deviations > absolute_tolerance)),
                "tolerance": float(absolute_tolerance),
            }
            continue
        low, high = np.percentile(rv, (1, 99))
        percentile[name] = {
            "constant_reference": False,
            "real_p01": float(low), "real_p99": float(high),
            "below_real_p01": float(np.mean(gv < low)),
            "above_real_p99": float(np.mean(gv > high)),
            "largest_below_real_min": float(max(np.min(rv) - np.min(gv), 0)),
            "largest_above_real_max": float(max(np.max(gv) - np.max(rv), 0)),
        }
    return constant, percentile


def compare_datasets(
    real,
    generated,
    *,
    top_k: int = 5,
    constant_state_std_tolerance: float = CONSTANT_STATE_STD_TOL,
    constant_state_abs_tolerance: float = CONSTANT_STATE_ABS_TOL,
) -> dict[str, object]:
    """Compare provided datasets against real distribution references."""
    real_stats, generated_stats = aggregate_diagnostics(real), aggregate_diagnostics(generated)
    real_fields = [split_fields(item) for item in real]
    generated_fields = [split_fields(item) for item in generated]
    real_state = np.concatenate([item["current_state"] for item in real_fields])
    generated_state = np.concatenate([item["current_state"] for item in generated_fields])
    real_action = np.concatenate([item["action"] for item in real_fields])
    generated_action = np.concatenate([item["action"] for item in generated_fields])
    generated_deltas = np.concatenate([diagnose_trajectory(item)["state_continuity"]["deltas"] for item in generated])
    generated_deltas = generated_deltas[np.isfinite(generated_deltas)]
    references = {name: real_stats["state_delta"][name] for name in ("p90", "p95", "p99")}
    tail = {}
    for name, reference in references.items():
        count = int(np.sum(generated_deltas > reference))
        tail[f">real_{name}"] = {"count": count, "total": int(generated_deltas.size),
            "fraction": float(count / generated_deltas.size) if generated_deltas.size else 0.0}
    constant_state, state_support = state_reference_diagnostics(
        real_state,
        generated_state,
        std_tolerance=constant_state_std_tolerance,
        absolute_tolerance=constant_state_abs_tolerance,
    )
    action_support = {}
    for dimension in range(ACTION_SHAPE[0]):
        rv = real_action[:, dimension]; rv = rv[np.isfinite(rv)]
        gv = generated_action[:, dimension]; gv = gv[np.isfinite(gv)]
        low, high = np.percentile(rv, (1, 99))
        action_support[f"action_{dimension:02d}"] = {"real_p01": float(low), "real_p99": float(high),
            "generated_outside_real_p01_p99_fraction": float(np.mean((gv < low) | (gv > high)))}
    return {
        "metadata": {"real_label": "provided real reference", "generated_label": "provided generated dataset",
            "interpretation": "Descriptive distribution diagnostics only; these metrics do not prove simulator or physical validity."},
        "real": real_stats, "generated": generated_stats,
        "state_delta_reference": {f"real_{key}": value for key, value in references.items()},
        "generated_tail_exceedance": tail,
        "constant_state_diagnostics": constant_state,
        "state_support_exceedance": state_support,
        "action_support_exceedance": action_support,
        "generated_oob_by_timestep": action_oob_by_timestep(generated, top_k=top_k),
        "generated_action_overshoot": action_overshoot_statistics(generated, top_k=top_k),
    }
