#!/usr/bin/env python3
"""Inspect stored DiffCRL physical trajectories without modifying them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continual.trajectory_inspection import (  # noqa: E402
    DEFAULT_TOLERANCE,
    action_oob_by_timestep,
    compare_datasets,
    diagnose_trajectory,
    export_csv,
    export_txt,
    format_report,
    load_trajectories,
    scan_trajectories,
    split_fields,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, help="A real .npz or packed .npy file")
    result.add_argument("--traj-index", type=int, default=0)
    result.add_argument("--output-txt", type=Path)
    result.add_argument("--output-csv", type=Path)
    result.add_argument("--output-json", type=Path, help="Write comparison results as JSON")
    result.add_argument("--scan-all", action="store_true")
    result.add_argument("--top-k", type=int, default=5)
    result.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    result.add_argument("--compare-real", type=Path)
    result.add_argument("--compare-generated", type=Path)
    result.add_argument("--plot", action="store_true")
    result.add_argument("--state-dims", type=int, nargs="*", default=[0, 1, 2])
    return result


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def _plot_single(trajectory, title, state_dims):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("--plot requires matplotlib; no dependency was added.") from error
    fields = split_fields(trajectory)
    report = diagnose_trajectory(trajectory)
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    axes[0].plot(fields["action"])
    axes[0].set_ylabel("action")
    for dimension in state_dims:
        if not 0 <= dimension < fields["current_state"].shape[1]:
            raise SystemExit(f"Invalid state dimension: {dimension}")
        axes[1].plot(fields["current_state"][:, dimension], label=f"state_{dimension:02d}")
    axes[1].legend()
    axes[1].set_ylabel("state")
    axes[2].plot(range(1, len(trajectory)), report["state_continuity"]["deltas"])
    axes[2].set_ylabel("state delta L2")
    axes[3].plot(range(1, len(trajectory)), report["previous_alignment"]["errors"])
    axes[3].set_ylabel("previous error L2")
    axes[3].set_xlabel("timestep")
    figure.suptitle(title)
    figure.tight_layout()
    plt.show()


def _plot_comparison(real, generated):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise SystemExit("--plot requires matplotlib; no dependency was added.") from error
    real_actions = np.concatenate([split_fields(x)["action"].ravel() for x in real])
    gen_actions = np.concatenate([split_fields(x)["action"].ravel() for x in generated])
    real_delta = np.concatenate([diagnose_trajectory(x)["state_continuity"]["deltas"] for x in real])
    gen_delta = np.concatenate([diagnose_trajectory(x)["state_continuity"]["deltas"] for x in generated])
    overshoot = np.maximum(np.abs(gen_actions) - 1, 0)
    overshoot = overshoot[overshoot > 0]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(real_delta, bins=50, alpha=0.6, density=True, label="real")
    axes[0].hist(gen_delta, bins=50, alpha=0.6, density=True, label="generated")
    axes[0].set_title("State-step L2 delta")
    axes[1].hist(real_actions, bins=50, alpha=0.6, density=True, label="real")
    axes[1].hist(gen_actions, bins=50, alpha=0.6, density=True, label="generated")
    axes[1].set_title("Action components")
    axes[2].hist(overshoot, bins=50, density=True, label="generated OOB")
    axes[2].set_title("Generated action overshoot")
    for axis in axes:
        axis.legend()
    figure.suptitle("Descriptive diagnostics only — not a physical-validity test")
    figure.tight_layout()
    temporal = action_oob_by_timestep(generated)["per_timestep"]
    temporal_figure, temporal_axis = plt.subplots(figsize=(10, 4))
    temporal_axis.plot(
        [item["timestep"] for item in temporal],
        [item["trajectory_fraction"] for item in temporal],
        label="trajectory OOB fraction",
    )
    temporal_axis.plot(
        [item["timestep"] for item in temporal],
        [item["component_fraction"] for item in temporal],
        label="component OOB fraction",
    )
    temporal_axis.set(xlabel="timestep", ylabel="fraction", title="Generated action OOB by timestep")
    temporal_axis.legend()
    temporal_figure.tight_layout()
    plt.show()


def _comparison_table(result):
    real, generated = result["real"], result["generated"]
    rows = [
        ("trajectories", real["trajectory_count"], generated["trajectory_count"]),
        ("timesteps", real["timestep_count"], generated["timestep_count"]),
    ]
    for key in ("mean", "median", "p95", "p99", "max"):
        rows.append((f"state delta {key}", real["state_delta"][key], generated["state_delta"][key]))
    for threshold in (">0.90", ">0.95", ">0.99", ">1.00"):
        rows.append((f"|a| {threshold} component frac", real["action_saturation"]["component_fraction"][threshold], generated["action_saturation"]["component_fraction"][threshold]))
    rows.extend([
        ("action OOB timestep frac", real["action_oob_timestep_fraction"], generated["action_oob_timestep_fraction"]),
        ("NaN count", real["nan_count"], generated["nan_count"]),
        ("Inf count", real["inf_count"], generated["inf_count"]),
    ])
    print(f'{"Metric":<34} {"Real":>14} {"Generated":>14}')
    print("-" * 64)
    for name, real_value, generated_value in rows:
        def display(value):
            return "n/a" if value is None else f"{value:.8g}" if isinstance(value, float) else str(value)
        print(f"{name:<34} {display(real_value):>14} {display(generated_value):>14}")


def _focused_comparison_summary(result):
    constant = result["constant_state_diagnostics"]
    if constant:
        print("\nNear-constant real state dimensions:")
        for name, item in constant.items():
            deviation = item["generated_abs_deviation"]
            print(
                f"{name} ref={item['real_reference_value']:.8g} "
                f"max_abs_dev={deviation['max']:.8g} "
                f"frac>|{item['tolerance']:.0e}|={item['fraction_above_tolerance']:.3f}"
            )
    print("\nTop generated OOB timesteps:")
    for item in result["generated_oob_by_timestep"]["top_trajectory_fraction_timesteps"]:
        print(
            f"t={item['timestep']:<4} traj_frac={item['trajectory_fraction']:.6g} "
            f"component_frac={item['component_fraction']:.6g}"
        )


def main(args: argparse.Namespace) -> int:
    comparing = args.compare_real is not None or args.compare_generated is not None
    if comparing:
        if args.compare_real is None or args.compare_generated is None:
            raise SystemExit("Comparison requires both --compare-real and --compare-generated.")
        real = load_trajectories(args.compare_real).trajectories
        generated = load_trajectories(args.compare_generated).trajectories
        comparison = compare_datasets(real, generated, top_k=args.top_k)
        comparison["metadata"].update({"real_path": str(args.compare_real), "generated_path": str(args.compare_generated)})
        print(comparison["metadata"]["interpretation"])
        _comparison_table(comparison)
        _focused_comparison_summary(comparison)
        print("\nDetailed comparison:")
        _print_json(comparison)
        if args.output_json:
            args.output_json.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.plot:
            _plot_comparison(real, generated)
    if args.input is not None:
        loaded = load_trajectories(args.input)
        if not 0 <= args.traj_index < len(loaded.trajectories):
            raise SystemExit(f"--traj-index must be in [0, {len(loaded.trajectories) - 1}].")
        trajectory = loaded.trajectories[args.traj_index]
        if args.scan_all:
            _print_json(scan_trajectories(loaded.trajectories, tolerance=args.tolerance, top_k=args.top_k))
        else:
            print(format_report(trajectory, trajectory_index=args.traj_index, tolerance=args.tolerance, top_k=args.top_k), end="")
        if args.output_txt:
            export_txt(trajectory, args.output_txt, trajectory_index=args.traj_index, tolerance=args.tolerance, top_k=args.top_k)
        if args.output_csv:
            export_csv(trajectory, args.output_csv)
        if args.plot:
            _plot_single(trajectory, f"{args.input.name}: trajectory {args.traj_index}", args.state_dims)
    elif not comparing:
        if args.output_json:
            raise SystemExit("--output-json requires a real-vs-generated comparison.")
        raise SystemExit("Provide --input or both comparison inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
