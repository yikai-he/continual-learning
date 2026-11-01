#!/usr/bin/env python3
"""Aggregate only hash-validated, completed Phase 7B experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.support.reproducibility import file_hash


def collect(root: Path) -> list[dict]:
    rows = []
    for marker_path in sorted(root.glob("*/experiment.complete.json")):
        marker = json.loads(marker_path.read_text())
        if marker.get("kind") != "flow_matching_ablation_completion_v1":
            raise ValueError(f"Invalid completion marker: {marker_path}")
        for artifact in marker["artifacts"].values():
            path = Path(artifact["path"])
            if not path.is_file() or file_hash(path) != artifact["sha256"]:
                raise ValueError(f"Completed artifact changed or is missing: {path}")
        directory = marker_path.parent
        generation = json.loads(Path(marker["artifacts"]["generation_metrics"]["path"]).read_text())
        diagnostics = json.loads(Path(marker["artifacts"]["diagnostics"]["path"]).read_text())
        bc = json.loads(Path(marker["artifacts"]["bc_results"]["path"]).read_text())
        summary = diagnostics["summary"]
        rows.append({
            "experiment_id": directory.name,
            "generation_seconds": generation["seconds"], "nfe": generation["nfe"],
            "checkpoint_sha256": generation["checkpoint_sha256"],
            "action_oob_component_rate": summary["action_oob_component_rate"],
            "action_oob_timestep_rate": summary["action_oob_timestep_rate"],
            "state_mean_absolute_discrepancy": summary["state_mean_absolute_discrepancy"],
            "action_mean_absolute_discrepancy": summary["action_mean_absolute_discrepancy"],
            "expert_action_mse": diagnostics["expert_action_disagreement"]["overall"]["mse"],
            "bc_validation_action_mse": bc["fit"]["validation_action_mse"],
            "bc_success_rate": bc["evaluation"]["success_rate"],
            "bc_mean_return": bc["evaluation"]["mean_return"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root.resolve())
    if not rows:
        raise ValueError("No completed experiments found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output.with_suffix(".json")
    if json_path.exists():
        raise FileExistsError(json_path)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Aggregated {len(rows)} completed experiments into {args.output}")


if __name__ == "__main__":
    main()
