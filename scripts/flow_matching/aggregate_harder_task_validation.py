#!/usr/bin/env python3
"""Hash-validate and aggregate a completed harder-task validation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def marker(root, name):
    value = json.loads((root / f"{name}.complete.json").read_text())
    for item in value["artifacts"].values():
        path = Path(item["path"])
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError(f"Invalid completed artifact: {path}")
    return value


def collect(root):
    marker(root, "validation")
    rows = []
    for source in ("real", "flow-10", "flow-50", "diffusion"):
        bc_marker = marker(root, f"bc-{source}")
        bc = json.loads(Path(bc_marker["artifacts"]["results"]["path"]).read_text())
        row = {"source": source, "bc_validation_action_mse": bc["fit"]["validation_action_mse"], "success_rate": bc["evaluation"]["success_rate"], "mean_return": bc["evaluation"]["mean_return"]}
        if source != "real":
            generated = marker(root, f"generate-{source}")
            metrics = json.loads(Path(generated["artifacts"]["metrics"]["path"]).read_text())
            diagnostics = json.loads(Path(generated["artifacts"]["diagnostics"]["path"]).read_text())
            summary = diagnostics["summary"]
            row.update(generation_seconds=metrics["seconds"], nfe=metrics["nfe"], expert_action_mse=diagnostics["expert_action_disagreement"]["overall"]["mse"], action_oob_rate=summary["action_oob_component_rate"], state_delta_mean=summary["state_delta"]["mean"], state_delta_above_real_p95=summary["state_delta_tail_vs_real"][">real_p95"]["fraction"], state_mean_discrepancy=summary["state_mean_absolute_discrepancy"], action_mean_discrepancy=summary["action_mean_absolute_discrepancy"])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    rows = collect(args.root.resolve())
    if args.output.exists() or args.output.with_suffix(".json").exists(): raise FileExistsError(args.output)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__": main()
