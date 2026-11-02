#!/usr/bin/env python3
"""Aggregate completed paired forward-transfer runs without evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.sac.forward_transfer import (
    compare_condition_summaries,
    load_summary,
    mean_sample_sd,
)


def aggregate(root: Path, seeds: list[int]) -> dict[str, object]:
    pairs = []
    for seed in seeds:
        summaries = {}
        for condition in ("random", "transferred"):
            run = root / "hammer-v3" / condition / f"seed_{seed}"
            if not (run / "RUN_COMPLETE").is_file():
                raise ValueError(f"Run is not complete: {run}")
            summaries[condition] = load_summary(run / "summary.json")
        pairs.append(
            compare_condition_summaries(
                summaries["random"], summaries["transferred"]
            )
        )
    return {
        "kind": "forward_transfer_aggregate_v1",
        "seeds": seeds,
        "paired": pairs,
        "delta_normalized_success_auc": mean_sample_sd(
            [row["delta_normalized_success_auc"] for row in pairs]
        ),
        "delta_final_success_rate": mean_sample_sd(
            [row["delta_final_success_rate"] for row in pairs]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("runs/forward_transfer")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = aggregate(args.root, args.seeds)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
