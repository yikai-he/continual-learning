#!/usr/bin/env python3
"""Aggregate completed four-layer SAC benchmark results without evaluation."""

import argparse
from pathlib import Path

from src.sac.architecture_benchmark import (
    TASKS, aggregate_results, write_aggregate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/sac_4layer_benchmark"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    write_aggregate(
        aggregate_results(args.root, args.tasks, args.seeds), args.json, args.csv
    )


if __name__ == "__main__":
    main()
