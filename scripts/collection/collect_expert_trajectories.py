"""Collect and validate a small standalone expert trajectory dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.continual.expert_collection import collect_expert_dataset, load_collection_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_collection_config(args.config)
    report = collect_expert_dataset(config)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
