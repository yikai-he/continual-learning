"""Validate, synchronize, inspect, or restore DiffCRL stage backups."""

import argparse
import json

from src.config import load_config
from src.support.stage_sync import (
    backup_status,
    restore_backup,
    sync_all_completed,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sync", "validate", "status", "restore"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args(argv)
    config = load_config(f"{args.source}/config.yaml", expected_experiment="diffcrl")
    tasks = config.continual.tasks
    if args.action in ("sync", "restore") and not args.destination:
        parser.error(f"{args.action} requires --destination")
    if args.action == "sync":
        result = {
            "latest_synchronized_stage": sync_all_completed(
                args.source, args.destination, tasks
            )
        }
    elif args.action == "restore":
        result = {
            "latest_restored_stage": restore_backup(
                args.source, args.destination, tasks
            )
        }
    else:
        result = backup_status(args.source, tasks)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
