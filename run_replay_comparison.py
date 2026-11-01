"""Run matched No Replay, Real Replay, and DDPM Replay policy experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from src.config import load_config
from src.continual.bc_policy import GeneralPolicy
from src.continual.diffcrl import DiffCRLTrainer
from src.support.reproducibility import file_hash, state_hash


ARM_SETTINGS = {
    "no_replay": {
        "replay_mode": "none",
        "replay_allocation": "generator-default",
        "replay_budget": 200,
        "historical_replay_training_only": False,
    },
    "real_replay": {
        "replay_mode": "real",
        "replay_allocation": "fixed-total",
        "historical_replay_training_only": True,
    },
    "ddpm_replay": {
        "replay_mode": "diffusion",
        "replay_allocation": "fixed-total",
        "historical_replay_training_only": True,
    },
}


def comparison_configs(base, output_root: Path):
    """Derive three isolated arms while preserving all shared science settings."""
    if base.continual.replay_mode != "diffusion":
        raise ValueError("The comparison base config must select DDPM replay.")
    if base.evaluation.mode != "fixed-tasks" or base.evaluation.episodes != 50:
        raise ValueError("Comparison requires 50 fixed-task evaluation episodes.")
    result = {}
    for name, settings in ARM_SETTINGS.items():
        continual = replace(base.continual, **settings)
        runtime = replace(base.runtime, output=str(output_root), run_name=name)
        result[name] = replace(base, continual=continual, runtime=runtime)
    return result


def _current_data_hash(trainer, stage):
    task = trainer.sequence.task(stage).task_name
    return file_hash(
        trainer.output / f"stage_{stage}_{task}" / f"data_task_{stage}.npy"
    )


def _stage_report(trainer, stage):
    task = trainer.sequence.task(stage).task_name
    path = trainer.output / f"stage_{stage}_{task}" / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_comparison(base, output_root: Path, *, progress=True):
    """Execute arms stage-by-stage and enforce the matched-data contracts."""
    configs = comparison_configs(base, output_root)
    trainers = {
        name: DiffCRLTrainer(config, progress=progress)
        for name, config in configs.items()
    }
    shared_stage_zero_hash = None
    for stage in range(len(base.continual.tasks)):
        for trainer in trainers.values():
            trainer.train_task(stage)
        data_hashes = {name: _current_data_hash(t, stage) for name, t in trainers.items()}
        if len(set(data_hashes.values())) != 1:
            raise RuntimeError(f"Current-task data differs at stage {stage}: {data_hashes}")
        if stage:
            paired = {
                name: _stage_report(trainers[name], stage)
                for name in ("real_replay", "ddpm_replay")
            }
            task_names = base.continual.tasks[:stage]
            allocations = {
                name: [report["splits"][task]["trajectories"] for task in task_names]
                for name, report in paired.items()
            }
            train_counts = {
                name: sum(
                    len(split["train_ids"]) for split in report["splits"].values()
                )
                for name, report in paired.items()
            }
            validation_ids = {
                name: {
                    task: split["validation_ids"]
                    for task, split in report["splits"].items()
                }
                for name, report in paired.items()
            }
            if len({tuple(value) for value in allocations.values()}) != 1:
                raise RuntimeError(f"Historical allocations differ: {allocations}")
            if len(set(train_counts.values())) != 1:
                raise RuntimeError(f"BC training counts differ: {train_counts}")
            if validation_ids["real_replay"] != validation_ids["ddpm_replay"]:
                raise RuntimeError(f"Validation splits differ: {validation_ids}")
            for name, values in validation_ids.items():
                if any(values[task] for task in task_names):
                    raise RuntimeError(
                        f"Historical validation leakage in {name}: {values}"
                    )
        if stage == 0:
            policy_hashes = {name: state_hash(t.policy) for name, t in trainers.items()}
            if len(set(policy_hashes.values())) != 1:
                raise RuntimeError(f"Stage-0 GeneralPolicy differs: {policy_hashes}")
            shared_stage_zero_hash = next(iter(policy_hashes.values()))
            task = trainers["ddpm_replay"].sequence.task(0).task_name
            shared_checkpoint = (
                trainers["ddpm_replay"].output
                / f"general_policy_after_task_0_{task}.pt"
            )
            # All arms continue from separately loaded copies of the exact same
            # immutable checkpoint; subsequent weights and optimizers are independent.
            for trainer in trainers.values():
                policy, optimizer, metadata = GeneralPolicy.load_training(
                    shared_checkpoint
                )
                if metadata.get("stage") != 0:
                    raise RuntimeError("Shared policy checkpoint is not Stage 0.")
                trainer.policy = policy.to(trainer.config.runtime.device)
                trainer.policy_optimizer = optimizer
    for trainer in trainers.values():
        trainer.run()
    summary = {
        "kind": "policy_replay_comparison_v1",
        "seed": base.runtime.seed,
        "shared_stage_zero_policy_state_sha256": shared_stage_zero_hash,
        "arms": {
            name: {
                "run_directory": str(trainer.output),
                "evaluation_matrix": str(trainer.output / "evaluation_matrix.json"),
                "final_metrics": trainer.matrix.to_dict()["metrics_by_stage"][
                    str(trainer.sequence.num_tasks - 1)
                ],
                "final_task_mean_returns": [
                    cell["mean_return"]
                    for cell in trainer.matrix.cells[trainer.sequence.num_tasks - 1]
                ],
            }
            for name, trainer in trainers.items()
        },
    }
    path = output_root / "comparison_summary.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    overrides = {
        key: value for key, value in (("seed", args.seed), ("device", args.device))
        if value is not None
    }
    base = load_config(
        args.config, expected_experiment="diffcrl", overrides=overrides
    )
    run_comparison(base, args.output_root.resolve(), progress=not args.no_progress)


if __name__ == "__main__":
    main()
