#!/usr/bin/env python3
"""Train continual BC from immutable saved stage datasets without replay generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continual.bc_policy import GeneralPolicy, fit_bc  # noqa: E402
from continual_manipulation.evaluation import EvaluationMatrix, evaluate_stage  # noqa: E402
from src.continual.stage_dataset import bc_dataset  # noqa: E402
from continual_manipulation.sequences import TaskSequence  # noqa: E402
from src.support.io import atomic_replace_json, write_json  # noqa: E402
from src.support.reproducibility import file_hash, state_hash  # noqa: E402


def load_config(path: Path) -> dict:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"experiment", "source_run", "bc", "runtime"}
    if not isinstance(values, dict) or set(values) != required:
        raise ValueError(f"Configuration fields must be exactly {sorted(required)}.")
    if values["experiment"] != "fixed_dataset_bc":
        raise ValueError("experiment must be fixed_dataset_bc.")
    bc = values["bc"]
    expected_bc = {
        "loss", "checkpoint_selection", "epochs", "batch_size",
        "learning_rate", "initial_log_std", "activation", "hidden_dim",
    }
    if not isinstance(bc, dict) or set(bc) != expected_bc:
        raise ValueError(f"bc fields must be exactly {sorted(expected_bc)}.")
    if bc["loss"] != "squashed_gaussian_nll":
        raise ValueError("Fixed-dataset ablation requires squashed_gaussian_nll.")
    if bc["checkpoint_selection"] != "validation_physical_mse":
        raise ValueError("Primary checkpoint selection must be validation_physical_mse.")
    runtime = values["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"seed", "device", "output"}:
        raise ValueError("runtime requires exactly seed, device, and output.")
    if runtime["device"] != "cpu" or not isinstance(runtime["seed"], int):
        raise ValueError("This controlled runner requires an integer seed and CPU.")
    source = (ROOT / values["source_run"]).resolve()
    output = (ROOT / runtime["output"]).resolve()
    if not (source / "RUN_COMPLETE").is_file():
        raise ValueError("source_run must be a completed continual run.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    values["source_run"] = source
    runtime["output"] = output
    return values


def source_contract(config: dict) -> dict:
    source = config["source_run"]
    resolved = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    tasks = resolved["continual"]["tasks"]
    if resolved["bc"]["loss"] != "deterministic_mse":
        raise ValueError("Source run is not the stochastic-demonstration MSE baseline.")
    if resolved["continual"].get("expert_action_mode", "deterministic") != "stochastic":
        raise ValueError("Source run does not use stochastic expert actions.")
    expected = config["bc"]
    # initial_log_std belongs only to the independently initialized NLL head;
    # it is an explicit ablation variable, not a source-dataset compatibility field.
    for field in ("epochs", "batch_size", "learning_rate", "activation", "hidden_dim"):
        if resolved["bc"][field] != expected[field]:
            raise ValueError(f"BC setting differs from source: {field}.")
    artifacts = []
    reports = []
    for stage, task in enumerate(tasks):
        directory = source / f"stage_{stage}_{task}"
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append(report)
        for index in range(stage + 1):
            path = directory / f"data_task_{index}.npy"
            name = tasks[index]
            digest = file_hash(path)
            if report["sources"][name]["data_sha256"] != digest:
                raise ValueError(f"Source data hash mismatch: {path}")
            values = np.load(path, mmap_mode="r")
            split = report["splits"][name]
            train_ids, validation_ids = split["train_ids"], split["validation_ids"]
            if sorted(train_ids + validation_ids) != list(range(len(values))):
                raise ValueError(f"Invalid saved split for {path}.")
            artifacts.append({
                "stage": stage, "task_index": index, "task": name,
                "path": str(path), "sha256": digest, "split": split,
            })
    return {"resolved": resolved, "tasks": tasks, "reports": reports, "artifacts": artifacts}


def stage_datasets(contract: dict, stage: int):
    source, tasks = Path(contract["resolved"]["runtime"]["run_dir"]), contract["tasks"]
    directory = source / f"stage_{stage}_{tasks[stage]}"
    report = contract["reports"][stage]
    trains, validations = [], []
    for index in range(stage + 1):
        values = torch.from_numpy(np.load(directory / f"data_task_{index}.npy")).float()
        split = report["splits"][tasks[index]]
        trains.append(values[split["train_ids"]])
        validations.append(values[split["validation_ids"]])
    return bc_dataset(torch.cat(trains)), bc_dataset(torch.cat(validations))


def run(config: dict, contract: dict) -> None:
    source, output = config["source_run"], config["runtime"]["output"]
    resolved, bc = contract["resolved"], config["bc"]
    sequence = TaskSequence.from_names(contract["tasks"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "source_artifacts.json", {
        "source_run": str(source),
        "source_config_sha256": file_hash(source / "config.yaml"),
        "artifacts": contract["artifacts"],
    })
    write_json(output / "config.json", {
        "experiment": "fixed_dataset_bc", "source_run": str(source),
        "bc": bc, "runtime": {**config["runtime"], "output": str(output)},
        "evaluation": resolved["evaluation"], "tasks": contract["tasks"],
    })
    torch.set_num_threads(1)
    torch.manual_seed(config["runtime"]["seed"])
    source_policy = GeneralPolicy.load(
        source / f"general_policy_after_task_0_{contract['tasks'][0]}.pt",
        expected_objective="deterministic_mse",
    )
    policy = GeneralPolicy(
        source_policy.action_low.numpy(), source_policy.action_high.numpy(),
        initial_log_std=bc["initial_log_std"], activation=bc["activation"],
        hidden_dim=bc["hidden_dim"], objective=bc["loss"],
    )
    initial_hash = state_hash(policy)
    matrix = EvaluationMatrix(sequence)
    for stage, task in enumerate(contract["tasks"]):
        before_hash = state_hash(policy)
        train, validation = stage_datasets(contract, stage)
        optimizer = torch.optim.Adam(policy.parameters(), lr=bc["learning_rate"])
        started = time.perf_counter()
        losses = fit_bc(
            policy, train, validation, epochs=bc["epochs"],
            seed=config["runtime"]["seed"] + stage,
            batch_size=bc["batch_size"], learning_rate=bc["learning_rate"],
            allow_out_of_bounds_targets=True, optimizer=optimizer,
            bc_loss=bc["loss"], progress=True,
            checkpoint_selection=bc["checkpoint_selection"],
        )
        training_seconds = time.perf_counter() - started
        checkpoint = output / f"general_policy_after_task_{stage}_{task}.pt"
        policy.save(checkpoint, optimizer=optimizer, metadata={
            "stage": stage, "task": task, "source_run": str(source),
            "source_stage_report_sha256": file_hash(
                source / f"stage_{stage}_{task}" / "report.json"
            ),
            "checkpoint_selection": bc["checkpoint_selection"],
            "initial_state_sha256": initial_hash,
            "policy_before_state_sha256": before_hash,
            "policy_after_state_sha256": state_hash(policy),
        })
        evaluate_stage(
            policy, matrix, stage, episodes=resolved["evaluation"]["episodes"],
            seed=resolved["evaluation"]["seed"],
            evaluation_mode=resolved["evaluation"]["mode"],
            task_set_seed=resolved["evaluation"]["task_set_seed"],
            reward_function_version=resolved["environment"]["reward_function_version"],
            backend=resolved["environment"]["backend"], progress=True,
        )
        matrix.stage_provenance[str(stage)] = {
            "policy_checkpoint": str(checkpoint),
            "policy_sha256": file_hash(checkpoint),
            "source_stage": str(source / f"stage_{stage}_{task}"),
        }
        report = {
            "stage": stage, "task": task, "bc_losses": losses,
            "bc_training_seconds": training_seconds,
            "policy_before_state_sha256": before_hash,
            "policy_after_state_sha256": state_hash(policy),
            "policy_checkpoint": str(checkpoint),
            "source_splits": contract["reports"][stage]["splits"],
        }
        stage_dir = output / f"stage_{stage}_{task}"
        stage_dir.mkdir()
        write_json(stage_dir / "report.json", report)
        write_json(stage_dir / "evaluation_matrix.json", matrix.to_dict())
        atomic_replace_json(output / "evaluation_matrix.json", matrix.to_dict())
        write_json(output / f"stage_{stage}_COMPLETE.json", {
            "stage": stage, "task": task, "policy_sha256": file_hash(checkpoint),
        })
    write_json(output / "RUN_COMPLETE", {"stages": sequence.num_tasks})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    contract = source_contract(config)
    print(json.dumps({
        "source_run": str(config["source_run"]),
        "output": str(config["runtime"]["output"]),
        "tasks": contract["tasks"], "artifacts": len(contract["artifacts"]),
        "source_hashes_verified": True,
    }, indent=2))
    if not args.validate_only:
        run(config, contract)


if __name__ == "__main__":
    main()
