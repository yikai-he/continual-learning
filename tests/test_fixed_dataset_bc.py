"""Immutable saved-dataset contracts for the continual NLL BC runner."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.experiments.train_fixed_dataset_bc import (
    load_config,
    source_contract,
    stage_datasets,
)
from src.support.reproducibility import file_hash
from src.continual.bc_policy import GeneralPolicy


TASKS = ["kuka-drawer-close-v3", "kuka-window-close-v3"]


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "RUN_COMPLETE").write_text("{}\n")
    resolved = {
        "environment": {"backend": "kuka-v3", "reward_function_version": "v2"},
        "bc": {
            "loss": "deterministic_mse", "epochs": 3, "batch_size": 8,
            "learning_rate": 0.001, "initial_log_std": -3.0,
            "activation": "relu", "hidden_dim": 256,
        },
        "continual": {"tasks": TASKS, "expert_action_mode": "stochastic"},
        "evaluation": {"episodes": 2, "seed": 10, "mode": "fixed-tasks", "task_set_seed": 10},
        "runtime": {"run_dir": str(source)},
    }
    (source / "config.yaml").write_text(yaml.safe_dump(resolved))
    for stage, task in enumerate(TASKS):
        directory = source / f"stage_{stage}_{task}"
        directory.mkdir()
        sources, splits = {}, {}
        for index in range(stage + 1):
            values = np.full((5, 200, 43), stage * 10 + index, dtype=np.float32)
            path = directory / f"data_task_{index}.npy"
            np.save(path, values, allow_pickle=False)
            sources[TASKS[index]] = {"data_sha256": file_hash(path)}
            splits[TASKS[index]] = {
                "train_ids": [0, 1, 2, 3], "validation_ids": [4],
                "trajectories": 5, "train_samples": 800, "validation_samples": 200,
            }
        (directory / "report.json").write_text(json.dumps({"sources": sources, "splits": splits}))
    return source


def make_config(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "fixed.yaml"
    path.write_text(yaml.safe_dump({
        "experiment": "fixed_dataset_bc", "source_run": str(source),
        "bc": {
            "loss": "squashed_gaussian_nll",
            "checkpoint_selection": "validation_physical_mse",
            "epochs": 3, "batch_size": 8, "learning_rate": 0.001,
            "initial_log_std": -3.0, "activation": "relu", "hidden_dim": 256,
        },
        "runtime": {"seed": 0, "device": "cpu", "output": str(tmp_path / "new")},
    }))
    return path


def test_source_hashes_and_saved_splits_reconstruct_exact_datasets(tmp_path):
    source = make_source(tmp_path)
    config = load_config(make_config(tmp_path, source))
    contract = source_contract(config)
    assert len(contract["artifacts"]) == 3
    train, validation = stage_datasets(contract, 1)
    assert train.tensors[0].shape == (1600, 39)
    assert validation.tensors[0].shape == (400, 39)
    assert set(train.tensors[0][:, 0].tolist()) == {10.0, 11.0}
    assert set(validation.tensors[0][:, 0].tolist()) == {10.0, 11.0}


def test_source_mutation_and_output_collision_are_rejected(tmp_path):
    source = make_source(tmp_path)
    config_path = make_config(tmp_path, source)
    config = load_config(config_path)
    np.save(
        source / f"stage_1_{TASKS[1]}" / "data_task_0.npy",
        np.ones((5, 200, 43), dtype=np.float32), allow_pickle=False,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        source_contract(config)
    config["runtime"]["output"].mkdir()
    with pytest.raises(FileExistsError):
        load_config(config_path)


def test_initial_log_std_is_independent_of_source_and_mean_initialization(tmp_path):
    source = make_source(tmp_path)
    config_path = make_config(tmp_path, source)
    values = yaml.safe_load(config_path.read_text())
    values["bc"]["initial_log_std"] = -1.0
    config_path.write_text(yaml.safe_dump(values))
    config = load_config(config_path)
    source_contract(config)

    low, high = -np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    torch.manual_seed(0)
    baseline = GeneralPolicy(low, high, initial_log_std=-3.0)
    torch.manual_seed(0)
    ablation = GeneralPolicy(low, high, initial_log_std=-1.0)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            baseline.shared_network.state_dict().values(),
            ablation.shared_network.state_dict().values(),
        )
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            baseline.mean_head.state_dict().values(),
            ablation.mean_head.state_dict().values(),
        )
    )
    assert torch.count_nonzero(ablation.log_std_head.weight) == 0
    assert torch.all(ablation.log_std_head.bias == -1.0)
