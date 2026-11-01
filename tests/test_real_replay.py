import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.config import default_config
from src.continual.diffcrl import DiffCRLTrainer, StageContext
from src.continual.diffcrl_replay import ReplayResult
from src.continual.real_replay import RealReplayRequest, load_previous_task_real_replay
from src.continual.task_sequence import TaskSequence
from src.support.reproducibility import file_hash


def physical_group(count, seed=0):
    values = torch.randn((count, 200, 43), generator=torch.Generator().manual_seed(seed))
    values[:, :, 18:36] = torch.cat((values[:, :1, :18], values[:, :-1, :18]), dim=1)
    values[:, :, 36:39] = values[:, :1, 36:39]
    values[:, :, 39:] = values[:, :, 39:].tanh()
    return values.numpy().astype(np.float32)


def test_real_replay_uses_only_original_training_ids_and_records_reuse(tmp_path):
    stage = tmp_path / "stage_0_hammer-v3"
    stage.mkdir()
    data = physical_group(5)
    data_path = stage / "data_task_0.npy"
    np.save(data_path, data, allow_pickle=False)
    report = {
        "splits": {
            "hammer-v3": {
                "train_ids": [0, 2],
                "validation_ids": [1, 3, 4],
            }
        }
    }
    (stage / "report.json").write_text(json.dumps(report), encoding="utf-8")

    groups, sources = load_previous_task_real_replay(
        RealReplayRequest(
            stage=1,
            task_names=("hammer-v3", "button-press-v3"),
            trajectory_counts={0: 7},
            run_directory=tmp_path,
            runtime_seed=0,
        )
    )

    source = sources["hammer-v3"]
    assert groups[0].shape == (7, 200, 43)
    assert set(source["source_trajectory_ids"]) <= {0, 2}
    assert not set(source["source_trajectory_ids"]) & {1, 3, 4}
    assert sum(source["reuse_multiplicity"].values()) == 7
    assert source["unique_source_trajectory_count"] <= 2
    assert source["source_data_sha256"] == file_hash(data_path)


def test_real_replay_is_deterministic(tmp_path):
    stage = tmp_path / "stage_0_hammer-v3"
    stage.mkdir()
    np.save(stage / "data_task_0.npy", physical_group(4), allow_pickle=False)
    (stage / "report.json").write_text(
        json.dumps({"splits": {"hammer-v3": {"train_ids": [0, 1, 2], "validation_ids": [3]}}}),
        encoding="utf-8",
    )
    request = RealReplayRequest(1, ("hammer-v3",), {0: 10}, tmp_path, 9)
    first = load_previous_task_real_replay(request)
    second = load_previous_task_real_replay(request)
    assert torch.equal(first[0][0], second[0][0])
    assert first[1]["hammer-v3"]["source_trajectory_ids"] == second[1]["hammer-v3"]["source_trajectory_ids"]


def replay_trainer(mode, tmp_path):
    tasks = [
        "hammer-v3",
        "button-press-v3",
        "faucet-close-v3",
        "handle-press-side-v3",
        "window-close-v3",
    ]
    config = default_config("diffcrl")
    config = replace(
        config,
        continual=replace(
            config.continual,
            tasks=tasks,
            replay_mode=mode,
            replay_allocation="fixed-total" if mode != "none" else "generator-default",
            replay_budget=50 if mode != "none" else 200,
        ),
    )
    trainer = object.__new__(DiffCRLTrainer)
    trainer.config = config
    trainer.sequence = TaskSequence.from_names(tasks)
    trainer.output = tmp_path
    trainer.horizon = 200
    trainer.progress = False
    trainer.diffusion = object() if mode == "diffusion" else None
    trainer.flow_matching = None
    trainer.previous_diffusion = Path("previous.pt")
    trainer.previous_flow_matching = None
    trainer._progress_message = lambda *args, **kwargs: None
    return trainer


def context(trainer, stage):
    return StageContext(
        index=stage,
        task=trainer.sequence.task(stage),
        trajectory_count=50,
        directory=trainer.output / f"stage_{stage}",
        policy_before_hash=None,
        diffusion_before_hash="diffusion-state" if trainer.diffusion else None,
    )


def test_real_and_ddpm_stage_allocations_are_identical(tmp_path):
    expected = {
        1: {0: 50},
        2: {0: 25, 1: 25},
        3: {0: 17, 1: 17, 2: 16},
        4: {0: 13, 1: 13, 2: 12, 3: 12},
    }
    real = replay_trainer("real", tmp_path)
    ddpm = replay_trainer("diffusion", tmp_path)

    for stage, allocation in expected.items():
        with patch(
            "src.continual.diffcrl.load_previous_task_real_replay",
            return_value=({}, {}),
        ) as load_real:
            real._generate_replay(context(real, stage))
        assert load_real.call_args.args[0].trajectory_counts == allocation

        with patch(
            "src.continual.diffcrl.generate_previous_task_replay",
            return_value=ReplayResult({}, {}),
        ) as generate_ddpm:
            ddpm._generate_replay(context(ddpm, stage))
        assert generate_ddpm.call_args.args[0].trajectory_counts == allocation


def test_no_replay_stage_one_bypasses_replay_allocator(tmp_path):
    trainer = replay_trainer("none", tmp_path)
    with patch("src.continual.diffcrl.replay_counts") as allocate:
        result = trainer._generate_replay(context(trainer, 1))
    assert result.groups == {}
    assert result.sources == {}
    allocate.assert_not_called()
