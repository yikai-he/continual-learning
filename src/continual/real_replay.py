"""Stored-real historical replay for the Real Replay comparison arm."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from src.support.reproducibility import file_hash

from .diffusion_data import validate_stage_trajectory_group


@dataclass(frozen=True)
class RealReplayRequest:
    stage: int
    task_names: tuple[str, ...]
    trajectory_counts: dict[int, int]
    run_directory: Path
    runtime_seed: int


def load_previous_task_real_replay(request: RealReplayRequest):
    """Sample only original per-task training episodes, with replacement.

    The source trajectory and its original split live in the task's first stage.
    Validation IDs are never eligible. Returned tensors use the same physical
    ``(N,T,43)`` contract as generated replay.
    """
    expected = set(range(request.stage))
    if set(request.trajectory_counts) != expected:
        raise ValueError("Real replay allocation does not match historical tasks.")
    groups, sources = {}, {}
    for task_index in range(request.stage):
        count = request.trajectory_counts[task_index]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid real replay trajectory count.")
        if count == 0:
            continue
        task_name = request.task_names[task_index]
        stage_directory = request.run_directory / f"stage_{task_index}_{task_name}"
        data_path = stage_directory / f"data_task_{task_index}.npy"
        report_path = stage_directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        train_ids = report["splits"][task_name]["train_ids"]
        if not train_ids:
            raise ValueError(f"No original training trajectories for {task_name}.")
        seed = request.runtime_seed + 100000 + request.stage * 1000 + task_index
        selected = (
            np.random.default_rng(seed)
            .choice(np.asarray(train_ids, dtype=np.int64), size=count, replace=True)
            .tolist()
        )
        archived = np.load(data_path, allow_pickle=False)
        if any(index < 0 or index >= len(archived) for index in selected):
            raise ValueError(
                "Saved real replay split contains an invalid trajectory ID."
            )
        group = torch.from_numpy(archived[selected]).float()
        multiplicity = Counter(selected)
        source = {
            "kind": "real_replay",
            "source_data_path": str(data_path),
            "source_data_sha256": file_hash(data_path),
            "source_report": str(report_path),
            "source_report_sha256": file_hash(report_path),
            "sampling_seed": seed,
            "trajectory_count": count,
            "source_trajectory_ids": selected,
            "unique_source_trajectory_count": len(multiplicity),
            "reuse_multiplicity": {
                str(index): multiplicity[index] for index in sorted(multiplicity)
            },
            "original_train_ids": train_ids,
            "original_validation_ids": report["splits"][task_name]["validation_ids"],
        }
        validate_stage_trajectory_group(
            group, stage=request.stage, task=task_name, source=source
        )
        groups[task_index] = group
        sources[task_name] = source
    return groups, sources
