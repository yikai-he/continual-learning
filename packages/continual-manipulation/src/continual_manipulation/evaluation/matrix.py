"""Serialized continual stage-by-task evaluation matrix."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..metrics import average_performance, forgetting, forward_transfer
from ..sequences import PILOT_SEQUENCE, TaskSequence, TaskSpec


class EvaluationMatrix:
    """Store full stage/task results and expose their success-rate projection."""

    def __init__(self, sequence=PILOT_SEQUENCE):
        self.sequence = sequence
        self.cells = [[None for _ in sequence.tasks] for _ in sequence.tasks]
        self.stage_provenance = {}

    @property
    def R(self):
        return np.array([
            [np.nan if cell is None else cell["success_rate"] for cell in row]
            for row in self.cells
        ])

    def record(self, stage, task, result):
        self.sequence.task(stage)
        self.sequence.task(task)
        if self.cells[stage][task] is not None:
            raise ValueError("Evaluation cell already measured.")
        for field in ("mean_return", "std_return", "success_rate", "mean_episode_length"):
            if not np.isfinite(result[field]):
                raise ValueError("Nonfinite evaluation result.")
        if not 0 <= result["success_rate"] <= 1:
            raise ValueError("Invalid success rate.")
        self.cells[stage][task] = result

    def to_dict(self):
        return {
            "kind": "continual_stage_evaluation",
            "tasks": [asdict(task) for task in self.sequence.tasks],
            "R": [[None if cell is None else cell["success_rate"] for cell in row]
                  for row in self.cells],
            "cells": self.cells,
            "stage_provenance": self.stage_provenance,
            "metrics_by_stage": {
                str(stage): {
                    "average_performance": average_performance(self.R, stage),
                    "forgetting": forgetting(self.R, stage),
                    "forward_transfer": forward_transfer(),
                }
                for stage in range(self.sequence.num_tasks)
            },
        }

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["kind"] != "continual_stage_evaluation":
            raise ValueError("Not a continual-stage matrix.")
        result = cls(TaskSequence(tuple(TaskSpec(**task) for task in data["tasks"])))
        if len(data["cells"]) != result.sequence.num_tasks:
            raise ValueError("Matrix size mismatch.")
        for stage, row in enumerate(data["cells"]):
            if len(row) != result.sequence.num_tasks:
                raise ValueError("Matrix size mismatch.")
            for task, cell in enumerate(row):
                if cell is not None:
                    result.record(stage, task, cell)
        result.stage_provenance = data.get("stage_provenance", {})
        return result
