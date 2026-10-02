"""Reusable deterministic continual stage and cross-task evaluation.

R[k,j] is success after training stage k on evaluation task j. Unmeasured cells
are JSON null. Cross-task expert rows are NOT training stages or CL results.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.continual.collector import collect_trajectory
from src.continual.metrics import average_performance, forgetting, forward_transfer
from src.continual.reporting import format_evaluation_result
from src.continual.task_bank import bank_hashes, mt1_tasks
from src.continual.task_sequence import PILOT_SEQUENCE, TaskSequence, TaskSwitcher


class EvaluationMatrix:
    """Store full stage/task results and expose their success-rate projection.

    Missing cells become NaN in ``R``; stage provenance identifies saved artifacts.
    """

    def __init__(self, sequence=PILOT_SEQUENCE):
        self.sequence = sequence
        self.cells = [[None for _ in sequence.tasks] for _ in sequence.tasks]
        self.stage_provenance = {}

    @property
    def R(self):
        return np.array(
            [
                [np.nan if c is None else c["success_rate"] for c in row]
                for row in self.cells
            ]
        )

    def record(self, stage, task, result):
        self.sequence.task(stage)
        self.sequence.task(task)
        if self.cells[stage][task] is not None:
            raise ValueError("Evaluation cell already measured.")
        for field in (
            "mean_return",
            "std_return",
            "success_rate",
            "mean_episode_length",
        ):
            if not np.isfinite(result[field]):
                raise ValueError("Nonfinite evaluation result.")
        if not 0 <= result["success_rate"] <= 1:
            raise ValueError("Invalid success rate.")
        self.cells[stage][task] = result

    def to_dict(self):
        return {
            "kind": "continual_stage_evaluation",
            "tasks": [asdict(t) for t in self.sequence.tasks],
            "R": [
                [None if c is None else c["success_rate"] for c in row]
                for row in self.cells
            ],
            "cells": self.cells,
            "stage_provenance": self.stage_provenance,
            "metrics_by_stage": {
                str(k): {
                    "average_performance": average_performance(self.R, k),
                    "forgetting": forgetting(self.R, k),
                    "forward_transfer": forward_transfer(),
                }
                for k in range(self.sequence.num_tasks)
            },
        }

    @classmethod
    def load(cls, path):
        from src.continual.task_sequence import TaskSpec

        data = json.loads(Path(path).read_text())
        if data["kind"] != "continual_stage_evaluation":
            raise ValueError("Not a continual-stage matrix.")
        result = cls(TaskSequence(tuple(TaskSpec(**t) for t in data["tasks"])))
        if len(data["cells"]) != result.sequence.num_tasks:
            raise ValueError("Matrix size mismatch.")
        for k, row in enumerate(data["cells"]):
            if len(row) != result.sequence.num_tasks:
                raise ValueError("Matrix size mismatch.")
            for j, cell in enumerate(row):
                if cell is not None:
                    result.record(k, j, cell)
        result.stage_provenance = data.get("stage_provenance", {})
        return result


def fixed_mt1_tasks(task_name, task_set_seed):
    """Return the ordered, seeded MT1 configuration bank and stable identities."""
    tasks = mt1_tasks(task_name, task_set_seed)
    hashes = bank_hashes(tasks)
    identities = [
        {
            "task_index": i,
            "task_env_name": item.env_name,
            "task_data_sha256": digest,
        }
        for i, (item, digest) in enumerate(zip(tasks, hashes))
    ]
    if any(item["task_env_name"] != task_name for item in identities):
        raise ValueError("MT1 task bank contains a different environment.")
    if len({item["task_data_sha256"] for item in identities}) != len(tasks):
        raise ValueError("MT1 task bank contains duplicate configurations.")
    return tasks, identities


def disable_task_sampling(env):
    """Stop MetaWorld's wrapper from replacing an explicitly selected task on reset."""
    current = env
    while current is not None:
        toggle = getattr(type(current), "toggle_sample_tasks_on_reset", None)
        if toggle is not None:
            toggle(current, False)
            return
        current = getattr(current, "env", None)
    raise ValueError("Expected MetaWorld task-selection wrapper.")


def running_evaluation_statistics(returns, successes):
    """Return display-only running means from completed evaluation episodes."""
    if not returns or len(returns) != len(successes):
        raise ValueError("Running evaluation values must be nonempty and aligned.")
    return float(np.mean(successes)), float(np.mean(returns))


def evaluate_task(
    policy,
    task,
    *,
    episodes=None,
    seed=10000,
    evaluation_mode="sampled",
    task_set_seed=10000,
    reward_function_version="v2",
    backend="metaworld-v3",
    progress=False,
):
    """Fresh task env, reset(seed+i) each episode, deterministic act; any-step success."""
    if evaluation_mode not in ("sampled", "fixed-tasks"):
        raise ValueError("Unknown evaluation mode.")
    if backend != "metaworld-v3" and evaluation_mode == "fixed-tasks":
        raise ValueError("fixed-tasks evaluation is specific to MetaWorld-v3.")
    selected, identities = (
        fixed_mt1_tasks(task.task_name, task_set_seed)
        if evaluation_mode == "fixed-tasks"
        else (None, None)
    )
    if episodes is None:
        episodes = len(selected) if selected is not None else 100
    if episodes < 1 or selected is not None and episodes > len(selected):
        raise ValueError(
            "Episode count must be positive and cannot exceed the fixed task bank."
        )
    sequence = TaskSequence((task,))
    returns = []
    lengths = []
    successes = []
    episode_provenance = []
    with TaskSwitcher(
        sequence,
        backend=backend,
        reward_function_version=reward_function_version,
    ) as switcher:
        env = switcher.switch(0, seed=seed)
        model = getattr(policy, "model", None)
        if model is not None and (
            model.observation_space != env.observation_space
            or model.action_space != env.action_space
        ):
            raise ValueError("SAC model/environment spaces differ.")
        horizon = env.spec.max_episode_steps
        if selected is not None:
            disable_task_sampling(env)
        with tqdm(
            range(episodes),
            desc=f"Evaluation: {task.task_name}",
            unit="episode",
            disable=not progress,
        ) as evaluation_progress:
            for i in evaluation_progress:
                if selected is not None:
                    env.unwrapped.set_task(selected[i])
                trajectory = collect_trajectory(
                    env, policy, task.task_id, seed=seed + i, deterministic=True
                )
                returns.append(trajectory.episode_return)
                lengths.append(trajectory.length)
                successes.append(trajectory.success)
                running_success, running_return = running_evaluation_statistics(
                    returns, successes
                )
                evaluation_progress.set_postfix(
                    {
                        "Success Rate": f"{100 * running_success:.1f}%",
                        "Mean Return": f"{running_return:.3f}",
                    }
                )
                if selected is not None:
                    episode_provenance.append(
                        {
                            **identities[i],
                            "reset_seed": seed + i,
                            "episode_return": trajectory.episode_return,
                            "success": bool(trajectory.success),
                        }
                    )
    result = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "success_count": sum(successes),
        "mean_episode_length": float(np.mean(lengths)),
        "episodes": episodes,
        "horizon": horizon,
        "seeds": list(range(seed, seed + episodes)),
        "episode_returns": returns,
        "episode_lengths": lengths,
        "episode_successes": successes,
        "deterministic": True,
        "evaluation_mode": evaluation_mode,
        "environment_backend": backend,
        "task_set_seed": task_set_seed if selected is not None else None,
        "task_bank_size": len(selected) if selected is not None else None,
        "episode_provenance": episode_provenance if selected is not None else None,
    }
    if progress:
        tqdm.write(format_evaluation_result(task.task_name, result))
    return result


def evaluate_stage(
    policy,
    matrix,
    stage,
    *,
    include_future=False,
    episodes=None,
    seed=10000,
    evaluation_mode="sampled",
    task_set_seed=10000,
    reward_function_version="v2",
    backend="metaworld-v3",
    progress=False,
):
    """Evaluate row ``stage`` on learned tasks, or all tasks when requested.

    Results are recorded only after all cells complete under one consistent protocol.
    """
    matrix.sequence.task(stage)
    indices = range(matrix.sequence.num_tasks if include_future else stage + 1)
    if any(matrix.cells[stage][j] is not None for j in indices):
        raise ValueError("Stage cells already populated.")
    for row in matrix.cells:
        for cell in row:
            if cell is None:
                continue
            previous_mode = cell.get("evaluation_mode", "sampled")
            if previous_mode != evaluation_mode:
                raise ValueError("Cannot mix evaluation modes in one R matrix.")
            if evaluation_mode == "fixed-tasks" and (
                cell.get("task_set_seed") != task_set_seed
                or cell.get("seeds", [None])[0] != seed
            ):
                raise ValueError(
                    "Fixed task bank or reset seeds differ from existing R cells."
                )
    results = {
        j: evaluate_task(
            policy,
            matrix.sequence.task(j),
            episodes=episodes,
            seed=seed,
            evaluation_mode=evaluation_mode,
            task_set_seed=task_set_seed,
            reward_function_version=reward_function_version,
            backend=backend,
            progress=progress,
        )
        for j in indices
    }
    for j, result in results.items():
        matrix.record(stage, j, result)
    return matrix
