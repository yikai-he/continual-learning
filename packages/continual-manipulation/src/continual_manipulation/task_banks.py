"""Deterministic backend task-bank identity and split validation."""

import hashlib
import pickle

import numpy as np

from .environments.kuka_v3 import kuka_v3_tasks


def bank_hashes(tasks):
    return [hashlib.sha256(task.data).hexdigest() for task in tasks]


def mt1_tasks(task_name, seed):
    import metaworld

    return metaworld.MT1(task_name, seed=seed).train_tasks


def backend_tasks(task_name, seed, *, backend="metaworld-v3"):
    """Build the backend's ordered deterministic task bank."""
    if backend == "metaworld-v3":
        return mt1_tasks(task_name, seed)
    if backend == "kuka-v3":
        return kuka_v3_tasks(task_name, seed)
    raise ValueError(f"Backend {backend!r} has no deterministic task bank.")


def task_identities(tasks, task_name):
    """Return stable ordered identities and reject malformed task banks."""
    hashes = bank_hashes(tasks)
    identities = [
        {
            "task_index": index,
            "task_bank_index": index,
            "task_env_name": task.env_name,
            "task_data_sha256": digest,
            "task_hash": digest,
        }
        for index, (task, digest) in enumerate(zip(tasks, hashes))
    ]
    if any(identity["task_env_name"] != task_name for identity in identities):
        raise ValueError("Task bank contains a different environment.")
    if len(set(hashes)) != len(tasks):
        raise ValueError("Task bank contains duplicate configurations.")
    return identities


def task_wrapper(env):
    """Locate the MetaWorld task-selection wrapper in an adapter stack."""
    cursor = env
    while cursor is not None:
        if hasattr(cursor, "tasks") and hasattr(cursor, "toggle_sample_tasks_on_reset"):
            return cursor
        cursor = getattr(cursor, "env", None)
    raise ValueError("MetaWorld task-selection wrapper is missing.")


def task_rand_vec(task):
    data = pickle.loads(task.data)
    if "rand_vec" not in data:
        raise ValueError("Serialized MetaWorld task has no rand_vec.")
    return np.asarray(data["rand_vec"], dtype=np.float64)


def verify_disjoint_task_banks(
    task_name, training_seed, evaluation_seed, *, task_provider=mt1_tasks,
    backend="metaworld-v3",
):
    """Prove training and evaluation task identities are disjoint."""
    if task_provider is mt1_tasks:
        task_provider = lambda name, seed: backend_tasks(name, seed, backend=backend)
    training = task_provider(task_name, training_seed)
    evaluation = task_provider(task_name, evaluation_seed)
    training_hashes = bank_hashes(training)
    evaluation_hashes = bank_hashes(evaluation)
    overlap = sorted(set(training_hashes) & set(evaluation_hashes))
    if overlap:
        raise ValueError(
            f"Training/evaluation task-bank overlap for {task_name}: "
            f"{len(overlap)} overlapping configuration hash(es): {overlap}"
        )
    return {
        "task_name": task_name,
        "backend": backend,
        "training_bank_seed": training_seed,
        "evaluation_bank_seed": evaluation_seed,
        "ordered_training_task_hashes": training_hashes,
        "ordered_evaluation_task_hashes": evaluation_hashes,
        "overlapping_task_hashes": [],
        "disjoint": True,
    }
