"""Passive MT1 collection identity and deterministic training-bank goal replay."""

import hashlib
import pickle
from importlib.metadata import version

import metaworld
import numpy as np

from src.envs import GOAL_SLICE, make_metaworld_env


def bank_hashes(tasks):
    return [hashlib.sha256(task.data).hexdigest() for task in tasks]


def mt1_tasks(task_name, seed):
    return metaworld.MT1(task_name, seed=seed).train_tasks


def training_bank(task_name, seed):
    """Record ordered MT1 identities needed to verify and reconstruct replay goals."""
    tasks = mt1_tasks(task_name, seed)
    return {
        "task_name": task_name,
        "bank_seed": seed,
        "ordered_task_hashes": bank_hashes(tasks),
        "metaworld_version": version("metaworld"),
    }


def selected_configuration(env, bank, observation, episode_id):
    """Read the already selected frozen rand_vec; never draw from wrapper RNG."""
    cursor = env
    while cursor is not None and not hasattr(cursor, "tasks"):
        cursor = getattr(cursor, "env", None)
    if cursor is None or bank_hashes(cursor.tasks) != bank["ordered_task_hashes"]:
        raise ValueError("Collection wrapper task bank differs from recorded MT1 bank.")
    selected = np.asarray(env.unwrapped._last_rand_vec)
    matches = [
        i
        for i, task in enumerate(cursor.tasks)
        if np.array_equal(np.asarray(pickle.loads(task.data)["rand_vec"]), selected)
    ]
    if len(matches) != 1:
        raise ValueError("Selected MT1 configuration is not uniquely identifiable.")
    index = matches[0]
    if not np.array_equal(
        np.asarray(observation)[GOAL_SLICE],
        np.asarray(env.unwrapped._target_pos, dtype=np.float32),
    ):
        raise ValueError("Collection goal disagrees with environment target.")
    return {
        "configuration_index": index,
        "task_data_sha256": bank["ordered_task_hashes"][index],
        "task_name": bank["task_name"],
        "env_name": cursor.tasks[index].env_name,
        "bank_seed": bank["bank_seed"],
        "episode_id": episode_id,
    }


def sample_training_configurations(bank, train_indices, count, seed):
    """Sample replay conditions with replacement from an old task's training split."""
    if not train_indices or any(
        i < 0 or i >= len(bank["ordered_task_hashes"]) for i in train_indices
    ):
        raise ValueError("Invalid training configuration multiset.")
    return (
        np.random.default_rng(seed)
        .choice(np.asarray(train_indices), size=count, replace=True)
        .tolist()
    )


def reconstruct_goals(bank, indices, *, reward_function_version="v2"):
    """Only task identities are read; no expert trajectory archive is opened."""
    if bank["metaworld_version"] != version("metaworld"):
        raise ValueError("MetaWorld version differs from training bank provenance.")
    tasks = mt1_tasks(bank["task_name"], bank["bank_seed"])
    if bank_hashes(tasks) != bank["ordered_task_hashes"]:
        raise ValueError("Recreated MT1 training bank hash order differs.")
    env = make_metaworld_env(
        bank["task_name"],
        bank["bank_seed"],
        reward_function_version=reward_function_version,
    )
    try:
        cursor = env
        while cursor is not None and not hasattr(cursor, "_set_random_task"):
            cursor = getattr(cursor, "env", None)
        if cursor is None:
            raise ValueError("MT1 random-task wrapper is missing.")
        cursor._set_random_task = lambda: None
        goals = []
        for index in indices:
            if index < 0 or index >= len(tasks):
                raise ValueError("Invalid replay configuration index.")
            env.unwrapped.set_task(tasks[index])
            observation, _ = env.reset()
            goal = np.asarray(observation[GOAL_SLICE], dtype=np.float32).copy()
            if not np.array_equal(
                goal, np.asarray(env.unwrapped._target_pos, dtype=np.float32)
            ):
                raise ValueError("Replayed goal disagrees with environment target.")
            goals.append(goal)
        return np.stack(goals)
    finally:
        env.close()
