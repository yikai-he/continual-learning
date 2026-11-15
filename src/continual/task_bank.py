"""Passive MT1 collection identity and deterministic training-bank goal replay."""

import pickle
from importlib.metadata import version

import numpy as np
from continual_manipulation.environments import (
    GOAL_SLICE,
    make_env,
    make_metaworld_env,
    target_position,
    task_configuration,
)
from continual_manipulation.task_banks import (
    backend_tasks as _backend_tasks,
)
from continual_manipulation.task_banks import (
    bank_hashes as _bank_hashes,
)
from continual_manipulation.task_banks import (
    task_rand_vec as _task_rand_vec,
)
from continual_manipulation.task_banks import (
    task_wrapper as _task_wrapper,
)


def training_bank(task_name, seed, *, backend="metaworld-v3"):
    """Record identities needed to verify and reconstruct replay goals."""
    if backend not in ("metaworld-v3", "kuka-v3"):
        raise ValueError(f"Unsupported environment backend: {backend!r}.")
    tasks = _backend_tasks(task_name, seed, backend=backend)
    bank = {
        "backend": backend,
        "task_name": task_name,
        "bank_seed": seed,
        "ordered_task_hashes": _bank_hashes(tasks),
        "metaworld_version": version("metaworld"),
    }
    if backend == "kuka-v3":
        bank.update(reset_seeds=[], goals=[], configuration_records={})
    return bank


def selected_configuration(env, bank, observation, episode_id, *, reset_seed=None):
    """Read the already selected frozen rand_vec; never draw from wrapper RNG."""
    if bank.get("backend") == "kuka-v3":
        wrapper = _task_wrapper(env)
        hashes = _bank_hashes(wrapper.tasks)
        if hashes != bank["ordered_task_hashes"]:
            raise ValueError(
                "Collection wrapper task bank differs from recorded KUKA bank."
            )
        configuration = task_configuration(env, "kuka-v3")
        matches = [
            index
            for index, task in enumerate(wrapper.tasks)
            if np.array_equal(_task_rand_vec(task), configuration)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Selected KUKA configuration is not uniquely identifiable."
            )
        index = matches[0]
        goal = np.asarray(observation)[GOAL_SLICE].astype(np.float32, copy=True)
        target = target_position(env, "kuka-v3")
        if not np.allclose(goal, target, rtol=1e-6, atol=1e-7):
            raise ValueError("Collection goal disagrees with environment target.")
        actual_reset_seed = (
            bank["bank_seed"] + episode_id if reset_seed is None else reset_seed
        )
        bank["reset_seeds"].append(actual_reset_seed)
        bank["goals"].append(goal.tolist())
        record = {
            "task_bank_index": index,
            "task_hash": hashes[index],
            "recorded_goal": goal.tolist(),
            "last_rand_vec": configuration.tolist(),
            "reset_seed": actual_reset_seed,
        }
        prior = bank["configuration_records"].setdefault(str(index), record)
        if prior["task_hash"] != record["task_hash"] or not np.array_equal(
            prior["last_rand_vec"], record["last_rand_vec"]
        ):
            raise ValueError("KUKA task-bank identity changed during collection.")
        return {
            "configuration_index": index,
            "task_bank_index": index,
            "task_data_sha256": hashes[index],
            "task_hash": hashes[index],
            "task_name": bank["task_name"],
            "env_name": wrapper.tasks[index].env_name,
            "bank_seed": bank["bank_seed"],
            "reset_seed": actual_reset_seed,
            "recorded_goal": goal.tolist(),
            "last_rand_vec": configuration.tolist(),
            "episode_id": episode_id,
        }
    cursor = env
    while cursor is not None and not hasattr(cursor, "tasks"):
        cursor = getattr(cursor, "env", None)
    if cursor is None or _bank_hashes(cursor.tasks) != bank["ordered_task_hashes"]:
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
    if not np.allclose(
        np.asarray(observation)[GOAL_SLICE],
        target_position(env, "metaworld-v3"),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("Collection goal disagrees with environment target.")
    return {
        "configuration_index": index,
        "task_data_sha256": bank["ordered_task_hashes"][index],
        "task_name": bank["task_name"],
        "env_name": cursor.tasks[index].env_name,
        "bank_seed": bank["bank_seed"],
        "reset_seed": reset_seed,
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


def reconstruct_goals(
    bank,
    indices,
    *,
    backend="metaworld-v3",
    reward_function_version="v2",
):
    """Only task identities are read; no expert trajectory archive is opened."""
    if backend == "kuka-v3":
        if bank.get("backend") != backend:
            raise ValueError("Replay bank backend differs from requested backend.")
        required = ("ordered_task_hashes", "metaworld_version")
        if any(key not in bank for key in required):
            raise ValueError(
                "KUKA replay bank lacks exact task identity; recollect Stage 1."
            )
        if bank["metaworld_version"] != version("metaworld"):
            raise ValueError(
                "MetaWorld version differs from KUKA training bank provenance."
            )
        tasks = _backend_tasks(bank["task_name"], bank["bank_seed"], backend=backend)
        hashes = _bank_hashes(tasks)
        if hashes != bank["ordered_task_hashes"]:
            raise ValueError("Recreated KUKA task-bank hash order differs.")
        env = make_env(
            backend,
            bank["task_name"],
            bank["bank_seed"],
            reward_function_version=reward_function_version,
        )
        try:
            wrapper = _task_wrapper(env)
            if _bank_hashes(wrapper.tasks) != hashes:
                raise ValueError("Recreated KUKA wrapper task bank differs.")
            wrapper.toggle_sample_tasks_on_reset(False)
            goals = []
            for index in indices:
                if index < 0 or index >= len(tasks):
                    raise ValueError("Invalid replay configuration index.")
                env.unwrapped.set_task(tasks[index])
                observation, _ = env.reset(seed=bank["bank_seed"])
                goal = np.asarray(observation[GOAL_SLICE], dtype=np.float32).copy()
                target = target_position(env, backend)
                if not np.allclose(goal, target, rtol=1e-6, atol=1e-7):
                    raise ValueError("Recreated KUKA goal disagrees with target.")
                actual_configuration = task_configuration(env, backend)
                expected_configuration = _task_rand_vec(tasks[index])
                if not np.array_equal(actual_configuration, expected_configuration):
                    raise ValueError(
                        "Recreated KUKA rand_vec differs from selected task."
                    )
                actual_hash = hashes[index]
                if actual_hash != bank["ordered_task_hashes"][index]:
                    raise ValueError(
                        "Recreated KUKA task identity differs from training bank."
                    )
                record = bank.get("configuration_records", {}).get(str(index))
                if record is not None:
                    recorded_goal = np.asarray(
                        record["recorded_goal"], dtype=np.float32
                    )
                    if not np.allclose(goal, recorded_goal, rtol=1e-6, atol=1e-7):
                        raise ValueError(
                            "Recreated KUKA goal differs from training bank: "
                            f"recorded={recorded_goal.tolist()}, "
                            f"reconstructed={goal.tolist()}."
                        )
                goals.append(goal)
            return np.stack(goals)
        finally:
            env.close()
    if backend != "metaworld-v3" or bank.get("backend", backend) != backend:
        raise ValueError("Replay bank backend differs from requested backend.")
    if bank["metaworld_version"] != version("metaworld"):
        raise ValueError("MetaWorld version differs from training bank provenance.")
    tasks = _backend_tasks(bank["task_name"], bank["bank_seed"], backend=backend)
    if _bank_hashes(tasks) != bank["ordered_task_hashes"]:
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
            if not np.allclose(
                goal,
                target_position(env, "metaworld-v3"),
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError("Replayed goal disagrees with environment target.")
            goals.append(goal)
        return np.stack(goals)
    finally:
        env.close()
