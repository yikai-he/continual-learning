"""Passive MT1 collection identity and deterministic training-bank goal replay."""

import hashlib
import pickle
from importlib.metadata import version

import numpy as np

from src.envs import (
    GOAL_SLICE,
    make_env,
    make_metaworld_env,
    target_position,
    task_configuration,
)
from src.env_adapters.kuka_v3 import kuka_v3_tasks


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


def _task_wrapper(env):
    """Locate the MetaWorld task-selection wrapper in an adapter stack."""
    cursor = env
    while cursor is not None:
        if hasattr(cursor, "tasks") and hasattr(
            cursor, "toggle_sample_tasks_on_reset"
        ):
            return cursor
        cursor = getattr(cursor, "env", None)
    raise ValueError("MetaWorld task-selection wrapper is missing.")


def _task_rand_vec(task):
    data = pickle.loads(task.data)
    if "rand_vec" not in data:
        raise ValueError("Serialized MetaWorld task has no rand_vec.")
    return np.asarray(data["rand_vec"], dtype=np.float64)


def verify_disjoint_task_banks(
    task_name,
    training_seed,
    evaluation_seed,
    *,
    task_provider=mt1_tasks,
    backend="metaworld-v3",
):
    """Build two MT1 banks and prove their serialized task identities are disjoint."""
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


def training_bank(task_name, seed, *, backend="metaworld-v3"):
    """Record identities needed to verify and reconstruct replay goals."""
    if backend == "kuka-v2":
        return {
            "backend": backend,
            "task_name": task_name,
            "bank_seed": seed,
            "ordered_task_hashes": [],
            "reset_seeds": [],
            "goals": [],
        }
    if backend == "kuka-v3":
        tasks = backend_tasks(task_name, seed, backend=backend)
        return {
            "backend": backend,
            "task_name": task_name,
            "bank_seed": seed,
            "ordered_task_hashes": bank_hashes(tasks),
            "metaworld_version": version("metaworld"),
            "reset_seeds": [],
            "goals": [],
            "configuration_records": {},
        }
    if backend != "metaworld-v3":
        raise ValueError(f"Unsupported environment backend: {backend!r}.")
    tasks = backend_tasks(task_name, seed, backend=backend)
    return {
        "backend": backend,
        "task_name": task_name,
        "bank_seed": seed,
        "ordered_task_hashes": bank_hashes(tasks),
        "metaworld_version": version("metaworld"),
    }


def selected_configuration(env, bank, observation, episode_id):
    """Read the already selected frozen rand_vec; never draw from wrapper RNG."""
    if bank.get("backend", "metaworld-v3") == "kuka-v2":
        goal = np.asarray(observation)[GOAL_SLICE].astype(np.float32, copy=True)
        backend = bank["backend"]
        target = target_position(env, backend)
        if not np.allclose(goal, target, rtol=1e-6, atol=1e-7):
            raise ValueError("Collection goal disagrees with environment target.")
        reset_seed = bank["bank_seed"] + episode_id
        configuration = task_configuration(env, backend)
        identity = goal if configuration is None else configuration
        digest = hashlib.sha256(identity.tobytes()).hexdigest()
        bank["ordered_task_hashes"].append(digest)
        bank["reset_seeds"].append(reset_seed)
        bank["goals"].append(goal.tolist())
        return {
            "configuration_index": episode_id,
            "task_data_sha256": digest,
            "task_name": bank["task_name"],
            "env_name": bank["task_name"],
            "bank_seed": bank["bank_seed"],
            "reset_seed": reset_seed,
            "episode_id": episode_id,
        }
    if bank.get("backend") == "kuka-v3":
        wrapper = _task_wrapper(env)
        hashes = bank_hashes(wrapper.tasks)
        if hashes != bank["ordered_task_hashes"]:
            raise ValueError("Collection wrapper task bank differs from recorded KUKA bank.")
        configuration = task_configuration(env, "kuka-v3")
        matches = [
            index
            for index, task in enumerate(wrapper.tasks)
            if np.array_equal(_task_rand_vec(task), configuration)
        ]
        if len(matches) != 1:
            raise ValueError("Selected KUKA configuration is not uniquely identifiable.")
        index = matches[0]
        goal = np.asarray(observation)[GOAL_SLICE].astype(np.float32, copy=True)
        target = target_position(env, "kuka-v3")
        if not np.allclose(goal, target, rtol=1e-6, atol=1e-7):
            raise ValueError("Collection goal disagrees with environment target.")
        reset_seed = bank["bank_seed"] + episode_id
        bank["reset_seeds"].append(reset_seed)
        bank["goals"].append(goal.tolist())
        record = {
            "task_bank_index": index,
            "task_hash": hashes[index],
            "recorded_goal": goal.tolist(),
            "last_rand_vec": configuration.tolist(),
            "reset_seed": reset_seed,
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
            "reset_seed": reset_seed,
            "recorded_goal": goal.tolist(),
            "last_rand_vec": configuration.tolist(),
            "episode_id": episode_id,
        }
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
    hammer_reward_variant="original",
    hammer_nail_progress_weight=0.0,
):
    """Only task identities are read; no expert trajectory archive is opened."""
    if backend == "kuka-v2":
        if bank.get("backend") != backend:
            raise ValueError("Replay bank backend differs from requested backend.")
        env = make_env(
            backend,
            bank["task_name"],
            bank["bank_seed"],
            reward_function_version=reward_function_version,
            hammer_reward_variant=hammer_reward_variant,
            hammer_nail_progress_weight=hammer_nail_progress_weight,
        )
        try:
            goals = []
            for index in indices:
                if index < 0 or index >= len(bank["reset_seeds"]):
                    raise ValueError("Invalid replay configuration index.")
                observation, _ = env.reset(seed=bank["reset_seeds"][index])
                goal = np.asarray(observation[GOAL_SLICE], dtype=np.float32).copy()
                target = target_position(env, backend)
                if not np.allclose(goal, target, rtol=1e-6, atol=1e-7):
                    raise ValueError("Recreated KUKA goal disagrees with target.")
                configuration = task_configuration(env, backend)
                identity = goal if configuration is None else configuration
                if hashlib.sha256(identity.tobytes()).hexdigest() != bank[
                    "ordered_task_hashes"
                ][index]:
                    raise ValueError("Recreated KUKA goal differs from training bank.")
                goals.append(goal)
            return np.stack(goals)
        finally:
            env.close()
    if backend == "kuka-v3":
        if bank.get("backend") != backend:
            raise ValueError("Replay bank backend differs from requested backend.")
        required = ("ordered_task_hashes", "metaworld_version")
        if any(key not in bank for key in required):
            raise ValueError(
                "KUKA replay bank lacks exact task identity; recollect Stage 1."
            )
        if bank["metaworld_version"] != version("metaworld"):
            raise ValueError("MetaWorld version differs from KUKA training bank provenance.")
        tasks = kuka_v3_tasks(bank["task_name"], bank["bank_seed"])
        hashes = bank_hashes(tasks)
        if hashes != bank["ordered_task_hashes"]:
            raise ValueError("Recreated KUKA task-bank hash order differs.")
        env = make_env(
            backend,
            bank["task_name"],
            bank["bank_seed"],
            reward_function_version=reward_function_version,
            hammer_reward_variant=hammer_reward_variant,
            hammer_nail_progress_weight=hammer_nail_progress_weight,
        )
        try:
            wrapper = _task_wrapper(env)
            if bank_hashes(wrapper.tasks) != hashes:
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
                    raise ValueError("Recreated KUKA rand_vec differs from selected task.")
                actual_hash = hashlib.sha256(tasks[index].data).hexdigest()
                if actual_hash != bank["ordered_task_hashes"][index]:
                    raise ValueError("Recreated KUKA task identity differs from training bank.")
                record = bank.get("configuration_records", {}).get(str(index))
                if record is not None:
                    recorded_goal = np.asarray(record["recorded_goal"], dtype=np.float32)
                    if not np.allclose(
                        goal, recorded_goal, rtol=1e-6, atol=1e-7
                    ):
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
