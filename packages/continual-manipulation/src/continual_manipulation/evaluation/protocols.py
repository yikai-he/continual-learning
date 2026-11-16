"""Reusable deterministic continual stage and cross-task evaluation.

R[k,j] is success after training stage k on evaluation task j. Unmeasured cells
are JSON null. Cross-task expert rows are NOT training stages or CL results.
"""

import numpy as np
from .rollout import collect_trajectory
from ..task_banks import (
    backend_tasks,
    task_rand_vec,
    task_identities,
)
from ..sequences import TaskSequence
from ..switching import TaskSwitcher
from ..environments import task_configuration
from .matrix import EvaluationMatrix
from tqdm.auto import tqdm


def fixed_mt1_tasks(task_name, task_set_seed, backend="metaworld-v3"):
    """Return an ordered backend task bank and its stable identities."""
    tasks = backend_tasks(task_name, task_set_seed, backend=backend)
    identities = task_identities(tasks, task_name)
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


def validate_active_task(env, task, backend):
    """Prove reset preserved the explicitly selected serialized configuration."""
    actual = task_configuration(env, backend)
    expected = task_rand_vec(task)
    if not np.array_equal(actual, expected):
        raise ValueError("Fixed evaluation reset changed the selected task identity.")


def running_evaluation_statistics(returns, successes):
    """Return display-only running means from completed evaluation episodes."""
    if not returns or len(returns) != len(successes):
        raise ValueError("Running evaluation values must be nonempty and aligned.")
    return float(np.mean(successes)), float(np.mean(returns))


def format_evaluation_result(task_name: str, result: dict) -> str:
    """Format one completed evaluation result without recalculating it."""
    return (
        f"Result: {task_name} | "
        f"Success Rate: {100 * result['success_rate']:.1f}% | "
        f"Mean Return: {result['mean_return']:.3f}"
    )


def validate_policy_spaces(model, env):
    """Validate inference contracts without trusting stale observation bounds."""
    if (
        model.observation_space.shape != env.observation_space.shape
        or model.observation_space.dtype != env.observation_space.dtype
        or model.action_space.shape != env.action_space.shape
        or model.action_space.dtype != env.action_space.dtype
        or not np.array_equal(model.action_space.low, env.action_space.low)
        or not np.array_equal(model.action_space.high, env.action_space.high)
    ):
        raise ValueError("SAC model/environment spaces differ.")


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
    render_mode=None,
    policy_task_id=None,
):
    """Fresh task env, reset(seed+i) each episode, deterministic act; any-step success."""
    if evaluation_mode not in ("sampled", "fixed-tasks"):
        raise ValueError("Unknown evaluation mode.")
    if backend not in ("metaworld-v3", "kuka-v3") and evaluation_mode == "fixed-tasks":
        raise ValueError("fixed-tasks evaluation requires a deterministic task bank.")
    selected, identities = (
        fixed_mt1_tasks(task.task_name, task_set_seed, backend=backend)
        if evaluation_mode == "fixed-tasks"
        else (None, None)
    )
    if episodes is None:
        episodes = len(selected) if selected is not None else 100
    if episodes < 1:
        raise ValueError("Episode count must be positive.")
    sequence = TaskSequence((task,))
    returns = []
    lengths = []
    successes = []
    episode_provenance = []
    with TaskSwitcher(
        sequence,
        backend=backend,
        reward_function_version=reward_function_version,
        render_mode=render_mode,
    ) as switcher:
        env = switcher.switch(0, seed=seed)
        model = getattr(policy, "model", None)
        if model is not None:
            validate_policy_spaces(model, env)
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
                    task_index = i % len(selected)
                    env.unwrapped.set_task(selected[task_index])
                rollout_policy = policy
                if getattr(policy, "task_conditioning", "none") == "one_hot":
                    if policy_task_id is None:
                        raise ValueError("Task-conditioned evaluation requires a task ID.")

                    class _FixedTaskPolicy:
                        def act(self, observation, deterministic=True):
                            return policy.act(
                                observation,
                                deterministic=deterministic,
                                task_id=policy_task_id,
                            )

                    rollout_policy = _FixedTaskPolicy()
                trajectory = collect_trajectory(
                    env, rollout_policy, task.task_id, seed=seed + i, deterministic=True
                )
                if selected is not None:
                    validate_active_task(env, selected[task_index], backend)
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
                            **identities[task_index],
                            "task_bank_index": task_index,
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
        "task_bank_repetitions": (
            int(np.ceil(episodes / len(selected))) if selected is not None else None
        ),
        "ordered_task_hashes": (
            [identity["task_hash"] for identity in identities]
            if selected is not None
            else None
        ),
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
            policy_task_id=j,
        )
        for j in indices
    }
    for j, result in results.items():
        matrix.record(stage, j, result)
    return matrix
