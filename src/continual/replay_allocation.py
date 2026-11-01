"""Deterministic task-level allocation of a fixed historical replay budget."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping


def allocate_replay(
    task_ids: Iterable[Hashable],
    total_budget: int,
    strategy: str = "uniform",
    *,
    priority_scores: Mapping[Hashable, float] | None = None,
) -> dict[Hashable, int]:
    """Allocate exactly ``total_budget`` integer samples across ``task_ids``.

    Uniform allocation gives the remainder to the earliest task IDs in the
    supplied order.  Consequently, when the budget is smaller than the number
    of tasks, later tasks deterministically receive zero samples.
    """
    tasks = tuple(task_ids)
    if len(set(tasks)) != len(tasks):
        raise ValueError("task_ids must be unique.")
    if isinstance(total_budget, bool) or not isinstance(total_budget, int):
        raise TypeError("total_budget must be an integer.")
    if total_budget < 0:
        raise ValueError("total_budget must be nonnegative.")
    if strategy != "uniform":
        raise ValueError(f"Unsupported replay allocation strategy: {strategy!r}.")
    if priority_scores is not None:
        raise ValueError("priority_scores are not used by the uniform strategy.")
    if not tasks:
        if total_budget:
            raise ValueError("A positive replay budget requires historical tasks.")
        return {}

    quotient, remainder = divmod(total_budget, len(tasks))
    return {
        task_id: quotient + (index < remainder) for index, task_id in enumerate(tasks)
    }


def replay_counts(
    task_ids: Iterable[Hashable],
    *,
    generator: str,
    allocation: str,
    trajectories_per_task: int,
    total_budget: int,
    strategy: str = "uniform",
) -> dict[Hashable, int]:
    """Resolve legacy or fixed-total replay counts independently of sampling.

    ``generator-default`` is the compatibility policy: historical DDPM runs
    retain one ``trajectories_per_task`` count per old task, while historical
    Flow Matching runs retain their fixed-total budget. ``fixed-total`` gives
    either generator the exact same allocation.
    """
    tasks = tuple(task_ids)
    if generator not in ("diffusion", "flow-matching"):
        raise ValueError(f"Unsupported replay generator: {generator!r}.")
    if allocation not in ("generator-default", "fixed-total"):
        raise ValueError(f"Unsupported replay allocation: {allocation!r}.")
    if allocation == "fixed-total" or generator == "flow-matching":
        return allocate_replay(tasks, total_budget, strategy)
    if strategy != "uniform":
        raise ValueError(f"Unsupported replay allocation strategy: {strategy!r}.")
    if (
        isinstance(trajectories_per_task, bool)
        or not isinstance(trajectories_per_task, int)
        or trajectories_per_task < 0
    ):
        raise ValueError("trajectories_per_task must be a nonnegative integer.")
    if len(set(tasks)) != len(tasks):
        raise ValueError("task_ids must be unique.")
    return {task_id: trajectories_per_task for task_id in tasks}
