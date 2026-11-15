"""Fixed-budget historical replay allocation contract."""

import pytest

from src.continual.replay_allocation import (
    allocate_replay,
    allocate_weighted_replay,
    replay_counts,
)


@pytest.mark.parametrize(
    ("tasks", "budget", "expected"),
    [
        ([], 0, {}),
        ([0], 200, {0: 200}),
        ([0, 1], 200, {0: 100, 1: 100}),
        ([0, 1, 2], 200, {0: 67, 1: 67, 2: 66}),
        ([0, 1, 2, 3], 2, {0: 1, 1: 1, 2: 0, 3: 0}),
    ],
)
def test_uniform_allocation_is_exact_and_deterministic(tasks, budget, expected):
    assert allocate_replay(tasks, budget) == expected
    assert allocate_replay(tasks, budget) == expected
    assert sum(expected.values()) == budget


def test_invalid_allocation_requests_are_rejected():
    with pytest.raises(ValueError, match="positive replay budget"):
        allocate_replay([], 1)
    with pytest.raises(ValueError, match="unique"):
        allocate_replay([0, 0], 1)
    with pytest.raises(ValueError, match="Unsupported"):
        allocate_replay([0], 1, "forgetting-aware")
    with pytest.raises(ValueError, match="priority_scores"):
        allocate_replay([0], 1, priority_scores={0: 1.0})


@pytest.mark.parametrize("generator", ["diffusion", "flow-matching"])
def test_fixed_total_allocation_is_generator_independent(generator):
    counts = replay_counts(
        range(3),
        generator=generator,
        allocation="fixed-total",
        trajectories_per_task=50,
        total_budget=50,
    )
    assert counts == {0: 17, 1: 17, 2: 16}
    assert sum(counts.values()) == 50


def test_generator_default_preserves_legacy_allocation():
    common = dict(
        task_ids=range(3),
        allocation="generator-default",
        trajectories_per_task=50,
        total_budget=50,
    )
    assert replay_counts(generator="diffusion", **common) == {0: 50, 1: 50, 2: 50}
    assert replay_counts(generator="flow-matching", **common) == {
        0: 17,
        1: 17,
        2: 16,
    }


@pytest.mark.parametrize(
    ("tasks", "weights", "expected"),
    [
        ([0], [1.0], {0: 100}),
        ([0, 1], [1.0, 2.0], {0: 33, 1: 67}),
        ([0, 1, 2], [1.0, 2.0, 2.0], {0: 20, 1: 40, 2: 40}),
        ([0, 1, 2], [1.0, 1.0, 1.0], {0: 34, 1: 33, 2: 33}),
        ([0, 1, 2], [0.0, 1.0, 0.0], {0: 0, 1: 100, 2: 0}),
    ],
)
def test_weighted_allocation_is_exact_and_deterministic(tasks, weights, expected):
    assert allocate_weighted_replay(tasks, 100, weights) == expected
    assert allocate_weighted_replay(tasks, 100, weights) == expected
    assert sum(expected.values()) == 100


@pytest.mark.parametrize("generator", ["diffusion", "flow-matching"])
def test_fixed_weighted_allocation_is_shared_by_generators(generator):
    assert replay_counts(
        range(2),
        generator=generator,
        allocation="fixed-weighted",
        trajectories_per_task=100,
        total_budget=100,
        task_weights=[1.0, 2.0],
    ) == {0: 33, 1: 67}


@pytest.mark.parametrize(
    ("tasks", "weights", "message"),
    [
        ([0, 1], [1.0], "one weight"),
        ([0], [-1.0], "finite nonnegative"),
        ([0], [float("nan")], "finite nonnegative"),
        ([0, 1], [0.0, 0.0], "positive sum"),
    ],
)
def test_invalid_weighted_allocations_are_rejected(tasks, weights, message):
    with pytest.raises(ValueError, match=message):
        allocate_weighted_replay(tasks, 100, weights)
