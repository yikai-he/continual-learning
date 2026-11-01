"""Fixed-budget historical replay allocation contract."""

import pytest

from src.continual.replay_allocation import allocate_replay, replay_counts


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
