"""Offline reward-ranking checks for the post-success retention ablation."""

import numpy as np


def returns(rewards, gamma=0.99):
    rewards = np.asarray(rewards, dtype=float)
    discounts = gamma ** np.arange(rewards.size)
    return float(rewards.sum()), float(discounts @ rewards)


def post_success_rewards(trace):
    return [0.2 if inside else -1.0 for inside in trace]


def test_equal_pre_success_history_ranks_stable_over_transient() -> None:
    common_prefix = [0.4] * 100 + [100.0]
    stable = common_prefix + post_success_rewards([True] * 99)
    transient = common_prefix + post_success_rewards([False] * 99)
    reentry = common_prefix + post_success_rewards([False] * 10 + [True] * 89)
    v3_1 = common_prefix + [0.0] * 99

    assert returns(stable)[0] > returns(reentry)[0] > returns(transient)[0]
    assert returns(stable)[1] > returns(reentry)[1] > returns(transient)[1]
    assert returns(v3_1) == returns(common_prefix + [0.0] * 99)
    np.testing.assert_allclose(returns(stable)[0] - returns(v3_1)[0], 19.8)
    np.testing.assert_allclose(returns(transient)[0] - returns(v3_1)[0], -99.0)


def test_non_success_trajectories_are_identical_to_v3_1() -> None:
    for rewards in (
        [0.4] * 200,
        [0.05] * 200,
        [1.05, -0.95] * 100,
        [-0.95] * 200,
    ):
        assert returns(rewards) == returns(list(rewards))


def test_maximum_retention_bonus_over_horizon() -> None:
    post_success = [0.2] * 199
    undiscounted, discounted = returns([0.0] + post_success)
    np.testing.assert_allclose(undiscounted, 39.8)
    assert 0.0 < discounted < undiscounted
