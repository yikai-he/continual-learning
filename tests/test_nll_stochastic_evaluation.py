"""Action-mode and RNG contracts for saved NLL checkpoint evaluation."""

import numpy as np
import pytest
import torch

from scripts.evaluation.evaluate_nll_execution import (
    EvaluationActionPolicy,
    matrix_summary,
)
from src.continual.bc_policy import GeneralPolicy


def policy(objective="squashed_gaussian_nll"):
    torch.manual_seed(4)
    return GeneralPolicy(
        -np.ones(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        initial_log_std=-1.0,
        objective=objective,
    )


def sample_sequence(model, seed, count=4):
    observation = np.linspace(-0.2, 0.2, 39, dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return np.stack([model.act(observation) for _ in range(count)])


def test_deterministic_adapter_preserves_existing_mean_action():
    model = policy()
    observation = np.zeros(39, dtype=np.float32)
    expected = model.act(observation, deterministic=True)
    actual = EvaluationActionPolicy(model, "deterministic").act(observation)
    assert np.array_equal(actual, expected)


def test_stochastic_sampling_is_bounded_reproducible_and_not_reseeded_per_step():
    adapter = EvaluationActionPolicy(policy(), "stochastic")
    first = sample_sequence(adapter, 12)
    second = sample_sequence(adapter, 12)
    different = sample_sequence(adapter, 13)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert any(not np.array_equal(first[0], row) for row in first[1:])
    assert np.all(first >= -1) and np.all(first <= 1)


def test_stochastic_evaluation_restores_torch_rng_state():
    adapter = EvaluationActionPolicy(policy(), "stochastic")
    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    sample_sequence(adapter, 3)
    assert torch.equal(torch.get_rng_state(), before)


def test_mse_policy_rejects_stochastic_evaluation():
    with pytest.raises(ValueError, match="learned NLL distribution"):
        EvaluationActionPolicy(policy("deterministic_mse"), "stochastic")


def test_summary_uses_existing_matrix_metrics():
    matrix = {
        "R": [[1.0, None], [0.8, 0.6]],
        "metrics_by_stage": {
            "0": {"average_performance": 1.0, "forgetting": 0.0},
            "1": {"average_performance": 0.7, "forgetting": 0.2},
        },
    }
    assert matrix_summary(matrix) == {
        "final_average_performance": 0.7,
        "average_forgetting": 0.2,
        "immediate_performance": 0.8,
        "final_per_task_success": [0.8, 0.6],
    }
