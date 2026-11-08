"""Focused contracts for optional task-conditioned behavioral cloning."""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.continual.bc_policy import GeneralPolicy, fit_bc
from src.continual.evaluation import EvaluationMatrix, evaluate_stage
from src.continual.stage_dataset import bc_dataset
from src.continual.task_sequence import TaskSequence


TASKS = ("button-press-v3", "hammer-v3", "window-close-v3")


def policy():
    return GeneralPolicy(
        -np.ones(4), np.ones(4), task_conditioning="one_hot", task_order=TASKS
    )


def test_policy_input_one_hot_shape_bounds_and_validation():
    model = policy()
    observations = torch.zeros(2, 39)
    inputs = model.policy_input(observations, torch.tensor([0, 2]))
    assert model.network[0].in_features == 42
    assert torch.equal(inputs[:, 39:], torch.tensor([[1, 0, 0], [0, 0, 1]]).float())
    actions = model(observations, torch.tensor([0, 2]))
    assert actions.shape == (2, 4)
    assert torch.all(actions >= -1) and torch.all(actions <= 1)
    with pytest.raises(ValueError, match="outside"):
        model(observations, torch.tensor([0, 3]))
    with pytest.raises(ValueError, match="required"):
        model(observations)


def test_stage_bc_labels_are_repeated_per_timestep():
    packed = torch.zeros(2, 4, 43)
    dataset = bc_dataset(packed, torch.tensor([1, 2]))
    assert dataset.tensors[2].tolist() == [1] * 4 + [2] * 4


def test_task_conditioned_training_and_checkpoint_contract(tmp_path):
    model = policy()
    observations = torch.randn(12, 39)
    actions = torch.tanh(torch.randn(12, 4))
    labels = torch.arange(12) % len(TASKS)
    dataset = TensorDataset(observations, actions, labels)
    before = model.network[0].weight.detach().clone()
    fit_bc(model, dataset, dataset, epochs=1, batch_size=4)
    assert not torch.equal(before, model.network[0].weight)

    checkpoint = tmp_path / "conditioned.pt"
    model.save(checkpoint)
    restored = GeneralPolicy.load(
        checkpoint,
        expected_task_conditioning="one_hot",
        expected_task_order=TASKS,
    )
    assert restored.task_order == TASKS
    with pytest.raises(ValueError, match="ordering"):
        GeneralPolicy.load(checkpoint, expected_task_order=tuple(reversed(TASKS)))
    with pytest.raises(ValueError, match="mode"):
        GeneralPolicy.load(checkpoint, expected_task_conditioning="none")


def test_task_agnostic_architecture_is_unchanged():
    model = GeneralPolicy(-np.ones(4), np.ones(4))
    assert model.network[0].in_features == 39
    assert model(torch.zeros(2, 39)).shape == (2, 4)


def test_stage_evaluation_provides_configured_task_indices(monkeypatch):
    seen = []

    def fake_evaluate_task(policy, task, **kwargs):
        seen.append((task.task_name, kwargs["policy_task_id"]))
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "success_rate": 0.0,
            "mean_episode_length": 1.0,
            "evaluation_mode": "sampled",
        }

    monkeypatch.setattr("src.continual.evaluation.evaluate_task", fake_evaluate_task)
    sequence = TaskSequence.from_names(TASKS)
    matrix = EvaluationMatrix(sequence)
    evaluate_stage(object(), matrix, 2)
    assert seen == list(zip(TASKS, range(len(TASKS))))
