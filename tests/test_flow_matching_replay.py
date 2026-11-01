"""Flow Matching replay is deterministic, conditioned, decoded, and BC-compatible."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src.continual.diffusion_data import PHYSICAL_ACTION_SLICE
from src.continual.flow_matching_replay import (
    FlowMatchingReplayRequest,
    generate_previous_task_flow_matching_replay,
)
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
)
from src.continual.trajectory_diffusion import FeatureNormalizer
from src.support.dataset import bc_dataset


class RecordingFlow:
    def __init__(self, horizon=200):
        self.config = FlowMatchingConfig(
            horizon=horizon, width=4, num_tasks=3, integration_steps=7
        )
        self.velocity_model = Mock()
        self.calls = []
        self.last_sample_nfe = 0

    def sample(
        self,
        count,
        *,
        seed,
        task_ids,
        goals,
        integration_steps=None,
        execution_mode="legacy",
    ):
        self.calls.append(
            {
                "count": count,
                "seed": seed,
                "task_ids": task_ids.clone(),
                "goals": goals.clone(),
                "integration_steps": integration_steps,
                "execution_mode": execution_mode,
            }
        )
        generator = torch.Generator().manual_seed(seed)
        result = torch.randn((count, self.config.horizon, 22), generator=generator)
        result[..., :18] += task_ids[:, None, None]
        result[..., 18:] = 2.0
        self.last_sample_nfe = self.config.integration_steps if integration_steps is None else integration_steps
        return result


def metadata():
    names = ("reach-v3", "push-v3", "hammer-v3")
    return {
        "task_order": list(names),
        "generated_action_projection": "clip",
        "action_space": "raw",
        "action_clamp_epsilon": 1e-4,
        "training_banks": {
            name: {
                "task_name": name,
                "ordered_task_hashes": [f"{name}-{index}" for index in range(5)],
            }
            for name in names
        },
        "train_configurations": {
            name: {"indices": [1, 3]} for name in names
        },
    }


def request(**overrides):
    values = {
        "stage": 2,
        "trajectory_counts": {0: 3, 1: 3},
        "checkpoint": Path("/tmp/flow-checkpoint.pt"),
        "expected_state_hash": "expected-state",
        "task_names": ("reach-v3", "push-v3", "hammer-v3"),
        "backend": "metaworld-v3",
        "horizon": 200,
        "runtime_seed": 23,
        "device": "cpu",
        "reward_function_version": "v2",
        "action_space": "raw",
        "action_clamp_epsilon": 1e-4,
        "action_projection": "clip",
        "bc_clamp_epsilon": 1e-4,
        "progress": False,
    }
    values.update(overrides)
    return FlowMatchingReplayRequest(**values)


def reconstructed_goals(bank, selected, **kwargs):
    assert kwargs["backend"] == "metaworld-v3"
    assert kwargs["reward_function_version"] == "v2"
    base = 1.0 if bank["task_name"] == "reach-v3" else 2.0
    return [[base, float(index), -base] for index in selected]


def run_adapter(model=None, dynamic=None, goal=None, **request_overrides):
    model = RecordingFlow() if model is None else model
    dynamic = FeatureNormalizer(torch.zeros(22), torch.ones(22)) if dynamic is None else dynamic
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3)) if goal is None else goal
    with patch(
        "src.continual.flow_matching_replay.TrajectoryFlowMatching.load",
        return_value=(model, dynamic, goal, metadata()),
    ), patch(
        "src.continual.flow_matching_replay.reconstruct_goals",
        side_effect=reconstructed_goals,
    ), patch(
        "src.continual.flow_matching_replay.file_hash", return_value="checkpoint-sha"
    ), patch(
        "src.continual.flow_matching_replay.state_hash",
        return_value="expected-state",
    ):
        result = generate_previous_task_flow_matching_replay(
            request(**request_overrides)
        )
    return model, result


def test_shapes_conditions_group_counts_alignment_and_bc_compatibility():
    model, result = run_adapter()

    assert list(result.groups) == [0, 1]
    assert [group.shape for group in result.groups.values()] == [
        torch.Size((3, 200, 43)),
        torch.Size((3, 200, 43)),
    ]
    assert len(model.calls) == 2
    for old_index, (name, group) in enumerate(
        zip(("reach-v3", "push-v3"), result.groups.values())
    ):
        call = model.calls[old_index]
        assert call["count"] == 3
        assert call["execution_mode"] == "legacy"
        torch.testing.assert_close(
            call["task_ids"], torch.full((3,), old_index, dtype=torch.long)
        )
        assert call["goals"].shape == (3, 3)
        current = group[..., :18]
        previous = group[..., 18:36]
        torch.testing.assert_close(previous[:, 0], current[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(previous[:, 1:], current[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(
            group[..., 36:39], group[:, :1, 36:39].expand(-1, 200, -1), rtol=0, atol=0
        )
        assert bc_dataset(group).tensors[0].shape == (600, 39)
        assert bc_dataset(group).tensors[1].shape == (600, 4)
        source = result.sources[name]
        assert source["condition"] == old_index
        assert source["checkpoint_kind"] == "trajectory_flow_matching_22_v1"
        assert source["neural_function_evaluations"] == 7
        assert source["generation_seconds"] >= 0
        assert len(source["training_configuration_indices"]) == 3


def test_device_native_execution_is_forwarded():
    model, _ = run_adapter(
        stage=1,
        trajectory_counts={0: 3},
        execution_mode="device-native",
    )
    assert model.calls[0]["execution_mode"] == "device-native"


def test_goal_normalization_and_dynamic_denormalization_are_used():
    dynamic = FeatureNormalizer(torch.ones(22), torch.full((22,), 2.0))
    goal = FeatureNormalizer(
        torch.tensor([1.0, 0.0, -1.0]), torch.tensor([2.0, 4.0, 2.0])
    )
    model, result = run_adapter(dynamic=dynamic, goal=goal)

    first_call = model.calls[0]
    selected = result.sources["reach-v3"]["training_configuration_indices"]
    physical_goals = torch.tensor(
        reconstructed_goals(
            metadata()["training_banks"]["reach-v3"],
            selected,
            backend="metaworld-v3",
            reward_function_version="v2",
        )
    )
    expected_normalized = goal.normalize(physical_goals[:, None, :])[:, 0]
    torch.testing.assert_close(first_call["goals"], expected_normalized)
    expected_first_current = 2 * torch.randn(
        (3, 200, 22), generator=torch.Generator().manual_seed(first_call["seed"])
    )[..., :18] + 1
    torch.testing.assert_close(result.groups[0][..., :18], expected_first_current)


def test_generation_is_deterministic_with_existing_seed_convention():
    first_model, first = run_adapter()
    second_model, second = run_adapter()

    assert [call["seed"] for call in first_model.calls] == [102023, 102024]
    assert [call["seed"] for call in second_model.calls] == [102023, 102024]
    for sources in (first.sources, second.sources):
        for source in sources.values():
            source.pop("generation_seconds")
    assert first.sources == second.sources
    for index in first.groups:
        torch.testing.assert_close(first.groups[index], second.groups[index], rtol=0, atol=0)


def test_zero_allocations_are_not_generated():
    model, result = run_adapter(trajectory_counts={0: 2, 1: 0})
    assert list(result.groups) == [0]
    assert list(result.sources) == ["reach-v3"]
    assert result.groups[0].shape == (2, 200, 43)
    assert result.sources["reach-v3"]["trajectory_count"] == 2
    assert len(model.calls) == 1


def test_raw_actions_are_only_clipped_when_explicitly_requested():
    _, clipped = run_adapter(action_projection="clip")
    assert torch.all(clipped.groups[0][..., PHYSICAL_ACTION_SLICE] == 1)
    assert clipped.sources["reach-v3"]["pre_projection_oob_component_fraction"] == 1.0
    assert clipped.sources["reach-v3"]["post_projection_oob_component_fraction"] == 0.0

    unprojected_metadata = metadata()
    unprojected_metadata["generated_action_projection"] = "none"
    model = RecordingFlow()
    identity = FeatureNormalizer(torch.zeros(22), torch.ones(22))
    with patch(
        "src.continual.flow_matching_replay.TrajectoryFlowMatching.load",
        return_value=(model, identity, FeatureNormalizer(torch.zeros(3), torch.ones(3)), unprojected_metadata),
    ), patch(
        "src.continual.flow_matching_replay.reconstruct_goals",
        side_effect=reconstructed_goals,
    ), patch(
        "src.continual.flow_matching_replay.file_hash", return_value="sha"
    ), patch(
        "src.continual.flow_matching_replay.state_hash", return_value="expected-state"
    ):
        unprojected = generate_previous_task_flow_matching_replay(
            request(action_projection="none")
        )
    assert torch.all(unprojected.groups[0][..., PHYSICAL_ACTION_SLICE] == 2)
    assert unprojected.sources["reach-v3"]["post_projection_oob_component_fraction"] == 1.0


def test_pre_tanh_actions_decode_without_silent_projection():
    pre_tanh_metadata = metadata()
    pre_tanh_metadata.update(action_space="pre-tanh", generated_action_projection="none")
    model = RecordingFlow()
    identity = FeatureNormalizer(torch.zeros(22), torch.ones(22))
    with patch(
        "src.continual.flow_matching_replay.TrajectoryFlowMatching.load",
        return_value=(model, identity, FeatureNormalizer(torch.zeros(3), torch.ones(3)), pre_tanh_metadata),
    ), patch(
        "src.continual.flow_matching_replay.reconstruct_goals",
        side_effect=reconstructed_goals,
    ), patch(
        "src.continual.flow_matching_replay.file_hash", return_value="sha"
    ), patch(
        "src.continual.flow_matching_replay.state_hash", return_value="expected-state"
    ):
        result = generate_previous_task_flow_matching_replay(
            request(action_space="pre-tanh", action_projection="none")
        )
    expected = torch.tanh(torch.tensor(2.0))
    torch.testing.assert_close(
        result.groups[0][..., PHYSICAL_ACTION_SLICE],
        torch.full((3, 200, 4), expected),
    )


def test_checkpoint_kind_validation_propagates():
    with patch(
        "src.continual.flow_matching_replay.TrajectoryFlowMatching.load",
        side_effect=ValueError("Checkpoint is not trajectory_flow_matching_22_v1."),
    ), pytest.raises(ValueError, match="trajectory_flow_matching_22_v1"):
        generate_previous_task_flow_matching_replay(request())


def test_real_incompatible_checkpoint_kind_is_rejected(tmp_path):
    path = tmp_path / "diffusion.pt"
    torch.save({"kind": "trajectory_diffusion_22_v1"}, path)
    with pytest.raises(ValueError, match="trajectory_flow_matching_22_v1"):
        generate_previous_task_flow_matching_replay(request(checkpoint=path))


def test_checkpoint_metadata_mismatch_is_rejected_before_sampling():
    bad = metadata()
    bad["task_order"] = ["wrong-v3"]
    model = RecordingFlow()
    normalizer = FeatureNormalizer(torch.zeros(22), torch.ones(22))
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3))
    with patch(
        "src.continual.flow_matching_replay.TrajectoryFlowMatching.load",
        return_value=(model, normalizer, goal, bad),
    ), pytest.raises(ValueError, match="metadata differs"):
        generate_previous_task_flow_matching_replay(request())
    assert model.calls == []
