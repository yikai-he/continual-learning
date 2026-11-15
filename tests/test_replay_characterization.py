"""Exact pre-refactor characterization of both generated replay providers."""

from unittest.mock import patch

import pytest
import torch

from src.continual.diffcrl_replay import generate_previous_task_replay
from src.continual.flow_matching_replay import (
    generate_previous_task_flow_matching_replay,
)
from tests.test_diffcrl_replay import (
    FakeDiffusion,
    IdentityNormalizer,
)
from tests.test_diffcrl_replay import (
    metadata as ddpm_metadata,
)
from tests.test_diffcrl_replay import (
    request as ddpm_request,
)
from tests.test_flow_matching_replay import request as flow_request
from tests.test_flow_matching_replay import run_adapter


def _assert_shared_replay_contract(result):
    assert list(result.groups) == [0, 1]
    assert list(result.sources) == ["reach-v3", "push-v3"]
    assert [tuple(group.shape) for group in result.groups.values()] == [
        (3, 200, 43),
        (3, 200, 43),
    ]
    assert all(group.dtype == torch.float32 for group in result.groups.values())
    assert all(torch.isfinite(group).all() for group in result.groups.values())
    for group in result.groups.values():
        assert torch.all(group[..., 18:39] == 2.0)
        assert torch.all(group[..., 39:43] == 1.0)
        assert not torch.equal(group[:, 1:, 18:36], group[:, :-1, :18])
    for index, name in enumerate(("reach-v3", "push-v3")):
        source = result.sources[name]
        assert source["kind"] == "generated"
        assert source["condition"] == index
        assert source["trajectory_count"] == 3
        assert source["sampling_seed"] == 102023 + index
        assert source["training_configuration_indices"] == (
            [1, 3, 3] if index == 0 else [1, 1, 3]
        )
        assert source["training_configuration_hashes"] == [
            f"{name}-{selected}"
            for selected in source["training_configuration_indices"]
        ]


def test_ddpm_replay_direct_43d_and_metadata_characterization():
    model = FakeDiffusion()
    normalizer = IdentityNormalizer()
    with patch(
        "src.continual.diffcrl_replay.TrajectoryDiffusion.load",
        return_value=(model, normalizer, normalizer, ddpm_metadata()),
    ), patch(
        "src.continual.replay_common.reconstruct_goals",
        side_effect=lambda bank, selected, **_: [
            [1.0, float(index), -1.0] for index in selected
        ],
    ), patch(
        "src.continual.diffcrl_replay.file_hash", return_value="sha"
    ) as checkpoint_hash, patch(
        "src.continual.diffcrl_replay.state_hash",
        return_value="expected-state",
    ):
        result = generate_previous_task_replay(ddpm_request())

    _assert_shared_replay_contract(result)
    for source in result.sources.values():
        assert source["checkpoint"] == "/tmp/replay-checkpoint.pt"
        assert source["checkpoint_sha256"] == "sha"
        assert source["diffusion_steps"] == 100
        assert source["neural_function_evaluations"] == 100
    assert [call[1] for call in model.calls] == [102023, 102024]
    checkpoint_hash.assert_called_once_with(ddpm_request().checkpoint)


def test_flow_replay_direct_43d_and_metadata_characterization():
    with patch(
        "src.continual.flow_matching_replay.time.perf_counter",
        side_effect=[10.0, 10.25, 20.0, 20.5],
    ):
        model, result = run_adapter()

    _assert_shared_replay_contract(result)
    for source in result.sources.values():
        assert source["generator"] == "flow_matching"
        assert source["checkpoint_kind"] == "trajectory_flow_matching_43_v3"
        assert source["checkpoint"] == "/tmp/flow-checkpoint.pt"
        assert source["checkpoint_sha256"] == "checkpoint-sha"
        assert source["integration_steps"] == 7
        assert source["neural_function_evaluations"] == 7
        assert source["ode_solver"] == "euler"
        assert source["rtol"] == source["atol"] == 1e-5
    assert [call["seed"] for call in model.calls] == [102023, 102024]


def test_invalid_allocations_keep_provider_specific_errors():
    model = FakeDiffusion()
    normalizer = IdentityNormalizer()
    with patch(
        "src.continual.diffcrl_replay.TrajectoryDiffusion.load",
        return_value=(model, normalizer, normalizer, ddpm_metadata()),
    ), patch(
        "src.continual.diffcrl_replay.state_hash", return_value="expected-state"
    ), pytest.raises(ValueError, match="Invalid DDPM replay trajectory allocation"):
        generate_previous_task_replay(
            ddpm_request(trajectory_counts={0: -1, 1: 3})
        )

    with pytest.raises(ValueError, match="Invalid Flow Matching replay request"):
        generate_previous_task_flow_matching_replay(
            flow_request(trajectory_counts={0: -1, 1: 3})
        )


def test_zero_allocations_preserve_task_order_without_sampling():
    model = FakeDiffusion()
    normalizer = IdentityNormalizer()
    with patch(
        "src.continual.diffcrl_replay.TrajectoryDiffusion.load",
        return_value=(model, normalizer, normalizer, ddpm_metadata()),
    ), patch(
        "src.continual.replay_common.reconstruct_goals",
        side_effect=lambda bank, selected, **_: [
            [1.0, float(index), -1.0] for index in selected
        ],
    ), patch(
        "src.continual.diffcrl_replay.file_hash", return_value="sha"
    ), patch(
        "src.continual.diffcrl_replay.state_hash", return_value="expected-state"
    ):
        ddpm = generate_previous_task_replay(
            ddpm_request(trajectory_counts={0: 0, 1: 2})
        )

    flow_model, flow = run_adapter(trajectory_counts={0: 0, 1: 2})

    assert list(ddpm.groups) == list(flow.groups) == [1]
    assert list(ddpm.sources) == list(flow.sources) == ["push-v3"]
    assert [call[1] for call in model.calls] == [102024]
    assert [call["seed"] for call in flow_model.calls] == [102024]
    torch.testing.assert_close(ddpm.groups[1], flow.groups[1], rtol=0, atol=0)
