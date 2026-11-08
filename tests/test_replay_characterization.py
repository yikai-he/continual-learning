"""Exact pre-refactor characterization of both generated replay providers."""

import hashlib
import json
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
    metadata as ddpm_metadata,
    request as ddpm_request,
)
from tests.test_flow_matching_replay import request as flow_request, run_adapter


def _tensor_hashes(result):
    return {
        index: hashlib.sha256(group.numpy().tobytes()).hexdigest()
        for index, group in result.groups.items()
    }


def _source_hash(result):
    encoded = json.dumps(
        result.sources,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_shared_replay_contract(result):
    assert list(result.groups) == [0, 1]
    assert list(result.sources) == ["reach-v3", "push-v3"]
    assert [tuple(group.shape) for group in result.groups.values()] == [
        (3, 200, 43),
        (3, 200, 43),
    ]
    assert all(group.dtype == torch.float32 for group in result.groups.values())
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
        assert source["generated_action_projection"] == "clip"
        assert source["pre_projection_oob_component_fraction"] == 1.0
        assert source["post_projection_oob_component_fraction"] == 0.0


def test_ddpm_replay_exact_tensor_and_metadata_characterization():
    model = FakeDiffusion()
    normalizer = IdentityNormalizer()
    with patch(
        "src.continual.diffcrl_replay.TrajectoryDiffusion.load",
        return_value=(model, normalizer, normalizer, ddpm_metadata()),
    ), patch(
        "src.continual.diffcrl_replay.reconstruct_goals",
        side_effect=lambda bank, selected, **_: [
            [1.0, float(index), -1.0] for index in selected
        ],
    ), patch(
        "src.continual.diffcrl_replay.file_hash", return_value="sha"
    ), patch(
        "src.continual.diffcrl_replay.state_hash",
        return_value="expected-state",
    ):
        result = generate_previous_task_replay(ddpm_request())

    assert _tensor_hashes(result) == {
        0: "2ad41d45f49d9e47bb493988234da52e5dcacf6dfdf9a838857e47e754a20268",
        1: "13d277632a722ec0eba316bbe8a2119f9968c4dd4efcfbc482f4e6dbf2331e3d",
    }
    _assert_shared_replay_contract(result)
    assert _source_hash(result) == (
        "65a48d5865745b0d1e28bdcc3cd9e87e6530123c190f02b56a35b43c0c97e232"
    )
    for source in result.sources.values():
        assert source["checkpoint"] == "/tmp/replay-checkpoint.pt"
        assert source["checkpoint_sha256"] == "sha"
        assert source["diffusion_steps"] == 100
        assert source["neural_function_evaluations"] == 100
    assert [call[1] for call in model.calls] == [102023, 102024]


def test_flow_replay_exact_tensor_and_metadata_characterization():
    with patch(
        "src.continual.flow_matching_replay.time.perf_counter",
        side_effect=[10.0, 10.25, 20.0, 20.5],
    ):
        model, result = run_adapter()

    assert _tensor_hashes(result) == {
        0: "2ad41d45f49d9e47bb493988234da52e5dcacf6dfdf9a838857e47e754a20268",
        1: "d024eb5acffaa2c3c59518a6d34de763e29ae85c688f6d6778f4d53c162289b0",
    }
    _assert_shared_replay_contract(result)
    assert _source_hash(result) == (
        "ebc0d651e610f124dea0d9b820d44e3e43a4a2032e7dbe69b556aac45be33cb7"
    )
    for source in result.sources.values():
        assert source["generator"] == "flow_matching"
        assert source["checkpoint_kind"] == "trajectory_flow_matching_22_v1"
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
