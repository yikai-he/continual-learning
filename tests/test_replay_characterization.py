"""Exact pre-refactor characterization of both generated replay providers."""

import hashlib
import json
from unittest.mock import patch

import pytest

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
    assert _source_hash(result) == (
        "2f97c92d685ef894660e8c5fc84513ee38323b8a41d1c6c4d0c3731d0e503e6d"
    )
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
    assert _source_hash(result) == (
        "faae39a54ce04922d15a4cc5c65cdacf3f0b8df9c91785faabad28ec856c4eae"
    )
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
