"""Pure initial DiffCRL artifact construction contracts."""

from dataclasses import replace
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.config import default_config
from src.continual.diffcrl import DiffCRLTrainer
from src.continual.diffcrl_artifacts import (
    build_initial_run_summary,
    stage_checkpoint_paths,
    validate_finite_model_state,
)
from src.continual.task_sequence import TaskSequence
from src.continual.trajectory_diffusion import DiffusionConfig, TrajectoryDiffusion
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
)


def test_initial_summary_preserves_schema_order_values_and_rng(tmp_path):
    base = default_config("diffcrl")
    tasks = ["reach-v3", "push-v3"]
    config = replace(
        base,
        continual=replace(
            base.continual,
            tasks=tasks,
            replay_mode="diffusion",
            replay_allocation="fixed-total",
            replay_budget=17,
        ),
        evaluation=replace(base.evaluation, mode="fixed-tasks", task_set_seed=9001),
    )
    sequence = TaskSequence.from_names(tasks)
    diffusion = TrajectoryDiffusion(
        DiffusionConfig(horizon=200, steps=4, width=4, num_tasks=2)
    )
    experts = {name: tmp_path / f"{name}.zip" for name in tasks}
    hashes = {name: f"hash-{index}" for index, name in enumerate(tasks)}
    random.seed(47)
    np.random.seed(53)
    torch.manual_seed(59)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()

    result = build_initial_run_summary(
        config, sequence, 200, diffusion, None, experts, hashes
    )

    assert list(result) == [
        "trajectories_per_task", "replay_allocation", "replay_budget",
        "replay_priority_strategy", "diffusion_epochs", "bc_epochs",
        "eval_episodes", "seed", "eval_seed", "diffusion_steps",
        "diffusion_width", "bc_normalization_mode", "bc_use_layer_norm",
        "replay_mode", "bc_loss",
        "diffusion_action_space", "diffusion_action_clamp_epsilon",
        "generated_action_projection", "evaluation_mode", "tasks",
        "environment_backend", "horizon", "task_set_seed", "diffusion",
        "experts", "historical_replay_budget", "split", "bc",
        "diffusion_fit", "normalization", "optimizers", "collection",
    ]
    assert result["tasks"] == tasks
    assert result["experts"] == {
        name: {"path": str(experts[name]), "sha256": hashes[name]} for name in tasks
    }
    assert result["historical_replay_budget"] == 17
    assert result["bc_use_layer_norm"] is True
    assert result["bc"]["use_layer_norm"] is True
    assert result["diffusion"] == {
        "horizon": 200,
        "features": 22,
        "steps": 4,
        "width": 4,
        "num_tasks": 2,
        "diffusion_action_space": "raw",
        "diffusion_action_clamp_epsilon": 1e-4,
    }
    assert random.getstate() == python_rng
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_rng[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
    assert numpy_after[2:] == numpy_rng[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)


def test_initial_summary_adds_only_flow_specific_sections():
    base = default_config("diffcrl")
    config = replace(
        base,
        continual=replace(base.continual, tasks=["reach-v3"], replay_mode="flow-matching"),
    )
    sequence = TaskSequence.from_names(["reach-v3"])
    flow = TrajectoryFlowMatching(
        FlowMatchingConfig(horizon=200, width=4, num_tasks=1, integration_steps=2)
    )

    result = build_initial_run_summary(
        config, sequence, 200, None, flow, {"reach-v3": "expert.zip"}, {"reach-v3": "hash"}
    )

    assert result["diffusion"] is None
    assert result["flow_matching"] == {
        "horizon": 200,
        "features": 22,
        "width": 4,
        "num_tasks": 1,
        "integration_steps": 2,
        "time_embedding_scale": 100.0,
    }
    assert result["flow_matching_fit"]["epochs"] == config.flow_matching.epochs
    assert "Flow Matching Adam fresh per fit" in result["optimizers"]


def test_new_run_keeps_configuration_manifest_and_summary_write_order(tmp_path):
    base = default_config("diffcrl")
    expert = tmp_path / "expert.zip"
    expert.write_bytes(b"expert")
    config = replace(
        base,
        continual=replace(
            base.continual,
            tasks=["reach-v3"],
            experts={"reach-v3": str(expert)},
            replay_mode="none",
        ),
        evaluation=replace(base.evaluation, mode="sampled"),
        runtime=replace(base.runtime, output=str(tmp_path / "run"), device="cpu"),
    )
    events = []
    identity = SimpleNamespace(to_dict=lambda: {"checkpoint_sha256": "expert"})

    with patch(
        "src.continual.diffcrl.validate_expert_checkpoint", return_value=identity
    ), patch(
        "src.continual.diffcrl.save_resolved_config",
        side_effect=lambda *_: events.append("resolved-config"),
    ), patch(
        "src.continual.diffcrl.create_run_manifest",
        side_effect=lambda *_args, **_kwargs: events.append("manifest"),
    ), patch(
        "src.continual.diffcrl.build_initial_run_summary",
        side_effect=lambda *_: events.append("build-summary") or {"schema": "same"},
    ), patch(
        "src.continual.diffcrl.write_json",
        side_effect=lambda path, value: events.append(
            ("write-summary", path.name, value)
        ),
    ):
        DiffCRLTrainer(config, progress=False)

    assert events == [
        "resolved-config",
        "manifest",
        "build-summary",
        ("write-summary", "config.json", {"schema": "same"}),
    ]


def test_stage_checkpoint_paths_preserve_all_legacy_names_and_suffixes(tmp_path):
    assert stage_checkpoint_paths(
        tmp_path,
        stage=2,
        task="hammer-v3",
        artifact_suffix=".resume_3",
        has_diffusion=True,
        has_flow_matching=False,
    ) == (
        tmp_path / "diffusion_after_task_2_hammer-v3.resume_3.pt",
        tmp_path / "general_policy_after_task_2_hammer-v3.resume_3.pt",
        None,
    )
    assert stage_checkpoint_paths(
        tmp_path,
        stage=2,
        task="hammer-v3",
        artifact_suffix="",
        has_diffusion=False,
        has_flow_matching=True,
    ) == (
        None,
        tmp_path / "general_policy_after_task_2_hammer-v3.pt",
        tmp_path / "flow_matching_after_task_2_hammer-v3.pt",
    )


def test_finite_state_validation_preserves_acceptance_and_error_contract():
    finite = torch.nn.Linear(2, 1)
    validate_finite_model_state(None, finite, None)
    with torch.no_grad():
        finite.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite trained state"):
        validate_finite_model_state(finite)
    with torch.no_grad():
        finite.weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite trained state"):
        validate_finite_model_state(finite)
