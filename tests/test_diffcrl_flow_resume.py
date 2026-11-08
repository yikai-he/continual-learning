"""Validated deterministic stage-boundary resume for Flow Matching runs."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.config import default_config, save_resolved_config
from src.continual.bc_policy import GeneralPolicy
from src.continual.diffcrl import DiffCRLTrainer, StageContext
from src.continual.evaluation import EvaluationMatrix
from src.continual.resume import write_stage_boundary
from src.continual.task_sequence import TaskSequence
from src.continual.trajectory_diffusion import FeatureNormalizer
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
)
from src.support.reproducibility import file_hash, state_hash
from src.support.run_manifest import create_run_manifest


TASKS = ["reach-v3", "push-v3"]


def flow_config(tmp_path, run):
    base = default_config("diffcrl")
    experts = {}
    for task in TASKS:
        path = tmp_path / "experts" / task / "expert.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes((task + " expert").encode())
        experts[task] = str(path)
    return replace(
        base,
        continual=replace(
            base.continual,
            tasks=list(TASKS),
            experts=experts,
            replay_mode="flow-matching",
            trajectories_per_task=2,
        ),
        evaluation=replace(base.evaluation, mode="sampled", episodes=1),
        flow_matching=replace(
            base.flow_matching,
            epochs=1,
            batch_size=1,
            width=4,
            integration_steps=2,
        ),
        runtime=replace(base.runtime, output=str(run), device="cpu", seed=7),
    )


def metadata(training_banks, train_configurations, **overrides):
    value = {
        "stage": 0,
        "task": TASKS[0],
        "seed": 7,
        "task_order": list(TASKS),
        "training_banks": training_banks,
        "train_configurations": train_configurations,
        "generated_action_projection": "none",
        "action_space": "raw",
        "action_clamp_epsilon": 1e-4,
    }
    value.update(overrides)
    return value


def build_completed_flow_run(
    tmp_path,
    *,
    checkpoint_metadata=None,
    model_config=None,
):
    run = tmp_path / "run"
    run.mkdir()
    config = flow_config(tmp_path, run)
    save_resolved_config(config, run)
    experts = {task: Path(path) for task, path in config.continual.experts.items()}
    create_run_manifest(
        run,
        config,
        experts={
            task: {"checkpoint_sha256": file_hash(path)}
            for task, path in experts.items()
        },
    )
    training_banks = {TASKS[0]: {"ordered_task_hashes": ["task-0"]}}
    train_configurations = {TASKS[0]: {"indices": [0]}}

    torch.manual_seed(31)
    flow = TrajectoryFlowMatching(
        model_config
        or FlowMatchingConfig(
            horizon=200,
            width=4,
            num_tasks=2,
            integration_steps=2,
            time_embedding_scale=100.0,
        )
    )
    flow_path = run / "flow_matching_after_task_0_reach-v3.pt"
    dynamic = FeatureNormalizer(torch.arange(22.0), torch.arange(1.0, 23.0))
    goal = FeatureNormalizer(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 4.0]))
    flow.save(
        flow_path,
        dynamic,
        goal,
        checkpoint_metadata
        if checkpoint_metadata is not None
        else metadata(training_banks, train_configurations),
    )

    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy_path = run / "general_policy_after_task_0_reach-v3.pt"
    policy.save(policy_path, optimizer=optimizer, metadata={"stage": 0})

    matrix = EvaluationMatrix(TaskSequence.from_names(TASKS))
    matrix.record(
        0,
        0,
        {
            "mean_return": 1.0,
            "std_return": 0.0,
            "success_rate": 0.5,
            "mean_episode_length": 200.0,
        },
    )
    stage_dir = run / "stage_0_reach-v3"
    stage_dir.mkdir()
    report_path = stage_dir / "report.json"
    report_path.write_text("{}\n")
    matrix_path = stage_dir / "evaluation_matrix.json"
    matrix_path.write_text(json.dumps(matrix.to_dict()) + "\n")
    marker_path = write_stage_boundary(
        run,
        stage=0,
        task=TASKS[0],
        policy_path=policy_path,
        diffusion_path=None,
        flow_matching_path=flow_path,
        report_path=report_path,
        matrix_path=matrix_path,
        training_banks=training_banks,
        train_configurations=train_configurations,
    )
    return SimpleNamespace(
        run=run,
        config=config,
        flow=flow,
        flow_path=flow_path,
        dynamic=dynamic,
        goal=goal,
        policy=policy,
        matrix=matrix,
        marker_path=marker_path,
        training_banks=training_banks,
        train_configurations=train_configurations,
    )


def resume(completed, config=None):
    identity = SimpleNamespace(to_dict=lambda: {})
    with patch(
        "src.continual.diffcrl.validate_expert_checkpoint", return_value=identity
    ):
        return DiffCRLTrainer(
            completed.config if config is None else config,
            progress=False,
            resume_from=completed.run,
        )


def rewrite_marker_hash(completed, artifact, path):
    marker = json.loads(completed.marker_path.read_text())
    marker["artifacts"][artifact] = {
        "path": str(Path(path).relative_to(completed.run)),
        "sha256": file_hash(path),
    }
    completed.marker_path.write_text(json.dumps(marker) + "\n")


def test_successful_flow_resume_restores_all_stage_state(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    marker_before = completed.marker_path.read_bytes()
    restored = resume(completed)

    assert restored.next_stage == 1
    assert restored.previous_flow_matching == completed.flow_path
    assert restored.previous_diffusion is None
    assert state_hash(restored.flow_matching) == state_hash(completed.flow)
    assert state_hash(restored.policy) == state_hash(completed.policy)
    torch.testing.assert_close(
        restored.flow_dynamic_normalizer.mean, completed.dynamic.mean, rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored.flow_goal_normalizer.scale, completed.goal.scale, rtol=0, atol=0
    )
    assert restored.training_banks == completed.training_banks
    assert restored.train_configurations == completed.train_configurations
    assert restored.matrix.to_dict() == completed.matrix.to_dict()
    assert completed.marker_path.read_bytes() == marker_before


def test_flow_resume_reconstructs_four_layer_general_policy(tmp_path):
    hidden_sizes = (256, 256, 256, 256)
    completed = build_completed_flow_run(tmp_path)

    restored = resume(completed)

    assert restored.policy.hidden_sizes == hidden_sizes
    assert len(restored.policy.network) == 11
    for name, expected in completed.policy.state_dict().items():
        torch.testing.assert_close(
            restored.policy.state_dict()[name], expected, rtol=0, atol=0
        )


def test_resumed_replay_owns_previous_completed_checkpoint(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    restored = resume(completed)
    context = StageContext(
        index=1,
        task=restored.sequence.task(1),
        trajectory_count=2,
        directory=completed.run / "stage_1_push-v3",
        policy_before_hash=state_hash(restored.policy),
        diffusion_before_hash=None,
        flow_matching_before_hash=state_hash(restored.flow_matching),
    )
    generated = SimpleNamespace(groups={}, sources={})
    with patch(
        "src.continual.diffcrl.generate_previous_task_flow_matching_replay",
        return_value=generated,
    ) as replay:
        restored._generate_replay(context)
    request = replay.call_args.args[0]
    assert request.checkpoint == completed.flow_path
    assert request.expected_state_hash == state_hash(restored.flow_matching)
    assert request.runtime_seed == 7


def test_rng_state_is_restored_at_flow_boundary(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)
    resume(completed)
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_flow_artifact_is_rejected(tmp_path, damage):
    completed = build_completed_flow_run(tmp_path)
    if damage == "missing":
        completed.flow_path.unlink()
        expected = FileNotFoundError
    else:
        completed.flow_path.write_bytes(b"corrupt")
        expected = ValueError
    with pytest.raises(expected):
        resume(completed)


def test_missing_flow_artifact_key_is_rejected(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    marker = json.loads(completed.marker_path.read_text())
    marker["artifacts"].pop("flow_matching")
    completed.marker_path.write_text(json.dumps(marker) + "\n")
    with pytest.raises(ValueError, match="no Flow checkpoint"):
        resume(completed)


def test_wrong_checkpoint_kind_is_rejected(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    torch.save({"kind": "trajectory_diffusion_22_v1"}, completed.flow_path)
    rewrite_marker_hash(completed, "flow_matching", completed.flow_path)
    with pytest.raises(ValueError, match="trajectory_flow_matching_22_v1"):
        resume(completed)


def test_generator_artifact_mode_mismatch_is_rejected(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    diffusion = completed.run / "unexpected-diffusion.pt"
    diffusion.write_bytes(b"diffusion")
    rewrite_marker_hash(completed, "diffusion", diffusion)
    with pytest.raises(ValueError, match="unexpectedly contains diffusion"):
        resume(completed)


def test_missing_flow_metadata_is_rejected(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    replacement = completed.run / "replacement.pt"
    completed.flow.save(replacement, completed.dynamic, completed.goal, {})
    replacement.replace(completed.flow_path)
    rewrite_marker_hash(completed, "flow_matching", completed.flow_path)
    with pytest.raises(ValueError, match="metadata is incomplete"):
        resume(completed)


@pytest.mark.parametrize(
    "override",
    [
        {"stage": 1},
        {"task": "push-v3"},
        {"task_order": ["push-v3", "reach-v3"]},
    ],
)
def test_checkpoint_stage_task_and_order_mismatch_is_rejected(tmp_path, override):
    completed = build_completed_flow_run(tmp_path)
    replacement = completed.run / "replacement.pt"
    completed.flow.save(
        replacement,
        completed.dynamic,
        completed.goal,
        metadata(completed.training_banks, completed.train_configurations, **override),
    )
    replacement.replace(completed.flow_path)
    rewrite_marker_hash(completed, "flow_matching", completed.flow_path)
    with pytest.raises(ValueError, match="metadata differs"):
        resume(completed)


def test_checkpoint_model_configuration_mismatch_is_rejected(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    wrong = TrajectoryFlowMatching(
        FlowMatchingConfig(
            horizon=200,
            width=8,
            num_tasks=2,
            integration_steps=2,
            time_embedding_scale=100.0,
        )
    )
    replacement = completed.run / "replacement.pt"
    wrong.save(
        replacement,
        completed.dynamic,
        completed.goal,
        metadata(completed.training_banks, completed.train_configurations),
    )
    replacement.replace(completed.flow_path)
    rewrite_marker_hash(completed, "flow_matching", completed.flow_path)
    with pytest.raises(ValueError, match="model configuration differs"):
        resume(completed)


def test_scientific_flow_configuration_mismatch_is_rejected(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    changed = replace(
        completed.config,
        flow_matching=replace(completed.config.flow_matching, integration_steps=3),
    )
    with pytest.raises(ValueError, match="scientific configuration"):
        resume(completed, changed)


def test_incomplete_next_stage_is_ignored_and_preserved(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    partial = completed.run / "stage_1_push-v3"
    partial.mkdir()
    evidence = partial / "partial.bin"
    evidence.write_bytes(b"do not overwrite")
    restored = resume(completed)
    assert restored.next_stage == 1
    assert evidence.read_bytes() == b"do not overwrite"


def test_flow_save_load_continuation_is_bitwise_equivalent(tmp_path):
    completed = build_completed_flow_run(tmp_path)
    restored = resume(completed)
    generator = torch.Generator().manual_seed(101)
    updates = {
        name: torch.randn(value.shape, generator=generator) * 1e-5
        for name, value in completed.flow.state_dict().items()
        if value.is_floating_point()
    }
    with torch.no_grad():
        for model in (completed.flow, restored.flow_matching):
            state = model.state_dict()
            for name, update in updates.items():
                state[name].add_(update)
    assert state_hash(completed.flow) == state_hash(restored.flow_matching)


@pytest.mark.parametrize("replay_mode", ["real", "none"])
def test_generator_free_stage_restores_policy_optimizer_rng_and_matrix(
    tmp_path, replay_mode
):
    run = tmp_path / replay_mode
    run.mkdir()
    flow = flow_config(tmp_path, run)
    config = replace(
        flow,
        continual=replace(flow.continual, replay_mode=replay_mode),
        runtime=replace(flow.runtime, output=str(run)),
    )
    save_resolved_config(config, run)
    experts = {task: Path(path) for task, path in config.continual.experts.items()}
    create_run_manifest(
        run,
        config,
        experts={
            task: {"checkpoint_sha256": file_hash(path)}
            for task, path in experts.items()
        },
    )
    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss = policy(torch.zeros(2, 39)).square().sum()
    loss.backward()
    optimizer.step()
    policy_path = run / "general_policy_after_task_0_reach-v3.pt"
    policy.save(policy_path, optimizer=optimizer, metadata={"stage": 0})
    matrix = EvaluationMatrix(TaskSequence.from_names(TASKS))
    matrix.record(
        0,
        0,
        {
            "mean_return": 0.0,
            "std_return": 0.0,
            "success_rate": 0.0,
            "mean_episode_length": 200.0,
        },
    )
    stage_dir = run / "stage_0_reach-v3"
    stage_dir.mkdir()
    report = stage_dir / "report.json"
    report_payload = {
        "sources": {"reach-v3": {"kind": "real_expert"}},
        "generator_mode": replay_mode,
    }
    report.write_text(json.dumps(report_payload) + "\n")
    matrix_path = stage_dir / "matrix.json"
    matrix_path.write_text(json.dumps(matrix.to_dict()) + "\n")
    write_stage_boundary(
        run,
        stage=0,
        task=TASKS[0],
        policy_path=policy_path,
        diffusion_path=None,
        report_path=report,
        matrix_path=matrix_path,
        training_banks={"reach-v3": {"ordered_task_hashes": ["task-0"]}},
        train_configurations={"reach-v3": {"indices": [0]}},
    )
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)
    completed = SimpleNamespace(run=run, config=config)
    restored = resume(completed)
    assert restored.next_stage == 1
    assert restored.diffusion is None
    assert restored.flow_matching is None
    assert restored.previous_diffusion is None
    assert restored.previous_flow_matching is None
    assert state_hash(restored.policy) == state_hash(policy)
    assert restored.policy_optimizer is not None
    assert restored.policy_optimizer.state_dict()["state"]
    assert restored.matrix.to_dict() == matrix.to_dict()
    assert restored.training_banks == {
        "reach-v3": {"ordered_task_hashes": ["task-0"]}
    }
    assert restored.train_configurations == {"reach-v3": {"indices": [0]}}
    assert json.loads(report.read_text()) == report_payload
    assert (run / "stage_0_COMPLETE.json").is_file()
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)
