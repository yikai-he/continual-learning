"""Fast stage-boundary resume tests; no environments or training loops run."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.config import default_config, save_resolved_config
from src.continual.bc_policy import GeneralPolicy
from src.continual.diffcrl import DiffCRLTrainer
from src.continual.evaluation import EvaluationMatrix
from src.continual.resume import (
    completed_stage_markers,
    load_boundary_state,
    restore_rng_state,
    validate_resume_configuration,
    write_stage_boundary,
)
from src.continual.task_sequence import TaskSequence
from src.continual.trajectory_diffusion import (
    DiffusionConfig,
    FeatureNormalizer,
    TrajectoryDiffusion,
)
from src.support.reproducibility import file_hash
from src.support.run_manifest import create_run_manifest


TASKS = ["reach-v3", "push-v3", "hammer-v3"]


def config_for(run_dir, experts, *, replay="diffusion", seed=0):
    config = default_config("diffcrl")
    return replace(
        config,
        continual=replace(
            config.continual,
            tasks=list(TASKS),
            experts={task: str(experts[task]) for task in TASKS},
            replay_mode=replay,
        ),
        evaluation=replace(config.evaluation, mode="sampled"),
        runtime=replace(
            config.runtime, output=str(run_dir), seed=seed, device="cpu"
        ),
    )


def make_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    experts = {}
    for task in TASKS:
        path = tmp_path / "experts" / task / "expert.zip"
        path.parent.mkdir(parents=True)
        path.write_bytes((task + " expert").encode())
        experts[task] = path
    config = config_for(run, experts)
    save_resolved_config(config, run)
    create_run_manifest(
        run,
        config,
        experts={
            task: {"checkpoint_sha256": file_hash(path)}
            for task, path in experts.items()
        },
    )
    return run, experts, config


def add_boundary(run, stage, *, diffusion=True):
    task = TASKS[stage]
    stage_dir = run / f"stage_{stage}_{task}"
    stage_dir.mkdir()
    policy = run / f"policy_{stage}.pt"
    policy.write_bytes(f"policy {stage}".encode())
    diffusion_path = run / f"diffusion_{stage}.pt" if diffusion else None
    if diffusion_path:
        diffusion_path.write_bytes(f"diffusion {stage}".encode())
    report = stage_dir / "report.json"
    report.write_text("{}\n")
    matrix = stage_dir / "evaluation_matrix.json"
    matrix.write_text("{}\n")
    return write_stage_boundary(
        run,
        stage=stage,
        task=task,
        policy_path=policy,
        diffusion_path=diffusion_path,
        report_path=report,
        matrix_path=matrix,
        training_banks={task: {"bank_seed": 0}},
        train_configurations={task: {"indices": [0]}},
    )


@pytest.mark.parametrize("count", [1, 2, 3])
def test_resume_after_one_multiple_or_all_completed_stages(tmp_path, count):
    run, _, _ = make_run(tmp_path)
    for stage in range(count):
        add_boundary(run, stage)
    markers = completed_stage_markers(run, TASKS)
    assert [marker["stage"] for marker in markers] == list(range(count))


def test_incomplete_stage_is_ignored_and_never_overwritten(tmp_path):
    run, _, _ = make_run(tmp_path)
    add_boundary(run, 0)
    partial = run / f"stage_1_{TASKS[1]}"
    partial.mkdir()
    evidence = partial / "partial.bin"
    evidence.write_bytes(b"keep me")
    assert len(completed_stage_markers(run, TASKS)) == 1
    assert evidence.read_bytes() == b"keep me"
    with pytest.raises(FileExistsError):
        add_boundary(run, 0)


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_missing_or_corrupted_completed_checkpoint_is_rejected(tmp_path, mode):
    run, _, _ = make_run(tmp_path)
    add_boundary(run, 0)
    policy = run / "policy_0.pt"
    if mode == "missing":
        policy.unlink()
        expected = FileNotFoundError
    else:
        policy.write_bytes(b"changed")
        expected = ValueError
    with pytest.raises(expected):
        completed_stage_markers(run, TASKS)


def test_configuration_and_replay_mismatches_are_rejected(tmp_path):
    run, experts, config = make_run(tmp_path)
    hashes = {task: file_hash(path) for task, path in experts.items()}
    validate_resume_configuration(run, config, hashes)
    changed_seed = replace(config, runtime=replace(config.runtime, seed=7))
    with pytest.raises(ValueError, match="scientific configuration"):
        validate_resume_configuration(run, changed_seed, hashes)
    changed_replay = replace(
        config, continual=replace(config.continual, replay_mode="none")
    )
    with pytest.raises(ValueError, match="scientific configuration"):
        validate_resume_configuration(run, changed_replay, hashes)


def test_expert_hashes_allow_relocation_but_reject_different_bytes(tmp_path):
    run, experts, config = make_run(tmp_path)
    relocated = {}
    for task, source in experts.items():
        target = tmp_path / "relocated" / task / "model.zip"
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        relocated[task] = target
    relocated_config = replace(
        config,
        continual=replace(
            config.continual,
            experts={task: str(path) for task, path in relocated.items()},
        ),
    )
    hashes = {task: file_hash(path) for task, path in relocated.items()}
    validate_resume_configuration(run, relocated_config, hashes)
    relocated[TASKS[0]].write_bytes(b"different expert")
    hashes[TASKS[0]] = file_hash(relocated[TASKS[0]])
    with pytest.raises(ValueError, match="expert checkpoint mismatch"):
        validate_resume_configuration(run, relocated_config, hashes)


def test_rng_state_is_restored_exactly(tmp_path):
    run, _, _ = make_run(tmp_path)
    add_boundary(run, 0)
    marker = completed_stage_markers(run, TASKS)[0]
    state = load_boundary_state(marker)
    restore_rng_state(state["rng"])
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)
    restore_rng_state(state["rng"])
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)


def test_evaluation_matrix_continuity(tmp_path):
    sequence = TaskSequence.from_names(TASKS)
    matrix = EvaluationMatrix(sequence)
    result = {
        "mean_return": 1.0,
        "std_return": 0.0,
        "success_rate": 1.0,
        "mean_episode_length": 2.0,
    }
    matrix.record(0, 0, result)
    path = tmp_path / "matrix.json"
    import json

    path.write_text(json.dumps(matrix.to_dict()))
    restored = EvaluationMatrix.load(path)
    assert restored.to_dict() == matrix.to_dict()
    with pytest.raises(ValueError, match="already measured"):
        restored.record(0, 0, result)


def test_run_with_all_stages_complete_is_a_noop(tmp_path):
    trainer = object.__new__(DiffCRLTrainer)
    trainer.sequence = TaskSequence.from_names(TASKS)
    trainer.next_stage = len(TASKS)
    trainer.output = tmp_path
    trainer.matrix = EvaluationMatrix(trainer.sequence)
    import json

    (tmp_path / "evaluation_matrix.json").write_text(
        json.dumps(trainer.matrix.to_dict()) + "\n"
    )
    (tmp_path / "RUN_COMPLETE").write_text("complete\n")
    with patch.object(trainer, "train_task") as train, patch(
        "src.continual.diffcrl.print_final_report"
    ), patch("src.continual.diffcrl.complete_run") as complete:
        assert trainer.run() is trainer.matrix
    train.assert_not_called()
    complete.assert_not_called()


def _advance(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 1e-4)


def test_deterministic_stage_boundary_matches_uninterrupted_execution(tmp_path):
    """Compare exact final model state for uninterrupted vs restored stage 1."""
    torch.manual_seed(12)
    policy = GeneralPolicy(-np.ones(4), np.ones(4), hidden_sizes=(4, 4))
    diffusion = TrajectoryDiffusion(
        DiffusionConfig(horizon=2, steps=2, width=4, num_tasks=2)
    )
    _advance(policy, 100)
    _advance(diffusion, 101)

    run = tmp_path / "integration"
    stage_dir = run / "stage_0_reach-v3"
    stage_dir.mkdir(parents=True)
    policy_path = run / "policy.pt"
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy.save(policy_path, optimizer=optimizer, metadata={"stage": 0})
    diffusion_path = run / "diffusion.pt"
    diffusion.save(
        diffusion_path,
        FeatureNormalizer(torch.zeros(22), torch.ones(22)),
        FeatureNormalizer(torch.zeros(3), torch.ones(3)),
        {
            "task_order": TASKS[:2],
            "generated_action_projection": "none",
            "training_banks": {"reach-v3": {}},
            "train_configurations": {"reach-v3": {}},
        },
    )
    report = stage_dir / "report.json"
    matrix = stage_dir / "evaluation_matrix.json"
    report.write_text("{}\n")
    matrix.write_text("{}\n")
    write_stage_boundary(
        run,
        stage=0,
        task="reach-v3",
        policy_path=policy_path,
        diffusion_path=diffusion_path,
        report_path=report,
        matrix_path=matrix,
        training_banks={"reach-v3": {}},
        train_configurations={"reach-v3": {}},
    )

    resumed_policy, _, metadata = GeneralPolicy.load_training(policy_path)
    resumed_diffusion, _, _, diffusion_metadata = TrajectoryDiffusion.load(
        diffusion_path,
        expected_task_order=TASKS[:2],
        expected_projection="none",
    )
    assert metadata["stage"] == 0
    assert diffusion_metadata["training_banks"] == {"reach-v3": {}}

    _advance(policy, 200)
    _advance(diffusion, 201)
    _advance(resumed_policy, 200)
    _advance(resumed_diffusion, 201)
    for uninterrupted, resumed in (
        (policy, resumed_policy),
        (diffusion, resumed_diffusion),
    ):
        for name, tensor in uninterrupted.state_dict().items():
            torch.testing.assert_close(
                tensor, resumed.state_dict()[name], rtol=0, atol=0
            )
