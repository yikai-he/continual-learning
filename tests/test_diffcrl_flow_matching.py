"""Focused Phase 4 trainer integration tests for Flow Matching replay."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch import nn

from src.config import default_config
from src.continual.bc_policy import GeneralPolicy
from src.continual.diffcrl import (
    CurrentTaskData,
    DiffCRLTrainer,
    DiffusionTrainingResult,
    PolicyTrainingResult,
    StageArtifacts,
    StageContext,
    StageData,
    StageRawData,
)
from src.continual.evaluation import EvaluationMatrix
from src.continual.resume import write_stage_boundary
from src.continual.task_sequence import TaskSequence
from src.continual.trajectory_diffusion import (
    FeatureNormalizer,
    TrajectoryDiffusion,
)
from src.support.reproducibility import state_hash


TASKS = ["reach-v3", "push-v3"]


def trainer_config(tmp_path, mode):
    base = default_config("diffcrl")
    experts = {}
    for task in TASKS:
        path = tmp_path / f"{task}.zip"
        path.write_bytes(task.encode())
        experts[task] = str(path)
    return replace(
        base,
        continual=replace(
            base.continual,
            tasks=list(TASKS),
            experts=experts,
            replay_mode=mode,
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
        diffusion=replace(
            base.diffusion,
            epochs=1,
            batch_size=1,
            steps=2,
            width=4,
        ),
        runtime=replace(
            base.runtime, output=str(tmp_path / f"run-{mode}"), device="cpu"
        ),
    )


def initialize(tmp_path, mode, *, resume_from=None):
    identity = SimpleNamespace(to_dict=lambda: {"checkpoint_sha256": "expert"})
    with patch("src.continual.diffcrl.validate_expert_checkpoint", return_value=identity), patch(
        "src.continual.diffcrl.file_hash", return_value="expert-hash"
    ), patch("src.continual.diffcrl.create_run_manifest"), patch(
        "src.continual.diffcrl.write_json"
    ):
        return DiffCRLTrainer(
            trainer_config(tmp_path, mode), progress=False, resume_from=resume_from
        )


@pytest.mark.parametrize(
    "mode,has_diffusion,has_flow",
    [("diffusion", True, False), ("flow-matching", False, True), ("none", False, False)],
)
def test_generator_initialization_is_mutually_exclusive(
    tmp_path, mode, has_diffusion, has_flow
):
    trainer = initialize(tmp_path, mode)
    assert (trainer.diffusion is not None) is has_diffusion
    assert (trainer.flow_matching is not None) is has_flow
    if has_flow:
        assert trainer.flow_matching.config.horizon == trainer.horizon
        assert trainer.flow_matching.config.num_tasks == len(TASKS)
        assert trainer.flow_matching.config.integration_steps == 2
        assert trainer.flow_matching.config.width == 4


def bare_flow_trainer(tmp_path):
    trainer = object.__new__(DiffCRLTrainer)
    trainer.config = trainer_config(tmp_path, "flow-matching")
    trainer.sequence = TaskSequence.from_names(TASKS)
    trainer.horizon = 200
    trainer.progress = False
    trainer.diffusion = None
    trainer.flow_matching = nn.Linear(1, 1)
    trainer.previous_diffusion = None
    trainer.previous_flow_matching = tmp_path / "previous-flow.pt"
    trainer.output = tmp_path
    trainer.training_banks = {"reach-v3": {"ordered_task_hashes": ["a"]}}
    trainer.train_configurations = {"reach-v3": {"indices": [0]}}
    trainer._progress_message = Mock()
    return trainer


def context(tmp_path, stage=1):
    return StageContext(
        index=stage,
        task=TaskSequence.from_names(TASKS).task(stage),
        trajectory_count=2,
        directory=tmp_path,
        policy_before_hash=None,
        diffusion_before_hash=None,
        flow_matching_before_hash="flow-state",
    )


def test_replay_dispatches_only_to_flow_adapter(tmp_path):
    trainer = bare_flow_trainer(tmp_path)
    generated = SimpleNamespace(
        groups={0: torch.zeros(2, 200, 43)},
        sources={"reach-v3": {"kind": "generated"}},
    )
    with patch(
        "src.continual.diffcrl.generate_previous_task_flow_matching_replay",
        return_value=generated,
    ) as flow_replay, patch(
        "src.continual.diffcrl.generate_previous_task_replay"
    ) as diffusion_replay:
        result = trainer._generate_replay(context(tmp_path))

    diffusion_replay.assert_not_called()
    flow_replay.assert_called_once()
    request = flow_replay.call_args.args[0]
    assert request.checkpoint == trainer.previous_flow_matching
    assert request.expected_state_hash == "flow-state"
    assert request.trajectory_counts == {0: 200}
    assert request.integration_steps == 2
    assert result.groups is generated.groups


def physical_stage_data():
    current = torch.randn(4, 200, 18)
    previous = torch.cat((current[:, :1], current[:, :-1]), dim=1)
    goals = torch.randn(4, 1, 3).expand(-1, 200, -1).clone()
    actions = torch.tanh(torch.randn(4, 200, 4))
    values = torch.cat((current, previous, goals, actions), dim=-1)
    return StageData(
        train=values[:2],
        validation=values[2:],
        train_labels=torch.tensor([0, 1]),
        validation_labels=torch.tensor([0, 1]),
        sources={},
        splits={},
        real_encode_clip_fraction=0.0,
    )


def test_flow_training_reuses_normalized_stage_data_and_persistent_weights(tmp_path):
    trainer = bare_flow_trainer(tmp_path)
    trainer.policy = nn.Linear(1, 1)
    data = physical_stage_data()
    seen_models = []

    def fit(model, train, validation, train_goals, validation_goals, **kwargs):
        seen_models.append(model)
        assert train.shape == (2, 200, 22)
        assert validation.shape == (2, 200, 22)
        assert train_goals.shape == (2, 3)
        assert kwargs["seed"] in (0, 1)
        with torch.no_grad():
            model.weight.add_(1)
        return {"best_epoch": 1}

    with patch("src.continual.diffcrl.fit_flow_matching", side_effect=fit), patch(
        "src.continual.diffcrl.state_hash", return_value="policy-state"
    ):
        first = trainer._fit_stage_diffusion(context(tmp_path, stage=0), data)
        weight_after_first = trainer.flow_matching.weight.detach().clone()
        second = trainer._fit_stage_diffusion(context(tmp_path, stage=1), data)

    assert seen_models == [trainer.flow_matching, trainer.flow_matching]
    torch.testing.assert_close(
        trainer.flow_matching.weight, weight_after_first + 1, rtol=0, atol=0
    )
    assert first.losses == second.losses == {"best_epoch": 1}
    assert first.normalizer.mean.shape == (22,)


@pytest.mark.parametrize("mode", ["diffusion", "flow-matching"])
def test_generator_normalization_remains_inside_measured_interval(tmp_path, mode):
    trainer = bare_flow_trainer(tmp_path)
    trainer.config = trainer_config(tmp_path, mode)
    trainer.policy = nn.Linear(1, 1)
    trainer.diffusion = nn.Linear(1, 1) if mode == "diffusion" else None
    trainer.flow_matching = nn.Linear(1, 1) if mode == "flow-matching" else None
    normalizer = FeatureNormalizer(torch.zeros(22), torch.ones(22))
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3))
    prepared = SimpleNamespace(normalizer=normalizer, goal_normalizer=goal)
    normalized = SimpleNamespace(
        train=torch.zeros(1, 200, 22),
        validation=torch.zeros(1, 200, 22),
        train_goals=torch.zeros(1, 3),
        validation_goals=torch.zeros(1, 3),
    )
    events = []

    def normalize(_prepared):
        events.append("normalize")
        return normalized

    def fit(*_args, **_kwargs):
        events.append("fit")
        return {"best_epoch": 1}

    times = iter((10.0, 15.0))

    def clock():
        events.append("clock")
        return next(times)

    fit_target = (
        "src.continual.diffcrl.fit_diffusion"
        if mode == "diffusion"
        else "src.continual.diffcrl.fit_flow_matching"
    )
    with patch(
        "src.continual.diffcrl.prepare_generator_data", return_value=prepared
    ), patch(
        "src.continual.diffcrl.normalize_generator_data", side_effect=normalize
    ), patch(fit_target, side_effect=fit), patch(
        "src.continual.diffcrl.state_hash", return_value="policy-state"
    ), patch(
        "src.continual.diffcrl.time.perf_counter", side_effect=clock
    ):
        result = trainer._fit_stage_diffusion(context(tmp_path, stage=0), physical_stage_data())

    assert events == ["clock", "normalize", "fit", "clock"]
    assert result.training_seconds == 5.0


@pytest.mark.parametrize("mode", ["real", "none"])
def test_generator_free_modes_report_zero_training_seconds(tmp_path, mode):
    trainer = bare_flow_trainer(tmp_path)
    trainer.config = trainer_config(tmp_path, mode)
    trainer.policy = nn.Linear(1, 1)
    trainer.diffusion = None
    trainer.flow_matching = None
    with patch("src.continual.diffcrl.normalize_generator_data") as normalize:
        result = trainer._fit_stage_diffusion(context(tmp_path, stage=0), physical_stage_data())
    normalize.assert_not_called()
    assert result.losses is None
    assert result.training_seconds == 0.0


def test_flow_checkpoint_filename_metadata_and_artifact_key(tmp_path):
    trainer = bare_flow_trainer(tmp_path)
    trainer.policy = Mock()
    trainer.policy.state_dict.return_value = {"weight": torch.ones(1)}
    trainer.policy_optimizer = Mock()
    saved = {}

    def save_flow(path, dynamic, goal, metadata):
        Path(path).write_bytes(b"flow")
        saved.update(path=Path(path), dynamic=dynamic, goal=goal, metadata=metadata)

    trainer.flow_matching = Mock()
    trainer.flow_matching.state_dict.return_value = {"weight": torch.ones(1)}
    trainer.flow_matching.save.side_effect = save_flow
    trainer.policy.save.side_effect = lambda path, **_: Path(path).write_bytes(b"policy")
    normalizer = FeatureNormalizer(torch.zeros(22), torch.ones(22))
    goal = FeatureNormalizer(torch.zeros(3), torch.ones(3))
    training = DiffusionTrainingResult(normalizer, goal, {"best_epoch": 1})
    policy = PolicyTrainingResult({"best_epoch": 1})

    artifacts = trainer._save_stage_checkpoints(context(tmp_path), training, policy)

    assert artifacts.diffusion_path is None
    assert artifacts.flow_matching_path.name.startswith("flow_matching_after_task_1_")
    assert saved["path"] == artifacts.flow_matching_path
    required = {
        "task_order",
        "training_banks",
        "train_configurations",
        "generated_action_projection",
        "action_space",
        "action_clamp_epsilon",
    }
    assert required <= set(saved["metadata"])
    assert saved["metadata"]["action_space"] == trainer.config.diffusion.action_space


def test_ddpm_and_policy_checkpoint_metadata_optimizer_and_load_compatibility(tmp_path):
    trainer = initialize(tmp_path, "diffusion")
    trainer.output = tmp_path
    trainer.policy = GeneralPolicy(-np.ones(4), np.ones(4), hidden_sizes=(8, 8))
    trainer.policy_optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=1e-3)
    loss = trainer.policy(torch.zeros(2, 39)).square().sum()
    loss.backward()
    trainer.policy_optimizer.step()
    trainer.training_banks = {"reach-v3": {"ordered_task_hashes": ["task-hash"]}}
    trainer.train_configurations = {
        "reach-v3": {"indices": [0], "source_trajectory_ids": [0]}
    }
    dynamic = FeatureNormalizer(torch.arange(22.0), torch.arange(1.0, 23.0))
    goal = FeatureNormalizer(torch.tensor([1.0, 2.0, 3.0]), torch.ones(3))
    policy_training = PolicyTrainingResult({"best_epoch": 1})

    artifacts = trainer._save_stage_checkpoints(
        context(tmp_path),
        DiffusionTrainingResult(dynamic, goal, {"best_epoch": 1}),
        policy_training,
    )

    loaded_diffusion, loaded_dynamic, loaded_goal, metadata = TrajectoryDiffusion.load(
        artifacts.diffusion_path,
        expected_config=trainer.diffusion.config,
        expected_action_space=trainer.config.diffusion.action_space,
        expected_clamp_epsilon=trainer.config.diffusion.action_clamp_epsilon,
        expected_projection=trainer.config.diffusion.generated_action_projection,
        expected_task_order=trainer.sequence.task_names,
    )
    assert state_hash(loaded_diffusion) == state_hash(trainer.diffusion)
    torch.testing.assert_close(loaded_dynamic.mean, dynamic.mean, rtol=0, atol=0)
    torch.testing.assert_close(loaded_goal.scale, goal.scale, rtol=0, atol=0)
    assert metadata["task_order"] == trainer.sequence.task_names
    assert metadata["training_banks"] == trainer.training_banks
    assert metadata["train_configurations"] == trainer.train_configurations
    assert metadata["generated_action_projection"] == (
        trainer.config.diffusion.generated_action_projection
    )
    assert metadata["diffusion_action_space"] == trainer.config.diffusion.action_space

    loaded_policy, loaded_optimizer, policy_metadata = GeneralPolicy.load_training(
        artifacts.policy_path
    )
    assert state_hash(loaded_policy) == state_hash(trainer.policy)
    assert loaded_optimizer is not None
    loaded_optimizer_state = loaded_optimizer.state_dict()
    expected_optimizer_state = trainer.policy_optimizer.state_dict()
    assert loaded_optimizer_state["state"]
    assert loaded_optimizer_state["param_groups"] == expected_optimizer_state["param_groups"]
    assert set(loaded_optimizer_state["state"]) == set(expected_optimizer_state["state"])
    for parameter, loaded_values in loaded_optimizer_state["state"].items():
        for name, loaded_value in loaded_values.items():
            expected_value = expected_optimizer_state["state"][parameter][name]
            if isinstance(loaded_value, torch.Tensor):
                torch.testing.assert_close(loaded_value, expected_value, rtol=0, atol=0)
            else:
                assert loaded_value == expected_value
    assert policy_metadata == {
        "stage": 1,
        "task": "push-v3",
        "tasks": trainer.sequence.task_names,
        "seed": trainer.config.runtime.seed,
        "best_epoch": 1,
        "bc_loss": trainer.config.bc.loss,
        "bc_clamp_epsilon": 1e-4,
        "bc_normalization_mode": trainer.config.bc.normalization_mode,
        "bc_use_layer_norm": trainer.config.bc.use_layer_norm,
        "diffusion_action_space": trainer.config.diffusion.action_space,
        "diffusion_action_clamp_epsilon": trainer.config.diffusion.action_clamp_epsilon,
        "generated_action_projection": trainer.config.diffusion.generated_action_projection,
    }


def test_stage_boundary_uses_distinct_flow_artifact_key(tmp_path):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    paths = {
        "policy_path": tmp_path / "policy.pt",
        "flow_matching_path": tmp_path / "flow.pt",
        "report_path": stage_dir / "report.json",
        "matrix_path": stage_dir / "matrix.json",
    }
    for path in paths.values():
        path.write_bytes(b"artifact")
    marker_path = write_stage_boundary(
        tmp_path,
        stage=0,
        task="reach-v3",
        diffusion_path=None,
        training_banks={},
        train_configurations={},
        **paths,
    )
    marker = json.loads(marker_path.read_text())
    assert "flow_matching" in marker["artifacts"]
    assert "diffusion" not in marker["artifacts"]


def test_train_task_keeps_collection_bc_and_evaluation_shared(tmp_path):
    trainer = bare_flow_trainer(tmp_path)
    trainer.next_stage = 0
    trainer.policy = None
    trainer.matrix = EvaluationMatrix(trainer.sequence)
    trainer.experts = {}
    trainer.expert_hashes = {}
    trainer.stage_complete_callback = None
    current = CurrentTaskData(
        group=torch.zeros(2, 200, 43),
        source={"kind": "real_expert"},
        collected_configurations=[],
        real_encode_clip_fraction=0.0,
    )
    prepared = physical_stage_data()
    generator_training = DiffusionTrainingResult(
        FeatureNormalizer(torch.zeros(22), torch.ones(22)),
        FeatureNormalizer(torch.zeros(3), torch.ones(3)),
        {"best_epoch": 1},
    )
    policy_training = PolicyTrainingResult({"best_epoch": 1})
    artifacts = StageArtifacts(None, tmp_path / "policy.pt", tmp_path / "flow.pt")
    events = []

    def step(name, value=None):
        def invoke(*_args, **_kwargs):
            events.append(name)
            return value

        return invoke

    with patch.object(
        trainer,
        "_generate_replay",
        side_effect=step("replay", StageRawData({}, {})),
    ), patch.object(
        trainer, "_collect_current_task", side_effect=step("collect", current)
    ) as collect, patch.object(
        trainer, "_prepare_stage_data", side_effect=step("prepare", prepared)
    ), patch.object(
        trainer,
        "_fit_stage_diffusion",
        side_effect=step("generator", generator_training),
    ) as fit_generator, patch.object(
        trainer, "_fit_stage_policy", side_effect=step("bc", policy_training)
    ) as fit_policy, patch.object(
        trainer, "_save_stage_checkpoints", side_effect=step("checkpoints", artifacts)
    ), patch.object(
        trainer, "_evaluate_stage", side_effect=step("evaluate")
    ) as evaluate, patch.object(
        trainer, "_finish_stage", side_effect=step("finalize", {"stage": 0})
    ):
        report = trainer.train_task(0)

    collect.assert_called_once()
    fit_generator.assert_called_once()
    fit_policy.assert_called_once()
    assert fit_generator.call_args.args[1] is prepared
    assert fit_policy.call_args.args[1] is prepared
    evaluate.assert_called_once_with(0)
    assert events == [
        "replay",
        "collect",
        "prepare",
        "generator",
        "bc",
        "checkpoints",
        "evaluate",
        "finalize",
    ]
    assert report == {"stage": 0}
