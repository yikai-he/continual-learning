"""Read-only preflight contracts for the Hammer-first Flow experiment."""

from dataclasses import replace
import inspect
from pathlib import Path

from src.config import config_to_dict, load_config, output_directory
from src.continual.replay_allocation import allocate_replay
from src.continual.diffcrl import DiffCRLTrainer, StageContext
from src.continual.task_sequence import TaskSequence


ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "configs/diffcrl/flow_matching_5task_pilot.yaml"
HAMMER_FIRST = ROOT / "configs/diffcrl/flow_matching_5task_hammer_first.yaml"
TASKS = [
    "hammer-v3",
    "button-press-v3",
    "faucet-close-v3",
    "handle-press-side-v3",
    "window-close-v3",
]


def test_only_task_order_and_runtime_identity_differ_from_reference():
    reference = load_config(REFERENCE, expected_experiment="diffcrl")
    experiment = load_config(HAMMER_FIRST, expected_experiment="diffcrl")
    assert experiment.continual.tasks == TASKS
    assert experiment.continual.experts == reference.continual.experts
    normalized = replace(
        experiment,
        continual=replace(experiment.continual, tasks=reference.continual.tasks),
        runtime=replace(
            experiment.runtime,
            output=reference.runtime.output,
            run_name=reference.runtime.run_name,
        ),
    )
    assert config_to_dict(normalized) == config_to_dict(reference)


def test_hammer_first_allocation_and_evaluation_contract():
    experiment = load_config(HAMMER_FIRST, expected_experiment="diffcrl")
    expected = (
        {},
        {"hammer-v3": 200},
        {"hammer-v3": 100, "button-press-v3": 100},
        {"hammer-v3": 67, "button-press-v3": 67, "faucet-close-v3": 66},
        {
            "hammer-v3": 50,
            "button-press-v3": 50,
            "faucet-close-v3": 50,
            "handle-press-side-v3": 50,
        },
    )
    for stage, expected_allocation in enumerate(expected):
        allocation = allocate_replay(range(stage), 200 if stage else 0)
        assert {
            experiment.continual.tasks[index]: count
            for index, count in allocation.items()
        } == expected_allocation
    assert experiment.evaluation.episodes == 50
    assert experiment.evaluation.mode == "fixed-tasks"
    assert experiment.evaluation.seed == 10_000
    assert experiment.evaluation.task_set_seed == 10_000


def test_experts_and_unique_output_identity():
    reference = load_config(REFERENCE, expected_experiment="diffcrl")
    experiment = load_config(HAMMER_FIRST, expected_experiment="diffcrl")
    assert all((ROOT / path).is_file() for path in experiment.continual.experts.values())
    assert output_directory(experiment) != output_directory(reference)
    assert experiment.runtime.output == "runs/pilots/flow_matching_5task_hammer_first"
    assert experiment.runtime.run_name == "flow_matching_5task_hammer_first_seed0"


def test_stage_zero_has_no_replay_and_uses_scratch_policy_branch(tmp_path):
    trainer = object.__new__(DiffCRLTrainer)
    task = TaskSequence.from_names(TASKS).task(0)
    replay = trainer._generate_replay(
        StageContext(
            index=0,
            task=task,
            trajectory_count=200,
            directory=tmp_path,
            policy_before_hash=None,
            diffusion_before_hash=None,
        )
    )
    assert replay.groups == {}
    assert replay.sources == {}
    source = inspect.getsource(DiffCRLTrainer._collect_current_task)
    assert "if self.policy is None:" in source
    assert "self.policy = GeneralPolicy(" in source
