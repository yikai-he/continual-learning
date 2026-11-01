"""Preflight for the Hammer-first 100-real/100-pseudo Flow pilot."""

from dataclasses import replace
import inspect
from pathlib import Path

from src.config import config_to_dict, load_config, output_directory
from src.continual.diffcrl import DiffCRLTrainer, StageContext
from src.continual.replay_allocation import allocate_replay
from src.continual.task_sequence import TaskSequence


ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "configs/diffcrl/flow_matching_5task_hammer_first.yaml"
PILOT = ROOT / "configs/diffcrl/flow_matching_5task_hammer_first_100real_100pseudo.yaml"
TASKS = [
    "hammer-v3",
    "button-press-v3",
    "faucet-close-v3",
    "handle-press-side-v3",
    "window-close-v3",
]


def test_only_budgets_and_runtime_identity_differ_from_hammer_first_reference():
    reference = load_config(REFERENCE, expected_experiment="diffcrl")
    pilot = load_config(PILOT, expected_experiment="diffcrl")
    assert pilot.continual.tasks == TASKS
    assert pilot.continual.trajectories_per_task == 100
    assert pilot.continual.replay_budget == 100
    normalized = replace(
        pilot,
        continual=replace(
            pilot.continual,
            trajectories_per_task=reference.continual.trajectories_per_task,
            replay_budget=reference.continual.replay_budget,
        ),
        runtime=replace(
            pilot.runtime,
            output=reference.runtime.output,
            run_name=reference.runtime.run_name,
        ),
    )
    assert config_to_dict(normalized) == config_to_dict(reference)


def test_exact_stage_allocations_and_combined_counts():
    expected = (
        {},
        {"hammer-v3": 100},
        {"hammer-v3": 50, "button-press-v3": 50},
        {"hammer-v3": 34, "button-press-v3": 33, "faucet-close-v3": 33},
        {
            "hammer-v3": 25,
            "button-press-v3": 25,
            "faucet-close-v3": 25,
            "handle-press-side-v3": 25,
        },
    )
    pilot = load_config(PILOT, expected_experiment="diffcrl")
    for stage, expected_allocation in enumerate(expected):
        allocation = allocate_replay(range(stage), 100 if stage else 0)
        named = {pilot.continual.tasks[index]: count for index, count in allocation.items()}
        assert named == expected_allocation
        assert pilot.continual.trajectories_per_task + sum(allocation.values()) == (
            100 if stage == 0 else 200
        )


def test_stage_zero_is_real_only_and_experts_and_output_are_isolated(tmp_path):
    pilot = load_config(PILOT, expected_experiment="diffcrl")
    assert all((ROOT / path).is_file() for path in pilot.continual.experts.values())
    assert pilot.continual.replay_mode == "flow-matching"
    assert pilot.continual.replay_priority_strategy == "uniform"
    assert output_directory(pilot) != output_directory(
        load_config(REFERENCE, expected_experiment="diffcrl")
    )
    trainer = object.__new__(DiffCRLTrainer)
    replay = trainer._generate_replay(
        StageContext(
            index=0,
            task=TaskSequence.from_names(TASKS).task(0),
            trajectory_count=100,
            directory=tmp_path,
            policy_before_hash=None,
            diffusion_before_hash=None,
        )
    )
    assert replay.groups == {}
    assert replay.sources == {}


def test_flow_and_bc_share_the_same_combined_stage_data_contract():
    # The orchestrator passes the one prepared StageData object to both fits.
    source = inspect.getsource(DiffCRLTrainer.train_task)
    assert "_fit_stage_diffusion(context, data)" in source
    assert "_fit_stage_policy(context, data)" in source
