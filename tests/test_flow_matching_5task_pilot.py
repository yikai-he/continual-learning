from pathlib import Path

from src.config import load_config
from src.continual.replay_allocation import allocate_replay


ROOT = Path(__file__).parents[1]
PILOT = ROOT / "configs/diffcrl/flow_matching_5task_pilot.yaml"
FORMAL = ROOT / "configs/diffcrl/diffcrl_5task_4layer.yaml"


def test_pilot_matches_formal_diffusion_science_except_generator():
    pilot = load_config(PILOT, expected_experiment="diffcrl")
    formal = load_config(FORMAL, expected_experiment="diffcrl")
    assert pilot.continual.tasks == formal.continual.tasks
    assert pilot.continual.experts.keys() == formal.continual.experts.keys()
    assert all((ROOT / path).is_file() for path in pilot.continual.experts.values())
    assert pilot.continual.trajectories_per_task == formal.continual.trajectories_per_task == 200
    assert pilot.continual.replay_budget == 200
    assert pilot.continual.replay_priority_strategy == "uniform"
    assert pilot.bc == formal.bc
    assert pilot.evaluation == formal.evaluation
    assert pilot.environment == formal.environment
    assert pilot.runtime.seed == formal.runtime.seed == 0
    assert pilot.runtime.device == formal.runtime.device == "cpu"
    assert pilot.diffusion == formal.diffusion
    assert formal.continual.replay_mode == "diffusion"
    assert pilot.continual.replay_mode == "flow-matching"


def test_pilot_flow_settings_and_independent_output():
    pilot = load_config(PILOT, expected_experiment="diffcrl")
    assert pilot.flow_matching.epochs == 200
    assert pilot.flow_matching.integration_steps == 10
    assert pilot.flow_matching.width == 64
    assert pilot.flow_matching.probability_path == "linear"
    assert pilot.flow_matching.ode_solver == "euler"
    assert pilot.diffusion.action_space == "raw"
    assert pilot.diffusion.generated_action_projection == "none"
    assert pilot.runtime.run_name == "flow_matching_5task_fixed_budget_seed0"
    assert pilot.runtime.output == "runs/pilots/flow_matching_5task_fixed_budget"


def test_fixed_budget_allocations_for_every_five_task_stage():
    expected = (
        {},
        {0: 200},
        {0: 100, 1: 100},
        {0: 67, 1: 67, 2: 66},
        {0: 50, 1: 50, 2: 50, 3: 50},
    )
    for stage, allocation in enumerate(expected):
        assert allocate_replay(range(stage), 200 if stage else 0) == allocation


def test_all_expert_assets_exist():
    pilot = load_config(PILOT, expected_experiment="diffcrl")
    assert all((ROOT / path).is_file() for path in pilot.continual.experts.values())
