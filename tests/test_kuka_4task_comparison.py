from pathlib import Path

from src.config import config_to_dict, load_config
from src.continual.replay_allocation import replay_counts
from src.support.expert_manifest import validate_expert_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DDPM_CONFIG = ROOT / "configs/diffcrl/kuka_4task_ddpm_comparison.yaml"
FLOW_CONFIG = ROOT / "configs/diffcrl/kuka_4task_flow_matching_comparison.yaml"
DDPM_15PLUS50_CONFIG = ROOT / "configs/diffcrl/kuka_4task_ddpm_15plus50.yaml"
FLOW_15PLUS50_CONFIG = ROOT / "configs/diffcrl/kuka_4task_flow_matching_15plus50.yaml"


def _differences(left, right, prefix=""):
    differences = set()
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            differences.update(_differences(left.get(key), right.get(key), path))
    elif left != right:
        differences.add(prefix)
    return differences


def test_kuka_comparison_configs_are_matched_and_experts_are_qualified():
    ddpm = load_config(DDPM_CONFIG, expected_experiment="diffcrl")
    flow = load_config(FLOW_CONFIG, expected_experiment="diffcrl")
    assert ddpm.environment == flow.environment
    assert ddpm.bc == flow.bc
    assert ddpm.diffusion == flow.diffusion
    assert ddpm.continual.tasks == flow.continual.tasks
    assert ddpm.continual.experts == flow.continual.experts
    assert ddpm.continual.trajectories_per_task == 15
    assert ddpm.continual.replay_allocation == "fixed-total"
    assert ddpm.continual.replay_budget == flow.continual.replay_budget == 15
    assert ddpm.evaluation == flow.evaluation
    assert ddpm.runtime.seed == flow.runtime.seed == 0
    assert ddpm.runtime.output != flow.runtime.output
    assert ddpm.diffusion.steps == 100
    assert flow.flow_matching.integration_steps == 10

    for task, checkpoint in ddpm.continual.experts.items():
        identity = validate_expert_checkpoint(
            ROOT / checkpoint,
            expected_task=task,
            expected_backend="kuka-v3",
            expected_reward_function_version="v2",
            expected_observation_shape=(39,),
            expected_action_shape=(4,),
            expected_horizon=200,
            allow_legacy=False,
        )
        assert identity.qualification["success_rate"] == 1.0
        assert identity.training_seed == 0


def test_15plus50_configs_change_only_budget_and_output():
    pairs = (
        (DDPM_CONFIG, DDPM_15PLUS50_CONFIG, "diffusion", "ddpm"),
        (FLOW_CONFIG, FLOW_15PLUS50_CONFIG, "flow-matching", "flow_matching"),
    )
    for reference_path, new_path, replay_mode, output_leaf in pairs:
        reference = load_config(reference_path, expected_experiment="diffcrl")
        configured = load_config(new_path, expected_experiment="diffcrl")
        assert _differences(config_to_dict(reference), config_to_dict(configured)) == {
            "continual.replay_budget",
            "runtime.output",
        }
        assert configured.continual.trajectories_per_task == 15
        assert configured.continual.replay_budget == 50
        assert configured.continual.replay_mode == replay_mode
        assert configured.runtime.output == (
            f"runs/kuka_4task_generative_comparison_15plus50/{output_leaf}"
        )
        expected = ({}, {0: 50}, {0: 25, 1: 25}, {0: 17, 1: 17, 2: 16})
        actual = tuple(
            {} if stage == 0 else replay_counts(
                task_ids=range(stage),
                allocation="fixed-total",
                trajectories_per_task=15,
                total_budget=50,
                strategy="uniform",
                generator=replay_mode,
            )
            for stage in range(4)
        )
        assert actual == expected

    flow = load_config(FLOW_15PLUS50_CONFIG, expected_experiment="diffcrl")
    assert flow.flow_matching.integration_steps == 10
