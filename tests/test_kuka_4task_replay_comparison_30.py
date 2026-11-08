from pathlib import Path
from unittest.mock import patch

from src.config import REPLAY_PRIORITY_STRATEGY, load_config
from src.continual.diffcrl import DiffCRLTrainer, StageContext
from src.continual.replay_allocation import replay_counts
from src.continual.task_sequence import TaskSequence
from src.support.expert_manifest import validate_expert_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "none": ROOT / "configs/diffcrl/kuka_4task_no_replay_30.yaml",
    "real": ROOT / "configs/diffcrl/kuka_4task_real_replay_30plus30.yaml",
    "diffusion": ROOT / "configs/diffcrl/kuka_4task_ddpm_30plus30.yaml",
    "flow-matching": ROOT / "configs/diffcrl/kuka_4task_flow_matching_30plus30.yaml",
}
TASKS = [
    "kuka-drawer-close-v3",
    "kuka-window-close-v3",
    "kuka-door-open-v3",
    "kuka-peg-unplug-side-v3",
]


def test_replay_comparison_configs_are_matched_qualified_and_isolated():
    loaded = {
        mode: load_config(path, expected_experiment="diffcrl")
        for mode, path in CONFIGS.items()
    }
    reference = loaded["diffusion"]
    expected_outputs = {
        "none": "no_replay",
        "real": "real_replay",
        "diffusion": "ddpm",
        "flow-matching": "flow_matching",
    }
    for mode, config in loaded.items():
        assert config.environment == reference.environment
        assert config.bc == reference.bc
        assert config.diffusion == reference.diffusion
        assert config.continual.tasks == TASKS
        assert config.continual.experts == reference.continual.experts
        assert config.continual.trajectories_per_task == 30
        assert config.continual.replay_mode == mode
        assert config.continual.replay_allocation == "fixed-total"
        assert config.continual.replay_budget == 30
        assert REPLAY_PRIORITY_STRATEGY == "uniform"
        assert config.evaluation == reference.evaluation
        assert config.runtime.seed == 0
        assert config.runtime.device == "cpu"
        assert config.runtime.output.endswith(expected_outputs[mode])
    assert loaded["diffusion"].diffusion.steps == 100
    assert loaded["flow-matching"].flow_matching.integration_steps == 10
    for task, checkpoint in reference.continual.experts.items():
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


def test_replay_allocations_match_and_no_replay_is_effectively_zero(tmp_path):
    expected = ({0: 30}, {0: 15, 1: 15}, {0: 10, 1: 10, 2: 10})
    for mode in ("real", "diffusion", "flow-matching"):
        config = load_config(CONFIGS[mode], expected_experiment="diffcrl")
        actual = tuple(
            replay_counts(
                range(stage),
                generator="diffusion" if mode == "real" else mode,
                allocation=config.continual.replay_allocation,
                trajectories_per_task=30,
                total_budget=30,
                strategy="uniform",
            )
            for stage in range(1, 4)
        )
        assert actual == expected

    config = load_config(CONFIGS["none"], expected_experiment="diffcrl")
    trainer = object.__new__(DiffCRLTrainer)
    trainer.config = config
    trainer.sequence = TaskSequence.from_names(TASKS)
    context = StageContext(
        index=3,
        task=trainer.sequence.task(3),
        trajectory_count=30,
        directory=tmp_path,
        policy_before_hash=None,
        diffusion_before_hash=None,
    )
    with patch("src.continual.diffcrl.replay_counts") as allocator:
        replay = trainer._generate_replay(context)
    assert replay.groups == {}
    assert replay.sources == {}
    allocator.assert_not_called()


def test_current_task_collection_protocol_is_identical_across_methods():
    configs = [load_config(path, expected_experiment="diffcrl") for path in CONFIGS.values()]
    signatures = {
        (
            tuple(config.continual.tasks),
            tuple(config.continual.experts.items()),
            config.continual.trajectories_per_task,
            config.runtime.seed,
            config.runtime.device,
            config.environment,
        )
        for config in configs
    }
    assert len(signatures) == 1
    assert list(range(configs[0].runtime.seed, configs[0].runtime.seed + 30)) == list(range(30))
