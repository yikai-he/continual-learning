"""Contracts for the prepared independent Sawyer SAC continuations."""

from pathlib import Path

from src.config import load_config, output_directory


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("peg_unplug_side.yaml", "peg-unplug-side-v3", 300_000),
    ("stick_pull.yaml", "stick-pull-v3", 500_000),
    ("push.yaml", "push-v3", 1_000_000),
    ("push_wall.yaml", "push-wall-v3", 1_000_000),
)


def test_continuation_configs_match_protocol_and_are_collision_free():
    for filename, task, budget in CASES:
        config = load_config(
            ROOT / "configs/sac/continuation/sawyer" / filename,
            expected_experiment="sac",
        )
        assert config.continual.tasks == [task]
        assert config.environment.backend == "metaworld-v3"
        assert config.environment.reward_function_version == "v2"
        assert config.runtime.seed == 0
        assert config.sac.steps == budget
        assert config.resume.enabled
        assert config.resume.replay_buffer is None
        assert config.resume.refill_steps == 10_000
        assert Path(config.resume.checkpoint).is_file()
        assert config.actor_init.mode == "scratch"
        assert config.actor_init.checkpoint is None
        stop = config.sac.early_stopping
        assert (stop.min_steps, stop.success_threshold, stop.patience) == (50_000, 1.0, 3)
        assert stop.plateau_patience is None
        assert config.evaluation.episodes == 50
        assert config.evaluation.final_episodes == 100
        assert config.evaluation.frequency == 20_000
        assert config.evaluation.task_set_seed == 10_000
        assert config.evaluation.final_task_set_seeds == [20_000]
        assert output_directory(config) == (
            ROOT / "runs/sac_continuation/sawyer" / task / "seed_0"
        )
        assert not output_directory(config).exists()


def test_launcher_order_and_explicit_execution_guard():
    launcher = (ROOT / "tools/experiments/train_sawyer_continuation.sh").read_text()
    order = [launcher.index(task) for _, task, _ in CASES]
    assert order == sorted(order)
    assert '--dry-run|--preflight-only' in launcher
    assert '--run' in launcher
    assert 'Usage: $0 --dry-run|--preflight-only|--run' in launcher
