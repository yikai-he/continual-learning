"""Contracts for the three prepared Sawyer SAC Early Stopping runs."""

from pathlib import Path
from unittest.mock import MagicMock

from src.config import load_config, output_directory
from src.support.reproducibility import file_hash
from src.support.expert_manifest import load_expert_manifest
from train_sac import FixedSeedEvalCallback


ROOT = Path(__file__).resolve().parents[1]


CASES = (
    ("sawyer_push_back_from_faucet_close_early_stop.yaml", "push-back-v3", 1_000_000, 200_000, 8, "sac_actor"),
    ("sawyer_stick_pull_continuation_early_stop.yaml", "stick-pull-v3", 500_000, 120_000, 6, "resume"),
    ("sawyer_push_continuation_early_stop.yaml", "push-v3", 1_000_000, 400_000, 10, "resume"),
)


def test_prepared_configs_are_valid_isolated_and_hash_bound():
    outputs = set()
    for filename, task, budget, minimum, plateau, mode in CASES:
        config = load_config(ROOT / "configs/sac" / filename, expected_experiment="sac")
        assert config.continual.tasks == [task]
        assert config.sac.steps == budget
        assert config.sac.early_stopping.min_steps == minimum
        assert config.sac.early_stopping.success_threshold == 1.0
        assert config.sac.early_stopping.patience == 3
        assert config.sac.early_stopping.plateau_patience == plateau
        assert config.sac.early_stopping.min_delta == 0.02
        assert config.evaluation.episodes == 100
        assert config.evaluation.frequency == 40_000
        assert config.evaluation.final_task_set_seeds == [20_000, 20_001]
        source_name = config.resume.checkpoint if mode == "resume" else config.actor_init.checkpoint
        source = ROOT / source_name
        manifest = load_expert_manifest(source.parent / "manifest.json")
        assert file_hash(source) == manifest["checkpoint_sha256"]
        outputs.add(output_directory(config))
    assert len(outputs) == 3


def test_sequential_launcher_order_and_safe_default():
    launcher = (ROOT / "tools/experiments/train_sawyer_early_stop_three.sh").read_text()
    positions = [launcher.index(name) for name in ("push-back-v3", "stick-pull-v3", "push-v3")]
    assert positions == sorted(positions)
    assert "--preflight-only|--dry-run|--run" in launcher
    assert 'case "${1:-}"' in launcher


def test_target_requires_three_consecutive_exactly_perfect_evaluations():
    config = load_config(
        ROOT / "configs/sac/sawyer_push_continuation_early_stop.yaml",
        expected_experiment="sac",
    )
    callback = FixedSeedEvalCallback(
        MagicMock(),
        eval_freq=config.evaluation.frequency,
        n_eval_episodes=1,
        seed=0,
        log_path=Path("evaluations"),
        best_success_model_save_path=Path("best"),
        early_stopping=config.sac.early_stopping,
    )
    callback.training_start_timestep = 0
    callback.num_timesteps = config.sac.early_stopping.min_steps
    assert callback._update_early_stopping(1.0)
    assert callback._update_early_stopping(1.0)
    assert callback._update_early_stopping(0.99)
    assert callback.consecutive_successful_evaluations == 0
    assert callback._update_early_stopping(1.0)
    assert callback._update_early_stopping(1.0)
    assert not callback._update_early_stopping(1.0)
    assert callback.stopping_reason == "target-success"
