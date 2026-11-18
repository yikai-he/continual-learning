from pathlib import Path

import pytest

import train_sac
from src.config import EarlyStoppingConfig, load_config, output_directory


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/sac/sawyer_scratch_2m"


@pytest.mark.parametrize(
    ("filename", "task"),
    (("push_back_v3.yaml", "push-back-v3"), ("push_v3.yaml", "push-v3")),
)
def test_sawyer_scratch_2m_configs(filename, task):
    config = load_config(CONFIG_DIR / filename, expected_experiment="sac")
    assert config.continual.tasks == [task]
    assert config.environment.backend == "metaworld-v3"
    assert config.environment.reward_function_version == "v2"
    assert config.sac.steps == 2_000_000
    assert config.sac.learning_rate == 0.001
    assert config.sac.buffer_size == 1_000_000
    assert config.sac.batch_size == 128
    assert config.sac.gamma == 0.99
    assert config.sac.tau == 0.005
    assert config.sac.train_freq == config.sac.gradient_steps == 50
    assert config.actor_init.mode == "scratch"
    assert config.actor_init.checkpoint is None
    assert not config.resume.enabled
    assert config.resume.checkpoint is config.resume.replay_buffer is None
    assert config.resume.refill_steps == 0
    stop = config.sac.early_stopping
    assert stop.enabled and stop.min_steps == 0
    assert stop.success_threshold == 1.0 and stop.patience == 3
    assert stop.plateau_patience is None
    assert config.evaluation.episodes == 50
    assert config.evaluation.final_episodes == 50
    assert config.evaluation.frequency == 40_000
    assert config.evaluation.mode == "fixed-tasks"
    assert config.evaluation.task_set_seed == 10_000
    assert config.evaluation.final_task_set_seeds == [20_000, 20_001]
    assert output_directory(config) == (
        ROOT / "runs/sac_sawyer_scratch_2m" / task / "seed_0"
    )


def test_target_only_stopping_requires_three_consecutive_perfect_results():
    callback = train_sac.FixedSeedEvalCallback.__new__(train_sac.FixedSeedEvalCallback)
    callback.early_stopping = EarlyStoppingConfig(
        enabled=True,
        min_steps=0,
        success_threshold=1.0,
        patience=3,
        plateau_patience=None,
    )
    callback.training_start_timestep = 0
    callback.num_timesteps = 40_000
    callback.consecutive_successful_evaluations = 0
    callback.best_plateau_success = float("-inf")
    callback.plateau_evaluations = 0
    callback.early_stopped = False
    callback.stop_timestep = None
    callback.trigger_success_rate = None
    callback.stopping_reason = None

    assert callback._update_early_stopping(1.0)
    callback.num_timesteps += 40_000
    assert callback._update_early_stopping(49 / 50)
    assert callback.consecutive_successful_evaluations == 0
    for _ in range(20):
        callback.num_timesteps += 40_000
        assert callback._update_early_stopping(0.0)
    for expected in (1, 2):
        callback.num_timesteps += 40_000
        assert callback._update_early_stopping(1.0)
        assert callback.consecutive_successful_evaluations == expected
    callback.num_timesteps += 40_000
    assert not callback._update_early_stopping(1.0)
    assert callback.stopping_reason == "target-success"


def test_launcher_uses_required_order_and_explicit_modes():
    launcher = (ROOT / "tools/experiments/train_sawyer_scratch_2m.sh").read_text()
    assert "tasks=(push-back-v3 push-v3)" in launcher
    assert '"--preflight-only"' in launcher
    assert '"--run"' in launcher
    assert "Refusing existing run" in launcher
