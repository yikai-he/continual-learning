from pathlib import Path

import pytest

from src.config import config_from_dict, load_config, output_directory


CONFIG = Path("configs/sac/kuka_push_cube_v3_reward_v3_3_200k.yaml")


def test_cube_push_v3_3_resolves_controlled_training_contract() -> None:
    config = load_config(CONFIG, expected_experiment="sac")
    assert config.continual.tasks == ["kuka-push-cube-v3"]
    assert config.environment.backend == "kuka-v3"
    assert config.environment.reward_function_version == "v3_3_contact_push_progress"
    assert config.runtime.seed == 0
    assert config.runtime.device == "auto"
    assert config.sac.steps == 200_000
    assert config.sac.learning_rate == 3e-4
    assert config.sac.buffer_size == 1_000_000
    assert config.sac.learning_starts == 10_000
    assert config.sac.batch_size == 128
    assert config.sac.gamma == 0.99 and config.sac.tau == 0.005
    assert config.sac.train_freq == config.sac.gradient_steps == 1
    assert config.sac.ent_coef == "auto"
    assert config.sac.net_arch == [256, 256, 256, 256]
    assert config.sac.checkpoint_freq == 20_000
    assert config.evaluation.mode == "fixed-tasks"
    assert config.evaluation.episodes == 50
    assert config.evaluation.frequency == 20_000
    assert config.evaluation.seed_offset == config.evaluation.task_set_seed == 10_000
    assert config.sac.early_stopping.enabled
    assert (config.sac.early_stopping.min_steps,
            config.sac.early_stopping.success_threshold,
            config.sac.early_stopping.patience) == (100_000, 0.95, 3)
    assert output_directory(config) == Path(
        "runs/reward_ablations/kuka-push-cube-v3/"
        "reward_v3_3_contact_push_progress/seed_0_200k"
    ).resolve()


def test_v3_3_is_rejected_outside_single_task_cube_sac() -> None:
    with pytest.raises(ValueError, match="only for single-task KUKA-v3 SAC"):
        config_from_dict(
            {
                "experiment": "sac",
                "environment": {
                    "backend": "kuka-v3",
                    "reward_function_version": "v3_3_contact_push_progress",
                },
                "continual": {"tasks": ["kuka-push-v3"]},
            },
            expected_experiment="sac",
        )
