from pathlib import Path

from src.config import load_config, output_directory


CONFIG = Path(
    "configs/sac/kuka_push_cube_v3_contact_start_v6_mass100g_lr1e4_200k.yaml"
)


def test_v6_contact_start_config() -> None:
    config = load_config(CONFIG, expected_experiment="sac")
    assert config.environment.backend == "kuka-v3"
    assert config.environment.reward_function_version == "v6_rear_contact_push_progress"
    assert config.environment.cube_push_curriculum_level == 2
    assert config.environment.cube_push_initialization == "contact"
    assert config.sac.learning_rate == 1e-4
    assert config.sac.batch_size == 256
    assert config.sac.learning_starts == 20_000
    assert config.sac.buffer_size == 1_000_000
    assert config.sac.net_arch == [256, 256, 256, 256]
    assert config.sac.steps == 200_000 and config.sac.checkpoint_freq == 20_000
    assert config.sac.early_stopping.enabled
    assert config.sac.early_stopping.min_steps == 100_000
    assert config.sac.early_stopping.success_threshold == 0.95
    assert config.sac.early_stopping.patience == 3
    assert config.evaluation.episodes == 50
    assert config.evaluation.frequency == 20_000
    assert config.evaluation.task_set_seed == 10_000
    assert output_directory(config) == Path(
        "runs/exploratory/kuka-push-cube-v3/contact_start/"
        "reward_v6_rear_contact_push_progress/mass_100g/lr_1e4/seed_0_200k"
    ).resolve()
