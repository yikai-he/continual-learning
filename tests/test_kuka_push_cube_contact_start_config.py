from pathlib import Path
from src.config import load_config, output_directory


def test_contact_start_config() -> None:
    config = load_config(Path("configs/sac/kuka_push_cube_v3_contact_start_v5_mass100g_lr1e4_200k.yaml"), expected_experiment="sac")
    assert config.environment.cube_push_curriculum_level == 2
    assert config.environment.cube_push_initialization == "contact"
    assert config.environment.reward_function_version == "v5_contact_start_progress"
    assert config.sac.learning_rate == 1e-4 and config.sac.batch_size == 256
    assert config.sac.learning_starts == 20_000 and config.sac.steps == 200_000
    assert config.evaluation.episodes == 50 and config.evaluation.task_set_seed == 10_000
    assert output_directory(config) == Path("runs/exploratory/kuka-push-cube-v3/contact_start/reward_v5_progress/mass_100g/lr_1e4/seed_0_200k").resolve()
