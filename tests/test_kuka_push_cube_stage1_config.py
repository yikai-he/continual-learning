from pathlib import Path

import pytest

from src.config import load_config, output_directory


CONFIG = Path(
    "configs/sac/kuka_push_cube_v3_initial_to_contact_v1_mass100g_lr1e4_400k.yaml"
)


def test_stage1_sac_configuration():
    config = load_config(CONFIG, expected_experiment="sac")
    assert config.environment.backend == "kuka-v3"
    assert config.environment.reward_function_version == "v1_initial_to_contact"
    assert config.environment.cube_push_curriculum_level == 2
    assert config.environment.cube_push_initialization == "default"
    assert config.continual.tasks == ["kuka-push-cube-v3"]
    assert config.sac.steps == 400_000
    assert config.sac.learning_rate == 1e-4
    assert config.sac.buffer_size == 1_000_000
    assert config.sac.learning_starts == 20_000
    assert config.sac.batch_size == 256
    assert config.sac.net_arch == [256, 256, 256, 256]
    assert config.sac.gamma == 0.99 and config.sac.tau == 0.005
    assert config.sac.train_freq == config.sac.gradient_steps == 1
    assert config.sac.ent_coef == "auto"
    assert config.sac.checkpoint_freq == 20_000
    assert config.evaluation.episodes == 50
    assert config.evaluation.frequency == 20_000
    assert config.evaluation.task_set_seed == 10_000
    assert config.evaluation.mode == "fixed-tasks"
    assert config.sac.early_stopping.enabled
    assert config.sac.early_stopping.min_steps == 100_000
    assert config.sac.early_stopping.success_threshold == 0.95
    assert config.sac.early_stopping.patience == 3
    assert output_directory(config) == Path(
        "runs/exploratory/kuka-push-cube-v3/initial_to_contact/"
        "reward_v1_initial_to_contact/mass_100g/lr_1e4/seed_0_400k"
    ).resolve()


@pytest.mark.parametrize(
    "replacement",
    [
        "cube_push_curriculum_level: 3",
        "cube_push_initialization: contact",
    ],
)
def test_stage1_rejects_incompatible_environment(tmp_path, replacement):
    text = CONFIG.read_text()
    if replacement.endswith("3"):
        text = text.replace("cube_push_curriculum_level: 2", replacement)
    else:
        text = text.replace("cube_push_initialization: default", replacement)
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="v1_initial_to_contact requires"):
        load_config(path, expected_experiment="sac")
