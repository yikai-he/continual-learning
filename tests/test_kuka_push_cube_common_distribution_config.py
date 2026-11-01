from pathlib import Path

import pytest

from src.config import load_config, output_directory


NEAR = Path("configs/sac/kuka_push_cube_v3_near_contact_5mm_v6_common_distribution_400k.yaml")
CONTACT = Path("configs/sac/kuka_push_cube_v3_contact_start_v6_common_distribution_400k.yaml")


@pytest.mark.parametrize("path, initialization", [(NEAR, "near_contact"), (CONTACT, "contact")])
def test_common_distribution_configs(path: Path, initialization: str) -> None:
    config = load_config(path, expected_experiment="sac")
    assert config.environment.backend == "kuka-v3"
    assert config.environment.reward_function_version == "v6_rear_contact_push_progress"
    assert config.environment.cube_push_curriculum_level == 2
    assert config.environment.cube_push_initialization == initialization
    assert config.environment.cube_y_min == pytest.approx(0.502)
    assert config.environment.near_contact_gap_m == pytest.approx(0.005)
    assert config.sac.steps == 400_000
    assert config.sac.learning_rate == 1e-4
    assert config.sac.buffer_size == 1_000_000
    assert config.sac.learning_starts == 20_000
    assert config.sac.batch_size == 256
    assert config.sac.net_arch == [256, 256, 256, 256]
    assert config.sac.checkpoint_freq == config.evaluation.frequency == 20_000
    assert config.sac.early_stopping.enabled
    assert config.evaluation.episodes == 50
    assert config.evaluation.mode == "fixed-tasks"
    assert config.evaluation.task_set_seed == 10_000
    assert "common_distribution" in str(output_directory(config))


def test_common_distribution_outputs_are_isolated() -> None:
    near = output_directory(load_config(NEAR, expected_experiment="sac"))
    contact = output_directory(load_config(CONTACT, expected_experiment="sac"))
    assert near != contact
    # Configuration isolation is a path-identity contract; existing experiment
    # artifacts must not make this test order- or workstation-dependent.
    assert "near_contact_5mm" in near.parts
    assert "contact_start" in contact.parts
