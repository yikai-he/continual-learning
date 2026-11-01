from pathlib import Path

import pytest

from src.config import config_from_dict, load_config, output_directory


CONFIG = Path("configs/sac/kuka_push_cube_v3_sawyer_push_v2_adapted_200k.yaml")
RESUME_CONFIG = Path(
    "configs/sac/kuka_push_cube_v3_sawyer_push_v2_adapted_resume_400k.yaml"
)


def test_sawyer_adapted_cube_config_matches_v3_1_optimization_contract() -> None:
    config = load_config(CONFIG, expected_experiment="sac")
    assert config.continual.tasks == ["kuka-push-cube-v3"]
    assert config.environment.backend == "kuka-v3"
    assert config.environment.reward_function_version == "sawyer_push_v2_adapted"
    assert config.runtime.seed == 0 and config.runtime.device == "auto"
    assert (config.sac.steps, config.sac.learning_rate) == (200_000, 3e-4)
    assert (config.sac.buffer_size, config.sac.learning_starts) == (1_000_000, 10_000)
    assert config.sac.batch_size == 128
    assert (config.sac.gamma, config.sac.tau) == (0.99, 0.005)
    assert config.sac.train_freq == config.sac.gradient_steps == 1
    assert config.sac.ent_coef == "auto"
    assert config.sac.net_arch == [256, 256, 256, 256]
    assert config.sac.checkpoint_freq == config.evaluation.frequency == 20_000
    assert config.evaluation.mode == "fixed-tasks"
    assert config.evaluation.episodes == 50
    assert config.evaluation.seed_offset == config.evaluation.task_set_seed == 10_000
    assert config.sac.early_stopping.enabled
    assert (config.sac.early_stopping.min_steps,
            config.sac.early_stopping.success_threshold,
            config.sac.early_stopping.patience) == (100_000, 0.95, 3)
    assert output_directory(config) == Path(
        "runs/reward_ablations/kuka-push-cube-v3/"
        "sawyer_push_v2_adapted/seed_0_200k"
    ).resolve()


def test_sawyer_adapted_reward_is_rejected_for_cylinder_config() -> None:
    with pytest.raises(ValueError, match="only for single-task KUKA-v3 SAC"):
        config_from_dict({"experiment":"sac","environment":{"backend":"kuka-v3",
            "reward_function_version":"sawyer_push_v2_adapted"},
            "continual":{"tasks":["kuka-push-v3"]}}, expected_experiment="sac")


def test_sawyer_adapted_resume_config_preserves_source_contract() -> None:
    source = load_config(CONFIG, expected_experiment="sac")
    resumed = load_config(RESUME_CONFIG, expected_experiment="sac")

    assert resumed.environment == source.environment
    assert resumed.continual == source.continual
    assert resumed.evaluation == source.evaluation
    assert resumed.runtime.seed == source.runtime.seed == 0
    assert resumed.runtime.device == source.runtime.device
    assert resumed.sac.steps == 400_000
    for field in (
        "learning_rate", "buffer_size", "learning_starts", "batch_size",
        "gamma", "tau", "train_freq", "gradient_steps", "ent_coef",
        "net_arch", "checkpoint_freq", "early_stopping",
    ):
        assert getattr(resumed.sac, field) == getattr(source.sac, field)
    assert resumed.runtime.output == source.runtime.output
    assert resumed.runtime.run_name == "seed_0_resume_200k_to_400k"
    assert output_directory(resumed) != output_directory(source)
