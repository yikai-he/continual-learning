from dataclasses import replace

from run_replay_comparison import comparison_configs
from src.config import default_config


def test_comparison_arms_are_matched_and_isolated(tmp_path):
    base = default_config("diffcrl")
    base = replace(
        base,
        evaluation=replace(base.evaluation, episodes=50, mode="fixed-tasks"),
        continual=replace(
            base.continual,
            replay_mode="diffusion",
            replay_allocation="fixed-total",
            replay_budget=50,
        ),
    )
    configs = comparison_configs(base, tmp_path)

    assert set(configs) == {"no_replay", "real_replay", "ddpm_replay"}
    assert len({cfg.runtime.run_name for cfg in configs.values()}) == 3
    shared = configs["ddpm_replay"]
    for cfg in configs.values():
        assert cfg.bc == shared.bc
        assert cfg.evaluation == shared.evaluation
        assert cfg.continual.tasks == shared.continual.tasks
        assert cfg.continual.trajectories_per_task == 200
    assert configs["real_replay"].continual.replay_mode == "real"
    assert configs["real_replay"].continual.replay_budget == 50
    assert configs["ddpm_replay"].continual.replay_budget == 50
    assert configs["ddpm_replay"].continual.historical_replay_training_only
    assert configs["real_replay"].continual.historical_replay_training_only
    assert not configs["no_replay"].continual.historical_replay_training_only


def test_comparison_arms_inherit_base_replay_budget(tmp_path):
    base = default_config("diffcrl")
    base = replace(
        base,
        evaluation=replace(base.evaluation, episodes=50, mode="fixed-tasks"),
        continual=replace(
            base.continual,
            replay_mode="diffusion",
            replay_allocation="fixed-total",
            replay_budget=25,
        ),
    )

    configs = comparison_configs(base, tmp_path)

    assert configs["real_replay"].continual.replay_budget == 25
    assert configs["ddpm_replay"].continual.replay_budget == 25
    assert configs["no_replay"].continual.replay_mode == "none"
