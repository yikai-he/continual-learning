"""Controlled GeneralPolicy LayerNorm ablation configuration contract."""

from dataclasses import asdict
from pathlib import Path

from src.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "diffcrl"


def _scientific_without_ablation(config):
    value = asdict(config)
    value.pop("runtime")
    value["bc"].pop("use_layer_norm")
    value["bc"].pop("normalization_mode")
    return value


def test_ablation_configs_change_only_layer_norm_and_output_identity():
    enabled = load_config(
        CONFIG_ROOT / "general_policy_layer_norm_ablation_enabled.yaml",
        expected_experiment="diffcrl",
    )
    disabled = load_config(
        CONFIG_ROOT / "general_policy_layer_norm_ablation_disabled.yaml",
        expected_experiment="diffcrl",
    )
    reference = load_config(
        CONFIG_ROOT / "diffcrl_5task_15plus15_2layer.yaml",
        expected_experiment="diffcrl",
    )

    assert enabled.bc.use_layer_norm is True
    assert enabled.bc.normalization_mode == "layer-norm"
    assert disabled.bc.use_layer_norm is False
    assert disabled.bc.normalization_mode == "none"
    assert enabled.runtime.output != disabled.runtime.output
    assert _scientific_without_ablation(enabled) == _scientific_without_ablation(
        disabled
    )
    assert _scientific_without_ablation(enabled) == _scientific_without_ablation(
        reference
    )


def test_ablation_uses_formal_curriculum_and_diffusion_replay():
    config = load_config(
        CONFIG_ROOT / "general_policy_layer_norm_ablation_enabled.yaml",
        expected_experiment="diffcrl",
    )
    assert config.continual.tasks == [
        "button-press-v3",
        "faucet-close-v3",
        "hammer-v3",
        "handle-press-side-v3",
        "window-close-v3",
    ]
    assert config.continual.replay_mode == "diffusion"
    assert config.continual.replay_allocation == "fixed-total"
    assert config.continual.replay_budget == 15
    assert config.evaluation.episodes == 50
    assert config.evaluation.task_set_seed == 10_000
