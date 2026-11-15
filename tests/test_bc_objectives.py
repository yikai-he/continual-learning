"""Controlled deterministic-MSE versus Squashed-Gaussian-NLL BC contracts."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.config import config_from_dict, default_config, load_config
from src.continual.bc_policy import (
    DETERMINISTIC_MSE,
    SQUASHED_GAUSSIAN_NLL,
    GeneralPolicy,
    fit_bc,
)


def dataset(count=16, target=0.4):
    observations = torch.randn(count, 39)
    actions = torch.full((count, 4), target)
    return TensorDataset(observations, actions)


def test_deterministic_policy_is_bounded_and_ignores_sampling_flag():
    policy = GeneralPolicy(
        -np.ones(4), np.ones(4), objective=DETERMINISTIC_MSE
    )
    observation = np.zeros(39, dtype=np.float32)
    deterministic = policy.act(observation, deterministic=True)
    nominally_stochastic = policy.act(observation, deterministic=False)
    assert deterministic.shape == (4,)
    assert np.array_equal(deterministic, nominally_stochastic)
    assert np.all(deterministic >= -1) and np.all(deterministic <= 1)
    assert policy.log_std_head is None


def test_deterministic_objective_is_physical_action_mse():
    policy = GeneralPolicy(
        -np.ones(4), np.ones(4), objective=DETERMINISTIC_MSE
    )
    for parameter in policy.parameters():
        nn.init.zeros_(parameter)
    data = dataset(count=8, target=1.0)
    result = fit_bc(
        policy,
        data,
        data,
        epochs=1,
        learning_rate=1e-8,
        bc_loss=DETERMINISTIC_MSE,
    )
    assert result["initial"]["train_loss"] == pytest.approx(1.0)
    assert result["initial"]["train_action_mse"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "objective", [SQUASHED_GAUSSIAN_NLL, DETERMINISTIC_MSE]
)
def test_each_objective_updates_parameters_and_trains_with_validation(objective):
    torch.manual_seed(7)
    policy = GeneralPolicy(-np.ones(4), np.ones(4), objective=objective)
    before = {name: value.detach().clone() for name, value in policy.named_parameters()}
    result = fit_bc(
        policy,
        dataset(),
        dataset(),
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        seed=11,
        bc_loss=objective,
    )
    assert result["best_epoch"] in (1, 2)
    assert np.isfinite(result["train_loss"])
    assert np.isfinite(result["validation_loss"])
    assert any(
        not torch.equal(before[name], value)
        for name, value in policy.named_parameters()
    )


@pytest.mark.parametrize(
    "objective,other",
    [
        (SQUASHED_GAUSSIAN_NLL, DETERMINISTIC_MSE),
        (DETERMINISTIC_MSE, SQUASHED_GAUSSIAN_NLL),
    ],
)
def test_checkpoint_kind_and_objective_are_strict(tmp_path, objective, other):
    policy = GeneralPolicy(-np.ones(4), np.ones(4), objective=objective)
    path = tmp_path / f"{objective}.pt"
    policy.save(path)
    restored = GeneralPolicy.load(path, expected_objective=objective)
    assert restored.objective == objective
    with pytest.raises(ValueError, match="objective does not match"):
        GeneralPolicy.load(path, expected_objective=other)


def test_training_rejects_policy_objective_mismatch():
    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="does not match"):
        fit_bc(
            policy,
            dataset(),
            dataset(),
            epochs=1,
            bc_loss=DETERMINISTIC_MSE,
        )


@pytest.mark.parametrize(
    "replay_mode", ["none", "expert-recollection", "diffusion", "flow-matching"]
)
@pytest.mark.parametrize(
    "objective", [SQUASHED_GAUSSIAN_NLL, DETERMINISTIC_MSE]
)
def test_objectives_validate_with_every_replay_mode(replay_mode, objective):
    values = asdict(default_config("diffcrl"))
    values["bc"]["loss"] = objective
    values["continual"]["replay_mode"] = replay_mode
    values["runtime"]["output"] = "runs/test_bc_objectives"
    config = config_from_dict(values, expected_experiment="diffcrl")
    assert config.bc.loss == objective
    assert config.continual.replay_mode == replay_mode


def test_unknown_bc_objective_is_rejected():
    values = asdict(default_config("diffcrl"))
    values["bc"]["loss"] = "mean_squared_error"
    values["runtime"]["output"] = "runs/test_bc_objectives"
    with pytest.raises(ValueError, match="bc.loss"):
        config_from_dict(values, expected_experiment="diffcrl")


def test_collection_configuration_defaults_and_validation():
    values = asdict(default_config("diffcrl"))
    values["runtime"]["output"] = "runs/test_collection_config"
    config = config_from_dict(values)
    assert config.bc.loss == DETERMINISTIC_MSE
    assert config.continual.collection_mode == "all"
    assert config.continual.collection_max_attempt_multiplier == 2
    assert config.continual.expert_action_mode == "deterministic"

    values["continual"]["collection_mode"] = "success_preferred"
    values["continual"]["collection_max_attempt_multiplier"] = 3
    config = config_from_dict(values)
    assert config.continual.collection_mode == "success_preferred"
    assert config.continual.collection_max_attempt_multiplier == 3

    values["continual"]["expert_action_mode"] = "stochastic"
    config = config_from_dict(values)
    assert config.continual.expert_action_mode == "stochastic"

    values["continual"]["expert_action_mode"] = "exploratory"
    with pytest.raises(ValueError, match="expert_action_mode"):
        config_from_dict(values)
    values["continual"]["expert_action_mode"] = "deterministic"

    values["continual"]["collection_mode"] = "success_only"
    with pytest.raises(ValueError, match="collection_mode"):
        config_from_dict(values)
    values["continual"]["collection_mode"] = "success_preferred"
    values["continual"]["collection_max_attempt_multiplier"] = 0
    with pytest.raises(ValueError, match="collection_max_attempt_multiplier"):
        config_from_dict(values)


@pytest.mark.parametrize(
    "replay_mode", ["none", "expert-recollection", "diffusion", "flow-matching"]
)
def test_success_preferred_collection_validates_with_every_replay_mode(replay_mode):
    values = asdict(default_config("diffcrl"))
    values["runtime"]["output"] = "runs/test_collection_config"
    values["continual"]["replay_mode"] = replay_mode
    values["continual"]["collection_mode"] = "success_preferred"
    config = config_from_dict(values)
    assert config.continual.collection_mode == "success_preferred"
    assert config.continual.replay_mode == replay_mode


def test_dual_checkpoint_selection_is_opt_in_and_nll_only():
    values = asdict(default_config("diffcrl"))
    values["runtime"]["output"] = "runs/test_bc_objectives"
    assert config_from_dict(values).bc.dual_checkpoint_selection is False
    values["bc"]["dual_checkpoint_selection"] = True
    values["bc"]["loss"] = DETERMINISTIC_MSE
    with pytest.raises(ValueError, match="requires squashed_gaussian_nll"):
        config_from_dict(values)


def test_one_nll_fit_tracks_independent_best_states_and_restores_best_nll():
    torch.manual_seed(0)
    policy = GeneralPolicy(
        -np.ones(4), np.ones(4), objective=SQUASHED_GAUSSIAN_NLL
    )
    observations = torch.randn(24, 39)
    actions = torch.tanh(torch.randn(24, 4))
    train = TensorDataset(observations[:16], actions[:16])
    validation = TensorDataset(observations[16:], actions[16:])
    result = fit_bc(
        policy,
        train,
        validation,
        epochs=5,
        batch_size=8,
        seed=0,
        bc_loss=SQUASHED_GAUSSIAN_NLL,
        track_best_mse=True,
    )
    best_mse = result["best_mse_checkpoint"]
    assert result["best_epoch"] == min(
        result["history"], key=lambda item: item["validation_loss"]
    )["epoch"]
    assert best_mse["epoch"] == min(
        result["history"], key=lambda item: item["validation_action_mse"]
    )["epoch"]
    assert result["best_epoch"] != best_mse["epoch"]

    with torch.no_grad():
        authoritative_mse = torch.nn.functional.mse_loss(
            policy(validation.tensors[0]), validation.tensors[1]
        ).item()
    assert authoritative_mse == pytest.approx(result["validation_action_mse"])

    alternative = GeneralPolicy(
        -np.ones(4), np.ones(4), objective=SQUASHED_GAUSSIAN_NLL
    )
    alternative.load_state_dict(result["_best_mse_state_dict"])
    with torch.no_grad():
        alternative_mse = torch.nn.functional.mse_loss(
            alternative(validation.tensors[0]), validation.tensors[1]
        ).item()
        alternative_nll = -alternative.normalized_log_prob(
            validation.tensors[0], validation.tensors[1]
        ).mean().item()
    assert alternative_mse == pytest.approx(best_mse["validation_action_mse"])
    assert alternative_nll == pytest.approx(best_mse["validation_loss"])


def test_disabled_tracking_preserves_existing_result_schema():
    torch.manual_seed(3)
    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    result = fit_bc(policy, dataset(), dataset(), epochs=1)
    assert "best_mse_checkpoint" not in result
    assert "_best_mse_state_dict" not in result


def test_nll_physical_mse_selection_restores_policy_and_optimizer_from_mse_epoch(
    tmp_path,
):
    torch.manual_seed(0)
    template = GeneralPolicy(
        -np.ones(4), np.ones(4), objective=SQUASHED_GAUSSIAN_NLL
    )
    initial = {name: value.clone() for name, value in template.state_dict().items()}
    observations = torch.randn(24, 39)
    actions = torch.tanh(torch.randn(24, 4))
    train = TensorDataset(observations[:16], actions[:16])
    validation = TensorDataset(observations[16:], actions[16:])

    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    policy.load_state_dict(initial)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    selected = fit_bc(
        policy,
        train,
        validation,
        epochs=5,
        batch_size=8,
        seed=0,
        optimizer=optimizer,
        checkpoint_selection="validation_physical_mse",
    )
    expected = min(
        selected["history"], key=lambda item: item["validation_action_mse"]
    )
    assert selected["best_epoch"] == expected["epoch"]
    assert selected["validation_action_mse"] == pytest.approx(
        expected["validation_action_mse"]
    )
    assert selected["validation_loss"] == pytest.approx(expected["validation_loss"])

    reference = GeneralPolicy(-np.ones(4), np.ones(4))
    reference.load_state_dict(initial)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    fit_bc(
        reference,
        train,
        validation,
        epochs=expected["epoch"],
        batch_size=8,
        seed=0,
        optimizer=reference_optimizer,
        checkpoint_selection="validation_physical_mse",
    )
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    selected_optimizer = optimizer.state_dict()
    expected_optimizer = reference_optimizer.state_dict()
    assert selected_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter, state in selected_optimizer["state"].items():
        for name, value in state.items():
            expected_value = expected_optimizer["state"][parameter][name]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
            else:
                assert value == expected_value

    from types import SimpleNamespace
    from src.continual.diffcrl import DiffCRLTrainer
    from continual_manipulation.sequences import TaskSequence
    from src.support.reproducibility import state_hash

    trainer = object.__new__(DiffCRLTrainer)
    trainer.next_stage = 1
    trainer.sequence = TaskSequence.from_names(["reach-v3", "button-press-v3"])
    trainer.config = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu"),
            continual=SimpleNamespace(
                trajectories_per_task=2,
                current_trajectory_schedule=[],
            ),
    )
    trainer.progress = False
    trainer.output = tmp_path
    trainer.policy = policy
    trainer.diffusion = None
    trainer.flow_matching = None
    context, _ = trainer._begin_stage(1)
    assert context.policy_before_hash == state_hash(policy)


def test_best_nll_selection_remains_the_default():
    torch.manual_seed(0)
    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    observations = torch.randn(24, 39)
    actions = torch.tanh(torch.randn(24, 4))
    result = fit_bc(
        policy,
        TensorDataset(observations[:16], actions[:16]),
        TensorDataset(observations[16:], actions[16:]),
        epochs=5,
        batch_size=8,
        seed=0,
    )
    expected = min(result["history"], key=lambda item: item["validation_loss"])
    assert result["best_epoch"] == expected["epoch"]


def test_invalid_checkpoint_selection_combinations_are_rejected():
    values = asdict(default_config("diffcrl"))
    values["runtime"]["output"] = "runs/test_bc_objectives"
    values["bc"]["checkpoint_selection"] = "validation_physical_mse"
    values["bc"]["loss"] = DETERMINISTIC_MSE
    with pytest.raises(ValueError, match="requires squashed_gaussian_nll"):
        config_from_dict(values)

    values["bc"]["loss"] = SQUASHED_GAUSSIAN_NLL
    values["bc"]["dual_checkpoint_selection"] = True
    with pytest.raises(ValueError, match="Dual checkpoint evaluation"):
        config_from_dict(values)


@pytest.mark.parametrize("replay", ["ddpm_replay"])
def test_formal_ablation_pairs_differ_only_in_objective_and_run_name(replay):
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "bc_objective_ablation"
    prefix = f"kuka_4task_{replay}_"
    mse = asdict(
        load_config(
            root / f"{prefix}deterministic_mse_seed0.yaml",
            expected_experiment="diffcrl",
        )
    )
    nll = asdict(
        load_config(
            root / f"{prefix}squashed_gaussian_nll_seed0.yaml",
            expected_experiment="diffcrl",
        )
    )
    assert mse["bc"].pop("loss") == DETERMINISTIC_MSE
    assert nll["bc"].pop("loss") == SQUASHED_GAUSSIAN_NLL
    assert mse["runtime"].pop("run_name") == "deterministic_mse_seed0"
    assert nll["runtime"].pop("run_name") == "squashed_gaussian_nll_seed0"
    assert mse == nll


def test_ddpm_and_flow_100_plus_100_configs_have_only_method_differences():
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "replay_budget_ablation"
    ddpm = asdict(load_config(
        root / "kuka_4task_ddpm_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))
    flow = asdict(load_config(
        root / "kuka_4task_fm_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))

    assert ddpm["continual"].pop("replay_mode") == "diffusion"
    assert flow["continual"].pop("replay_mode") == "flow-matching"
    assert ddpm["flow_matching"].pop("integration_steps") == 100
    assert flow["flow_matching"].pop("integration_steps") == 10
    assert ddpm["runtime"].pop("run_name") == (
        "kuka_4task_ddpm_mse_100real_100replay_seed0"
    )
    assert flow["runtime"].pop("run_name") == (
        "kuka_4task_fm_mse_100real_100replay_seed0"
    )
    assert ddpm == flow


def test_ddpm_and_expert_recollection_100_plus_100_configs_differ_only_in_identity():
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "replay_budget_ablation"
    ddpm = asdict(load_config(
        root / "kuka_4task_ddpm_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))
    recollection = asdict(load_config(
        root / "kuka_4task_expert_recollection_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))

    assert ddpm["continual"].pop("replay_mode") == "diffusion"
    assert recollection["continual"].pop("replay_mode") == "expert-recollection"
    assert ddpm["runtime"].pop("run_name") == (
        "kuka_4task_ddpm_mse_100real_100replay_seed0"
    )
    assert recollection["runtime"].pop("run_name") == (
        "kuka_4task_expert_recollection_mse_100real_100replay_seed0"
    )
    assert ddpm == recollection


@pytest.mark.parametrize(
    ("method", "mode"),
    [
        ("expert_recollection", "expert-recollection"),
        ("ddpm", "diffusion"),
        ("fm", "flow-matching"),
    ],
)
def test_100_plus_300_budget_configs_only_change_budget_and_run_identity(
    method, mode
):
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "replay_budget_ablation"
    baseline = asdict(load_config(
        root / f"kuka_4task_{method}_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))
    increased = asdict(load_config(
        root / f"kuka_4task_{method}_mse_100real_300replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))

    assert baseline["continual"]["replay_mode"] == mode
    assert increased["continual"]["replay_mode"] == mode
    assert baseline["continual"].pop("replay_budget") == 100
    assert increased["continual"].pop("replay_budget") == 300
    assert baseline["runtime"].pop("run_name").endswith("100replay_seed0")
    assert increased["runtime"].pop("run_name").endswith("300replay_seed0")
    assert baseline == increased


@pytest.mark.parametrize(
    ("method", "mode"),
    [("ddpm", "diffusion"), ("fm", "flow-matching")],
)
def test_fixed_weighted_configs_only_change_allocation_weights_and_identity(
    method, mode
):
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "replay_budget_ablation"
    uniform = asdict(load_config(
        root / f"kuka_4task_{method}_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))
    weighted = asdict(load_config(
        root / f"kuka_4task_{method}_mse_100real_100replay_weighted_1_2_2_seed0.yaml",
        expected_experiment="diffcrl",
    ))

    assert uniform["continual"]["replay_mode"] == mode
    assert weighted["continual"]["replay_mode"] == mode
    assert uniform["continual"].pop("replay_allocation") == "fixed-total"
    assert weighted["continual"].pop("replay_allocation") == "fixed-weighted"
    assert uniform["continual"].pop("replay_task_weights") == []
    assert weighted["continual"].pop("replay_task_weights") == [1.0, 2.0, 2.0, 1.0]
    assert uniform["runtime"].pop("run_name").endswith("100replay_seed0")
    assert weighted["runtime"].pop("run_name").endswith(
        "100replay_weighted_1_2_2_seed0"
    )
    assert uniform == weighted


@pytest.mark.parametrize(
    ("method", "mode"),
    [("ddpm", "diffusion"), ("fm", "flow-matching")],
)
def test_50_current_weighted_configs_only_add_schedule_and_change_identity(
    method, mode
):
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "replay_budget_ablation"
    baseline = asdict(load_config(
        root / f"kuka_4task_{method}_mse_100real_100replay_weighted_1_2_2_seed0.yaml",
        expected_experiment="diffcrl",
    ))
    scheduled = asdict(load_config(
        root
        / f"kuka_4task_{method}_mse_100initial_50current_100replay_weighted_1_2_2_seed0.yaml",
        expected_experiment="diffcrl",
    ))

    assert baseline["continual"]["replay_mode"] == mode
    assert scheduled["continual"]["replay_mode"] == mode
    assert baseline["continual"].pop("current_trajectory_schedule") == []
    assert scheduled["continual"].pop("current_trajectory_schedule") == [
        100, 50, 50, 50
    ]
    assert baseline["runtime"].pop("run_name").endswith(
        "100replay_weighted_1_2_2_seed0"
    )
    assert scheduled["runtime"].pop("run_name").endswith(
        "100initial_50current_100replay_weighted_1_2_2_seed0"
    )
    assert baseline == scheduled


def test_formal_fm_unet_config_only_changes_backbone_width_and_identity():
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "replay_budget_ablation"
    cnn = asdict(load_config(
        root / "kuka_4task_fm_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))
    unet = asdict(load_config(
        root / "kuka_4task_fm_unet1d_w32_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))

    assert cnn["flow_matching"].pop("backbone") == "temporal_cnn"
    assert unet["flow_matching"].pop("backbone") == "temporal_unet1d"
    assert cnn["flow_matching"].pop("width") == 64
    assert unet["flow_matching"].pop("width") == 32
    assert cnn["flow_matching"]["integration_steps"] == 10
    assert unet["flow_matching"]["integration_steps"] == 10
    assert cnn["runtime"].pop("run_name") == (
        "kuka_4task_fm_mse_100real_100replay_seed0"
    )
    assert unet["runtime"].pop("run_name") == (
        "kuka_4task_fm_unet1d_w32_mse_100real_100replay_seed0"
    )
    assert cnn == unet


def test_formal_ddpm_unet_config_only_changes_backbone_width_and_identity():
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "replay_budget_ablation"
    cnn = asdict(load_config(
        root / "kuka_4task_ddpm_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))
    unet = asdict(load_config(
        root / "kuka_4task_ddpm_unet1d_w32_mse_100real_100replay_seed0.yaml",
        expected_experiment="diffcrl",
    ))

    assert cnn["diffusion"].pop("backbone") == "temporal_cnn"
    assert unet["diffusion"].pop("backbone") == "temporal_unet1d"
    assert cnn["diffusion"].pop("width") == 64
    assert unet["diffusion"].pop("width") == 32
    assert cnn["diffusion"]["steps"] == unet["diffusion"]["steps"] == 100
    assert cnn["runtime"].pop("run_name") == (
        "kuka_4task_ddpm_mse_100real_100replay_seed0"
    )
    assert unet["runtime"].pop("run_name") == (
        "kuka_4task_ddpm_unet1d_w32_mse_100real_100replay_seed0"
    )
    assert cnn == unet


def test_unet_smoke_config_is_explicitly_small_and_keeps_euler_ten_steps():
    config = load_config(
        Path(__file__).parents[1]
        / "configs/diffcrl/flow_matching_unet1d_smoke.yaml",
        expected_experiment="diffcrl",
    )
    assert config.flow_matching.backbone == "temporal_unet1d"
    assert config.flow_matching.width == 8
    assert config.flow_matching.epochs == 2
    assert config.flow_matching.integration_steps == 10
    assert config.flow_matching.ode_solver == "euler"
    assert config.continual.trajectories_per_task == 4
    assert config.evaluation.episodes == 2


@pytest.mark.parametrize(
    "name",
    [
        "kuka_4task_real_replay_squashed_gaussian_nll_seed0.yaml",
        "kuka_4task_real_replay_squashed_gaussian_nll_dual_selection_seed0.yaml",
        "kuka_4task_real_replay_nll_best_mse_inheritance_seed0.yaml",
    ],
)
def test_archival_real_replay_configs_are_not_reinterpreted(name):
    root = Path(__file__).parents[1] / "configs" / "diffcrl" / "bc_objective_ablation"
    with pytest.raises(ValueError, match="retired stored-trajectory protocol"):
        load_config(root / name, expected_experiment="diffcrl")
