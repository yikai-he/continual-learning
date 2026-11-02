"""Synthetic-only tests for the forward-transfer experiment infrastructure."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from scripts.aggregate_forward_transfer import aggregate
from scripts.run_forward_transfer import _hydrate_evaluations, _write_run_metadata
from src.continual.bc_policy import GeneralPolicy
from src.sac.forward_transfer import (
    compare_condition_summaries,
    evaluation_step_grid,
    learning_curve_metrics,
    load_forward_transfer_config,
    load_temporally_valid_general_policy,
    resolved_run_directory,
    save_resolved_forward_transfer_config,
)
from src.sac.policy_transfer import transfer_general_policy_to_sac
from src.sac.transferable_actor import TransferableSACPolicy


ROOT = Path(__file__).resolve().parents[1]


class SpacesOnlyEnv(gym.Env):
    """Provides spaces for SB3 construction; interaction is forbidden."""

    observation_space = gym.spaces.Box(-1, 1, shape=(39,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # pragma: no cover
        raise AssertionError("Forward-transfer unit tests must not execute an env.")

    def step(self, action):  # pragma: no cover
        raise AssertionError("Forward-transfer unit tests must not execute an env.")


def _model(seed=0):
    return SAC(
        TransferableSACPolicy,
        SpacesOnlyEnv(),
        policy_kwargs={"net_arch": [256, 256, 256, 256], "log_std_init": -3.0},
        seed=seed,
        device="cpu",
        verbose=0,
    )


def test_paired_configs_differ_only_in_condition_and_output_path():
    random = load_forward_transfer_config(
        ROOT / "configs/forward_transfer/hammer_4layer_random.yaml"
    )
    transferred = load_forward_transfer_config(
        ROOT / "configs/forward_transfer/hammer_4layer_transferred.yaml"
    )
    left, right = asdict(random), asdict(transferred)
    left.pop("condition")
    right.pop("condition")
    assert left == right
    assert random.condition == "random"
    assert transferred.condition == "transferred"
    assert resolved_run_directory(random).as_posix().endswith(
        "runs/forward_transfer_4layer/hammer-v3/random/seed_0"
    )
    assert resolved_run_directory(transferred).as_posix().endswith(
        "runs/forward_transfer_4layer/hammer-v3/transferred/seed_0"
    )


def test_random_and_transferred_start_identically_except_mean_pathway():
    random = _model(seed=9)
    transferred = _model(seed=9)
    for left, right in zip(
        random.policy.actor.parameters(), transferred.policy.actor.parameters()
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(
        random.policy.critic.parameters(), transferred.policy.critic.parameters()
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    general = GeneralPolicy(-np.ones(4), np.ones(4))
    log_std = {
        name: value.detach().clone()
        for name, value in transferred.policy.actor.log_std.named_parameters()
    }
    critic = {
        name: value.detach().clone()
        for name, value in transferred.policy.critic.state_dict().items()
    }
    transfer_general_policy_to_sac(transferred, general)
    assert not torch.equal(
        random.policy.actor.latent_pi[0].weight,
        transferred.policy.actor.latent_pi[0].weight,
    )
    for name, value in transferred.policy.actor.log_std.named_parameters():
        torch.testing.assert_close(value, log_std[name], rtol=0, atol=0)
    for name, value in transferred.policy.critic.state_dict().items():
        torch.testing.assert_close(value, critic[name], rtol=0, atol=0)
    assert random.policy.actor.optimizer.state == {}
    assert transferred.policy.actor.optimizer.state == {}
    assert random.policy.critic.optimizer.state == {}
    assert transferred.policy.critic.optimizer.state == {}
    assert random.replay_buffer.size() == transferred.replay_buffer.size() == 0


def test_temporal_checkpoint_requires_faucet_before_hammer(tmp_path):
    config = load_forward_transfer_config(
        ROOT / "configs/forward_transfer/hammer_4layer_transferred.yaml"
    )
    checkpoint = tmp_path / "policy.pt"
    policy = GeneralPolicy(-np.ones(4), np.ones(4))
    policy.save(
        checkpoint,
        metadata={
            "stage": 1,
            "task": "faucet-close-v3",
            "tasks": [
                "button-press-v3", "faucet-close-v3", "hammer-v3",
                "handle-press-side-v3", "window-close-v3",
            ],
        },
    )
    value = asdict(config)
    value["source_checkpoint"] = str(checkpoint)
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    loaded = load_forward_transfer_config(path)
    _, identity = load_temporally_valid_general_policy(loaded)
    assert identity["stage"] == 1
    assert identity["task_history"] == ["button-press-v3", "faucet-close-v3"]

    policy.save(
        tmp_path / "invalid.pt",
        metadata={
            "stage": 2,
            "task": "hammer-v3",
            "tasks": ["button-press-v3", "faucet-close-v3", "hammer-v3"],
        },
    )
    value["source_checkpoint"] = str(tmp_path / "invalid.pt")
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="trained only through"):
        load_temporally_valid_general_policy(load_forward_transfer_config(path))


def test_evaluation_grid_auc_threshold_and_pairing():
    steps = evaluation_step_grid(100_000, 10_000)
    assert steps == list(range(0, 100_001, 10_000))
    success = [step / 100_000 for step in steps]
    metrics = learning_curve_metrics(
        steps, success, [float(step) for step in steps], success_threshold=0.8
    )
    assert metrics["normalized_success_auc"] == pytest.approx(0.5)
    assert metrics["threshold_step"] == 80_000
    never = learning_curve_metrics(
        steps, [0.1] * len(steps), [0.0] * len(steps), success_threshold=0.8
    )
    assert never["threshold_step"] is None
    random = {
        "training_seed": 0,
        "evaluation_steps": steps,
        "metrics": metrics,
    }
    transferred = {
        "training_seed": 0,
        "evaluation_steps": steps,
        "metrics": {**metrics, "normalized_success_auc": 0.7},
    }
    paired = compare_condition_summaries(random, transferred)
    assert paired["delta_normalized_success_auc"] == pytest.approx(0.2)


def test_config_exclusive_save_and_resume_evaluation_hydration(tmp_path):
    config = load_forward_transfer_config(
        ROOT / "configs/forward_transfer/hammer_4layer_random.yaml", seed=2
    )
    config_path = tmp_path / "config.yaml"
    save_resolved_forward_transfer_config(config, config_path)
    assert load_forward_transfer_config(config_path) == config
    with pytest.raises(FileExistsError):
        save_resolved_forward_transfer_config(config, config_path)

    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    np.savez(
        evaluations / "evaluations.npz",
        timesteps=np.array([0, 10_000]),
        results=np.array([[1.0, 3.0], [2.0, 4.0]]),
        ep_lengths=np.array([[200, 200], [200, 200]]),
        successes=np.array([[False, True], [True, True]]),
        return_stddevs=np.array([1.0, 1.0]),
    )
    (evaluations / "success_diagnostics.json").write_text(
        json.dumps(
            [
                {"step": 0, "episodes": [{"success": False}]},
                {"step": 10_000, "episodes": [{"success": True}]},
            ]
        ),
        encoding="utf-8",
    )
    callback = SimpleNamespace()
    _hydrate_evaluations(callback, tmp_path)
    assert callback.timesteps == [0, 10_000]
    assert callback.successes == [[False, True], [True, True]]
    assert len(callback.success_diagnostics) == 2
    assert callback.best_mean_return == 3.0


def test_completed_pair_aggregation_uses_recorded_summaries_only(tmp_path):
    steps = [0, 10_000]
    for seed in (0, 1, 2):
        for condition, auc in (("random", 0.2), ("transferred", 0.5)):
            run = tmp_path / "hammer-v3" / condition / f"seed_{seed}"
            run.mkdir(parents=True)
            (run / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "training_seed": seed,
                        "evaluation_steps": steps,
                        "metrics": {
                            "normalized_success_auc": auc,
                            "final_success_rate": auc,
                            "threshold_step": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
    result = aggregate(tmp_path, [0, 1, 2])
    assert result["delta_normalized_success_auc"]["mean"] == pytest.approx(0.3)
    assert result["delta_normalized_success_auc"][
        "sample_standard_deviation"
    ] == pytest.approx(0.0)


def test_run_metadata_is_created_exclusively_then_updated_atomically(tmp_path):
    path = tmp_path / "metadata.json"
    initial = {"kind": "forward_transfer_run_v1", "phase": "initialized"}
    _write_run_metadata(path, initial, initial=True)
    assert json.loads(path.read_text(encoding="utf-8")) == initial

    with pytest.raises(FileExistsError):
        _write_run_metadata(path, initial, initial=True)

    initialized_model = {**initial, "initial_actor_sha256": "actor-hash"}
    _write_run_metadata(path, initialized_model, initial=False)
    assert json.loads(path.read_text(encoding="utf-8")) == initialized_model

    completed = {**initialized_model, "actual_final_timestep": 100_000}
    _write_run_metadata(path, completed, initial=False)
    assert json.loads(path.read_text(encoding="utf-8")) == completed


def test_run_metadata_update_requires_existing_owned_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        _write_run_metadata(
            tmp_path / "metadata.json",
            {"kind": "forward_transfer_run_v1"},
            initial=False,
        )
