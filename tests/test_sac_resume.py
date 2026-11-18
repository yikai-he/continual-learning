"""Contracts for standalone SAC full-model continuation."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import train_sac
from src.config import EarlyStoppingConfig, config_from_dict, default_config


def test_resume_defaults_off_and_is_mutually_exclusive_with_actor_warm_start():
    config = default_config("sac")
    assert not config.resume.enabled
    assert config.resume.checkpoint is None
    with pytest.raises(ValueError, match="mutually exclusive"):
        config_from_dict(
            {
                "experiment": "sac",
                "actor_init": {"mode": "sac_actor", "checkpoint": "actor.zip"},
                "resume": {"enabled": True, "checkpoint": "model.zip"},
            }
        )
    with pytest.raises(ValueError, match="resume.checkpoint is required"):
        config_from_dict({"experiment": "sac", "resume": {"enabled": True}})
    with pytest.raises(ValueError, match="replay buffer is restored"):
        config_from_dict(
            {
                "experiment": "sac",
                "sac": {"steps": 100},
                "resume": {
                    "enabled": True,
                    "checkpoint": "model.zip",
                    "replay_buffer": "replay.pkl",
                    "refill_steps": 10,
                },
            }
        )


def test_minimum_steps_are_additional_and_plateau_is_optional():
    callback = train_sac.FixedSeedEvalCallback(
        MagicMock(),
        eval_freq=10,
        n_eval_episodes=1,
        seed=0,
        log_path=Path("eval"),
        best_success_model_save_path=Path("best"),
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            min_steps=20,
            success_threshold=1.0,
            patience=1,
            plateau_patience=2,
            min_delta=0.05,
        ),
    )
    callback.training_start_timestep = 1_000_000
    callback.num_timesteps = 1_000_010
    assert callback._update_early_stopping(0.5)
    callback.num_timesteps = 1_000_020
    assert callback._update_early_stopping(0.52)
    callback.num_timesteps = 1_000_030
    assert not callback._update_early_stopping(0.53)
    assert callback.stopping_reason == "validation-plateau"
    assert callback.stop_timestep - callback.training_start_timestep == 30


def test_resume_restores_model_state_and_reports_missing_replay_buffer(tmp_path: Path):
    checkpoint = tmp_path / "best_model.zip"
    checkpoint.write_bytes(b"immutable-source")
    (tmp_path / "manifest.json").write_text("{}")
    config = default_config("sac")
    config = replace(
        config,
        resume=replace(config.resume, enabled=True, checkpoint=str(checkpoint)),
    )
    model = MagicMock(num_timesteps=840_000)
    model.learning_rate = 1e-3
    model.buffer_size = 100
    model.learning_starts = 10
    model.batch_size = 2
    model.gamma = 0.99
    model.tau = 0.005
    model.gradient_steps = 1
    with patch("train_sac.load_expert_manifest", return_value={}), patch(
        "train_sac.validate_expert_checkpoint"
    ) as validate, patch.object(
        train_sac.LegacyScheduleSAC, "load", return_value=model
    ) as load:
        restored, actor, provenance = train_sac.initialize_sac_training(
            config, MagicMock(), tmp_path / "new-run"
        )
    assert restored is model
    assert "continuation" in actor["reason"]
    assert provenance["mode"] == "model-state-only"
    assert provenance["source_timestep"] == 840_000
    assert provenance["reset"] == [
        "replay buffer",
        "random-number-generator state (reseeded for continuation)",
        "environment state",
        "evaluation and checkpoint-selection callback state",
    ]
    assert "optimizer states" in provenance["restored"]
    validate.assert_called_once()
    assert load.call_args.kwargs["env"] is not None
    assert checkpoint.read_bytes() == b"immutable-source"


def test_resume_loads_explicit_replay_buffer(tmp_path: Path):
    checkpoint = tmp_path / "best_model.zip"
    checkpoint.write_bytes(b"source")
    metadata = tmp_path / "best_checkpoint.json"
    metadata.write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    config = default_config("sac")
    config = replace(
        config,
        resume=replace(
            config.resume,
            enabled=True,
            checkpoint=str(checkpoint),
            replay_buffer=str(tmp_path / "best_replay_buffer.pkl"),
        ),
    )
    model = MagicMock(num_timesteps=10)
    model.learning_rate = 1e-3
    model.buffer_size = 100
    model.learning_starts = 10
    model.batch_size = 2
    model.gamma = 0.99
    model.tau = 0.005
    model.gradient_steps = 1
    paired_replay = tmp_path / "best_replay_buffer.pkl"
    paired_replay.write_bytes(b"buffer")
    with patch("train_sac.load_expert_manifest", return_value={}), patch(
        "train_sac.validate_expert_checkpoint"
    ), patch.object(train_sac.LegacyScheduleSAC, "load", return_value=model), patch(
        "train_sac.validate_replay_checkpoint_pair", return_value={"name": "best"}
    ) as validate_pair:
        _, _, provenance = train_sac.initialize_sac_training(
            config, MagicMock(), tmp_path / "new-run"
        )
    validate_pair.assert_called_once()
    assert provenance["mode"] == "model-and-replay"
    assert provenance["reset"] == [
        "random-number-generator state (reseeded for continuation)",
        "environment state",
        "evaluation and checkpoint-selection callback state",
    ]


def test_refill_collects_without_updates_or_parameter_changes():
    class Buffer:
        def __init__(self):
            self.count = 0

        def size(self):
            return self.count

    class Env:
        num_envs = 1

        def reset(self):
            return np.zeros((1, 39), dtype=np.float32)

        def step(self, _actions):
            return (
                np.ones((1, 39), dtype=np.float32),
                np.asarray([1.0]),
                np.asarray([False]),
                [{}],
            )

    actor = torch.nn.Linear(39, 4)
    critic = torch.nn.Linear(43, 1)
    actor_before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in critic.state_dict().items()}
    model = MagicMock()
    model.num_timesteps = 1_000_000
    model._n_updates = 999_000
    model.actor = actor
    model.critic = critic
    model.replay_buffer = Buffer()
    model.get_env.return_value = Env()
    model._sample_action.return_value = (
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
    )

    def store(*_args):
        model.replay_buffer.count += 1

    model._store_transition.side_effect = store
    report = train_sac.refill_resume_replay_buffer(
        model, 5, additional_budget=100
    )
    assert report == {
        "requested_steps": 5,
        "collected_steps": 5,
        "buffer_size_before": 0,
        "buffer_size_after": 5,
        "completed": True,
    }
    assert model.num_timesteps == 1_000_005
    assert model._n_updates == 999_000
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[name])
    for name, value in critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[name])


def test_source_evaluation_does_not_increment_success_counter():
    callback = train_sac.FixedSeedEvalCallback(
        MagicMock(),
        eval_freq=10,
        n_eval_episodes=1,
        seed=0,
        log_path=Path("eval"),
        best_success_model_save_path=Path("best"),
        early_stopping=EarlyStoppingConfig(
            enabled=True, min_steps=0, success_threshold=1.0, patience=1
        ),
    )
    callback.model = MagicMock(num_timesteps=1_000_000)
    callback._logger = MagicMock()
    callback._evaluate = MagicMock(return_value=([1.0], [200], [True]))
    callback._save_evaluations = MagicMock()
    callback._save_model = MagicMock()
    assert callback._record_evaluation(allow_early_stop=False)
    assert callback.consecutive_successful_evaluations == 0
