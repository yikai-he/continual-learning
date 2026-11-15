"""Standalone SAC is a fresh, success-selected, low-disk workflow."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import train_sac
from src.config import config_from_dict


def test_removed_checkpoint_configuration_is_rejected() -> None:
    for field, value in (("checkpoint_mode", "full"), ("checkpoint_freq", 10_000)):
        with pytest.raises(ValueError, match=field):
            config_from_dict(
                {
                    "experiment": "sac",
                    "runtime": {"output": "/tmp/run"},
                    "sac": {field: value},
                }
            )


def test_resume_cli_is_not_supported() -> None:
    with pytest.raises(SystemExit):
        train_sac.parse_args(["--resume-from", "checkpoint.zip"])


def test_global_evaluation_boundaries_are_preserved() -> None:
    assert train_sac.next_global_boundary(0, 20_000) == 20_000
    assert train_sac.next_global_boundary(505_000, 20_000) == 520_000


def test_fresh_run_evaluates_at_zero_and_schedules_first_boundary() -> None:
    callback = train_sac.FixedSeedEvalCallback(
        MagicMock(),
        eval_freq=20_000,
        n_eval_episodes=1,
        seed=0,
        log_path=Path("evaluations"),
        best_success_model_save_path=Path("best-success"),
    )
    callback.model = MagicMock(num_timesteps=0)
    callback._record_evaluation = MagicMock()
    callback._on_training_start()
    callback._record_evaluation.assert_called_once_with(allow_early_stop=False)
    assert callback._next_evaluation_timestep == 20_000


def test_atomic_selected_expert_replacement() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best_model.zip"

        def save(target):
            Path(target).write_bytes(b"new-model")

        model = MagicMock()
        model.save.side_effect = save
        train_sac.atomic_replace_model(model, path)
        assert path.read_bytes() == b"new-model"
        assert list(Path(tmp).glob("*.tmp.zip")) == []
