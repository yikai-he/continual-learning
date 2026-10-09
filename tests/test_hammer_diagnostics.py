"""Tests for evaluation-only KUKA Hammer diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.hammer_diagnostics import HammerEpisodeTracker, summarize_hammer_episodes
from train_sac import FixedSeedEvalCallback


SCRIPT = Path("scripts/evaluation/compare_kuka_hammer_checkpoints.py")
SPEC = importlib.util.spec_from_file_location("compare_kuka_hammer_checkpoints", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def episode(maximum, final=0.0, contact=False):
    return {
        "final_nail_qpos": final,
        "max_nail_qpos": maximum,
        "progress_fraction": maximum / 0.09,
        "ge_003": maximum >= 0.03,
        "ge_005": maximum >= 0.05,
        "ge_008": maximum >= 0.08,
        "crossed_success_threshold": maximum > 0.09,
        "head_nail_contact": contact,
        "head_nail_contact_steps": int(contact),
        "minimum_proxy_goal_distance": 0.1,
        "minimum_striking_goal_distance": 0.2,
        "mean_proxy_error": 0.03,
    }


def test_threshold_fractions_and_maximum_aggregation() -> None:
    summary = summarize_hammer_episodes(
        [episode(0.02), episode(0.03), episode(0.05), episode(0.08), episode(0.091)]
    )
    assert summary["hammer_fraction_ge_003"] == 0.8
    assert summary["hammer_fraction_ge_005"] == 0.6
    assert summary["hammer_fraction_ge_008"] == 0.4
    assert summary["hammer_fraction_success_threshold"] == 0.2
    assert summary["hammer_best_max_nail_qpos"] == 0.091


def test_tracker_reads_final_nail_and_retains_episode_maximum() -> None:
    qpos = np.array([0.0])
    bodies = {
        "hammer": SimpleNamespace(id=1, xpos=np.array([0.0, 0.0, 0.0])),
        "nail_link": SimpleNamespace(id=2),
    }
    geoms = {
        "HammerHandleCollision": SimpleNamespace(id=0),
        1: SimpleNamespace(xpos=np.array([0.16, 0.056, 0.0])),
    }
    model = SimpleNamespace(
        geom_bodyid=np.array([1, 1, 2]),
        geom_contype=np.array([1, 1, 1]),
        geom_pos=np.array([[0.0, 0.0, 0.0], [0.16, 0.056, 0.0], [0.0, 0.0, 0.0]]),
        body=lambda name: bodies[name],
        geom=lambda name: geoms[name],
    )
    data = SimpleNamespace(
        body=lambda name: bodies[name],
        geom=lambda index: geoms[index],
        joint=lambda _name: SimpleNamespace(qpos=qpos),
        contact=[],
    )
    raw = SimpleNamespace(model=model, data=data, _target_pos=np.array([0.2, 0.2, 0.0]))
    tracker = HammerEpisodeTracker(SimpleNamespace(unwrapped=raw))
    for timestep, value in enumerate((0.01, 0.08, 0.04), start=1):
        qpos[0] = value
        tracker.sample(timestep)
    result = tracker.finish(False)
    assert result.final_nail_qpos == 0.04
    assert result.max_nail_qpos == 0.08
    assert result.progress_fraction == 0.08 / 0.09


def test_non_hammer_evaluation_never_builds_hammer_tracker(tmp_path) -> None:
    env = SimpleNamespace(
        unwrapped=MagicMock(),
        reset=MagicMock(return_value=(np.zeros(39), {})),
        step=MagicMock(return_value=(np.zeros(39), 0.0, True, False, {"success": False})),
    )
    callback = FixedSeedEvalCallback(
        env, eval_freq=1, n_eval_episodes=1, seed=0,
        task_name="kuka-reach-v3", log_path=tmp_path,
        best_model_save_path=tmp_path, best_success_model_save_path=tmp_path,
    )
    callback.model = MagicMock()
    callback.model.predict.return_value = (np.zeros(4), None)
    with patch("train_sac.HammerEpisodeTracker") as tracker:
        callback._evaluate()
    tracker.assert_not_called()
    env.unwrapped.data.joint.assert_not_called()


def test_hammer_callback_collects_tracker_result(tmp_path) -> None:
    env = SimpleNamespace(
        unwrapped=MagicMock(),
        reset=MagicMock(return_value=(np.zeros(39), {})),
        step=MagicMock(return_value=(np.zeros(39), 0.0, True, False, {"success": True})),
    )
    callback = FixedSeedEvalCallback(
        env, eval_freq=1, n_eval_episodes=1, seed=0,
        task_name="kuka-hammer-v3", log_path=tmp_path,
        best_model_save_path=tmp_path, best_success_model_save_path=tmp_path,
    )
    callback.model = MagicMock()
    callback.model.predict.return_value = (np.zeros(4), None)
    tracker = MagicMock()
    tracker.finish.return_value.to_dict.return_value = episode(0.091, 0.091, True)
    with patch("train_sac.HammerEpisodeTracker", return_value=tracker):
        callback._evaluate()
    assert tracker.sample.call_args_list[0].args == (0,)
    assert tracker.sample.call_args_list[-1].args == (1,)
    assert callback._last_hammer_metrics[0]["max_nail_qpos"] == 0.091


def test_checkpoint_comparison_reuses_identical_task_objects(monkeypatch) -> None:
    tasks = [SimpleNamespace(data=f"task-{index}".encode()) for index in range(3)]
    env = SimpleNamespace(
        tasks=tasks,
        toggle_sample_tasks_on_reset=MagicMock(),
        close=MagicMock(),
        observation_space=MagicMock(),
        action_space=MagicMock(),
    )
    env.unwrapped = env
    models = [
        SimpleNamespace(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        for _ in range(2)
    ]
    calls = []

    def evaluate(_model, _env, selected, seeds):
        calls.append((selected, seeds))
        return {
            "success_rate": 0.0,
            "mean_return": 0.0,
            "hammer_mean_final_nail_qpos": 0.0,
            "hammer_mean_max_nail_qpos": 0.0,
            "hammer_median_max_nail_qpos": 0.0,
            "hammer_best_max_nail_qpos": 0.0,
            "hammer_fraction_ge_005": 0.0,
            "hammer_fraction_ge_008": 0.0,
            "hammer_fraction_success_threshold": 0.0,
            "hammer_head_nail_contact_fraction": 0.0,
        }

    monkeypatch.setattr(comparison, "make_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(comparison.SAC, "load", MagicMock(side_effect=models))
    monkeypatch.setattr(comparison, "evaluate_checkpoint", evaluate)
    comparison.main(["first.zip", "second.zip", "--episodes", "3"])
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    assert calls[0][1] is calls[1][1]
    assert calls[0][0] == tuple(tasks)
