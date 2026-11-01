"""Behavioral baselines for SAC evaluation diagnostics and checkpoint ranking."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from train_sac import FixedSeedEvalCallback
from src.sac.checkpoint_ranking import best_return_index, best_success_index
from src.support.expert_manifest import qualification_record


def _observation(*, obj=(0.0, 0.0, 0.0), tcp=(0.0, 0.0, 0.0), goal=(0.1, 0.0, 0.0)):
    value = np.zeros(39, dtype=np.float64)
    value[:3] = tcp
    value[3] = 1.0
    value[4:7] = obj
    value[36:39] = goal
    return value


class ScriptedEnv:
    def __init__(self, initial, steps, *, reward_version="v2"):
        self.unwrapped = self
        self.initial = initial
        self.steps = steps
        self.reward_function_version = reward_version
        self.index = 0
        self.mocap_low = np.array([-1.0, -1.0, -1.0])
        self.mocap_high = np.array([1.0, 1.0, 1.0])
        self.data = SimpleNamespace(mocap_pos=np.zeros((1, 3)))

    def reset(self, seed=None):
        self.index = 0
        return self.initial.copy(), {}

    def step(self, _action):
        value = self.steps[self.index]
        self.index += 1
        return value

    def pad_contacting_object(self, _object_id):
        return False

    def _get_id_main_object(self):
        return 1

    @staticmethod
    def _stage1_approach_target(obj, goal):
        del goal
        return np.asarray(obj, dtype=float)


def _callback(env, tmp_path: Path, task_name=None):
    callback = FixedSeedEvalCallback(
        env,
        eval_freq=1,
        n_eval_episodes=1,
        seed=17,
        log_path=tmp_path,
        best_model_save_path=tmp_path / "best",
        best_success_model_save_path=tmp_path / "best-success",
        task_name=task_name,
    )
    callback.model = MagicMock()
    callback.model.predict.return_value = (np.array([0.1, -0.2, 0.3, 0.4]), None)
    return callback


def test_generic_metaworld_evaluation_baseline(tmp_path):
    env = ScriptedEnv(
        _observation(),
        [
            (_observation(), 1.25, False, False, {"success": False}),
            (_observation(), -0.25, False, True, {"success": True}),
        ],
    )
    callback = _callback(env, tmp_path, "reach-v3")

    assert callback._evaluate() == ([1.0], [2], [True])
    assert callback._last_behavior_metrics == []
    assert callback._last_hammer_metrics == []
    assert callback._last_handoff_metrics == []
    assert callback._last_success_diagnostics == [
        {
            "seed": 17,
            "any_step_success": True,
            "final_step_success": True,
            "sustained_success": False,
            "longest_success_run": 1,
            "first_success_step": 1,
            "success_retention_steps": 1,
            "success_retention_fraction": 1.0,
        }
    ]


def test_kuka_push_v3_1_diagnostic_baseline(tmp_path):
    first = {
        "success": False,
        "near_object": True,
        "reward_v3_1_caging": 0.2,
        "reward_v3_1_contact": True,
        "reward_v3_1_signed_progress": 0.01,
        "reward_v3_1_clipped_progress": 0.005,
        "reward_v3_1_first_success": False,
        "reward_v3_1_total": 5.05,
    }
    second = {**first, "success": True, "reward_v3_1_first_success": True}
    env = ScriptedEnv(
        _observation(obj=(0.0, 0.0, 0.0)),
        [
            (_observation(obj=(0.01, 0.0, 0.0)), 5.05, False, False, first),
            (_observation(obj=(0.02, 0.0, 0.0)), 100.0, False, True, second),
        ],
        reward_version="v3_1",
    )
    callback = _callback(env, tmp_path, "kuka-push-v3")

    assert callback._evaluate() == ([105.05], [2], [True])
    metric = callback._last_behavior_metrics[0]
    assert list(metric) == [
        "seed", "success", "any_step_success", "final_step_success",
        "sustained_success", "longest_success_run", "first_success_step",
        "success_retention_steps", "success_retention_fraction", "category",
        "initial_goal_distance", "maximum_goal_progress", "minimum_goal_distance",
        "final_goal_distance", "maximum_puck_displacement", "contact_rate",
        "stationary_contact_rate", "workspace_clipped_contact_without_progress_rate",
        "wrong_direction_step_rate", "correct_direction_step_rate",
        "signed_progress_contribution", "caging_contact_baseline_contribution",
        "pre_contact_caging_contribution", "contact_baseline_contribution",
        "success_bonus_occurrences", "retention_reward_contribution",
        "reward_approach_contribution", "reward_first_contact_contribution",
        "reward_push_contribution", "reward_goal_progress_contribution",
        "reward_first_success_contribution", "v4_reward_approach_contribution",
        "v4_reward_progress_contribution", "v4_reward_first_success_contribution",
        "v5_reward_progress_contribution", "v5_reward_first_success_contribution",
        "sawyer_adapted_reward_base_contribution",
        "sawyer_adapted_reward_near_object_contribution",
        "sawyer_adapted_reward_success_override_steps_contribution",
    ]
    assert metric["category"] == "overshoot"
    assert metric["contact_rate"] == 1.0
    assert metric["signed_progress_contribution"] == 10.0
    assert metric["success_bonus_occurrences"] == 1
    callback.timesteps = [100]
    callback.behavior_metrics = [[metric]]
    callback.success_diagnostics = [callback._last_success_diagnostics]
    callback._save_evaluations()
    persisted = json.loads((tmp_path / "behavior_metrics.json").read_text())
    assert list(persisted[0]) == ["step", "episodes"]
    assert list(persisted[0]["episodes"][0]) == list(metric)


def test_kuka_cube_v3_3_diagnostic_baseline(tmp_path):
    info = {
        "success": True,
        "near_object": True,
        "physical_robot_object_contact": True,
        "signed_goal_progress": 0.02,
        "reward_approach": 0.1,
        "reward_first_contact": 1.0,
        "reward_push": 2.0,
        "reward_goal_progress": 3.0,
        "reward_first_success": 4.0,
        "reward_total": 10.1,
    }
    env = ScriptedEnv(
        _observation(obj=(0.0, 0.0, 0.0)),
        [(_observation(obj=(0.02, 0.0, 0.0)), 10.1, False, True, info)],
        reward_version="v3_3_contact_push_progress",
    )
    callback = _callback(env, tmp_path, "kuka-push-cube-v3")

    assert callback._evaluate() == ([10.1], [1], [True])
    metric = callback._last_behavior_metrics[0]
    assert metric["reward_approach_contribution"] == 0.1
    assert metric["reward_first_contact_contribution"] == 1.0
    assert metric["reward_push_contribution"] == 2.0
    assert metric["reward_goal_progress_contribution"] == 3.0
    assert metric["reward_first_success_contribution"] == 4.0
    assert metric["caging_contact_baseline_contribution"] == 3.1


def test_stage1_handoff_diagnostic_baseline(tmp_path):
    info = {
        "success": True,
        "handoff_instantaneous": True,
        "handoff_sustained": True,
        "handoff_stability_counter": 10,
        "handoff_valid_rear_contact": True,
        "reward_approach": 0.5,
        "reward_tcp_shaping": 0.3,
        "reward_aperture_shaping": 0.2,
        "reward_handoff": 5.0,
        "potential_tcp_current": -0.01,
        "potential_aperture_current": -0.02,
        "handoff_violation_reasons": ["tilt"],
        "handoff_gripper_aperture": 0.08,
        "handoff_cube_displacement": 0.004,
        "handoff_cube_tilt": 0.1,
        "handoff_forward_offset": 0.01,
        "handoff_lateral_offset": 0.02,
        "handoff_vertical_offset": 0.03,
        "handoff_contact_loss_events": 2,
    }
    env = ScriptedEnv(
        _observation(obj=(0.0, 0.0, 0.0), tcp=(0.02, 0.0, 0.0)),
        [
            (
                _observation(obj=(0.001, 0.0, 0.0), tcp=(0.011, 0.0, 0.0)),
                5.5,
                False,
                True,
                info,
            )
        ],
        reward_version="v1_1_initial_to_contact_aperture",
    )
    callback = _callback(env, tmp_path, "kuka-push-cube-v3")

    assert callback._evaluate() == ([5.5], [1], [True])
    metric = callback._last_handoff_metrics[0]
    assert list(metric)[0:8] == [
        "seed", "instantaneous_success", "sustained_success",
        "first_instantaneous_step", "success_step", "maximum_stability_counter",
        "valid_rear_contact_fraction", "valid_rear_contact_episode",
    ]
    assert metric["reward_tcp_shaping_total"] == 0.3
    assert metric["reward_aperture_shaping_total"] == 0.2
    assert metric["discounted_episode_return_gamma_0_99"] == 5.5
    assert metric["contract_violation_counts"] == {"tilt": 1}


def test_hammer_diagnostic_and_tensorboard_baseline(tmp_path):
    env = ScriptedEnv(
        _observation(),
        [(_observation(), 2.0, False, True, {"success": True})],
    )
    finished = {
        "final_nail_qpos": 0.1,
        "max_nail_qpos": 0.11,
        "progress_fraction": 1.2,
        "ge_003": True,
        "ge_005": True,
        "ge_008": True,
        "crossed_success_threshold": True,
        "success": True,
        "minimum_proxy_goal_distance": 0.2,
        "minimum_striking_goal_distance": 0.1,
        "mean_proxy_error": 0.03,
        "final_proxy_position": [0.0, 0.0, 0.0],
        "final_striking_position": [0.0, 0.0, 0.0],
        "head_nail_contact": True,
        "head_nail_contact_steps": 2,
        "first_head_nail_contact_timestep": 1,
    }
    tracker = MagicMock()
    tracker.finish.return_value.to_dict.return_value = finished
    callback = _callback(env, tmp_path, "kuka-hammer-v3")
    callback._logger = MagicMock()
    callback.num_timesteps = 100

    with patch("train_sac.HammerEpisodeTracker", return_value=tracker):
        callback._record_evaluation(allow_early_stop=False)

    assert callback.hammer_metrics == [[finished]]
    tags = {call.args[0]: call.args[1] for call in callback.logger.record.call_args_list}
    assert tags["eval/mean_reward"] == 2.0
    assert tags["eval/std_reward"] == 0.0
    assert tags["eval/success_rate"] == 1.0
    assert tags["eval/hammer_mean_max_nail_qpos"] == 0.11
    assert tags["eval/hammer_head_nail_contact_fraction"] == 1.0


@pytest.mark.parametrize(
    "evaluations, expected_saves",
    [
        (
            [([10.0], [1], [True]), ([10.0], [1], [True])],
            ["best/best_model", "best-success/best_success_model"],
        ),
        (
            [([10.0], [1], [False]), ([9.0], [1], [True])],
            ["best/best_model", "best-success/best_success_model", "best-success/best_success_model"],
        ),
    ],
)
def test_checkpoint_selection_and_ties(tmp_path, evaluations, expected_saves):
    callback = _callback(MagicMock(), tmp_path)
    callback._logger = MagicMock()
    callback._evaluate = MagicMock(side_effect=evaluations)
    for step in range(len(evaluations)):
        callback.num_timesteps = step
        callback._record_evaluation(allow_early_stop=False)
    assert [str(call.args[0].relative_to(tmp_path)) for call in callback.model.save.call_args_list] == expected_saves


def test_export_ranking_ties_and_nan_behavior_are_characterized():
    assert best_return_index([2.0, 2.0, 1.0]) == 0
    assert best_success_index([0.5, 0.5, 0.4], [2.0, 2.0, 100.0]) == 0
    assert best_success_index([0.5, 0.5], [1.0, 2.0]) == 1
    assert best_return_index([1.0, np.nan, 3.0]) == 1
    assert best_success_index([0.5, np.nan, 1.0], [1.0, 100.0, 0.0]) == 2


def test_expert_qualification_metadata_baseline():
    value = qualification_record(
        success_rate=0.75,
        mean_return=12.5,
        return_std=1.25,
        episodes=4,
        task_set_seed=10_000,
        evaluation_seed=20_000,
        deterministic=True,
        task_hashes=["a", "b", "c", "d"],
    )
    assert value == {
        "protocol": "fixed-mt1-task-bank-v1",
        "success_rate": 0.75,
        "mean_return": 12.5,
        "return_std": 1.25,
        "episodes": 4,
        "task_set_seed": 10_000,
        "evaluation_seed": 20_000,
        "deterministic": True,
        "ordered_task_data_sha256": ["a", "b", "c", "d"],
        "selection_rule": "maximum success_rate; mean_return breaks ties",
        "selection_key": [0.75, 12.5],
    }
