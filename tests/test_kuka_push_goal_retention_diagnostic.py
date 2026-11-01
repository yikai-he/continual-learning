from types import SimpleNamespace

import numpy as np

from scripts.diagnostics.analyze_kuka_push_goal_retention import (
    EpisodeResult,
    aggregate,
    held_gripper_action,
    make_retraction_target,
    retraction_action,
    workspace_would_clip,
)


def test_stop_holds_current_robotiq_command_and_zero_translation() -> None:
    base = SimpleNamespace(data=SimpleNamespace(ctrl=np.array([191.25])))
    action = np.array([0.0, 0.0, 0.0, held_gripper_action(base)])
    np.testing.assert_allclose(action, [0.0, 0.0, 0.0, 0.5])


def test_retraction_is_away_from_cube_upward_and_workspace_feasible() -> None:
    base = SimpleNamespace(
        tcp_center=np.array([0.0, 0.60, 0.04]),
        mocap_low=np.array([-0.25, 0.42, 0.04]),
        mocap_high=np.array([0.25, 0.75, 0.50]),
        action_scale=0.01,
    )
    target = make_retraction_target(base, np.array([0.0, 0.64, 0.02]))
    np.testing.assert_allclose(target, [0.0, 0.52, 0.10])
    action, complete = retraction_action(base, target)
    np.testing.assert_array_equal(action, [0.0, -1.0, 1.0, -1.0])
    assert not complete


def test_workspace_clipping_detection() -> None:
    base = SimpleNamespace(
        data=SimpleNamespace(mocap_pos=np.array([[0.0, 0.745, 0.10]])),
        mocap_low=np.array([-0.25, 0.42, 0.04]),
        mocap_high=np.array([0.25, 0.75, 0.50]),
        action_scale=0.01,
    )
    assert workspace_would_clip(base, np.array([0.0, 1.0, 0.0, 0.0]))
    assert not workspace_would_clip(base, np.zeros(4))


def test_aggregate_distinguishes_ever_final_and_continuous_success() -> None:
    defaults = dict(
        mode="x", reset_seed=0, episode_length=200, initial_distance=0.2,
        minimum_distance=0.01, first_success_step=10, last_success_step=200,
        successful_steps=100, entries=1, exit_step=None,
        post_success_displacement=0.01, post_success_contact_steps=2,
        first_success_cube_speed=0.1, first_success_goal_velocity=0.1,
        workspace_clipping_events=0, contact_at_success=True,
        contact_released=True, contact_release_step=12,
        retraction_completed=True, retraction_moved_cube=0.01, total_reward=1.0,
    )
    rows = [
        EpisodeResult(task_index=0, final_distance=0.02, ever_success=True,
                      final_success=True, continuous_retention=True, exits=0, **defaults),
        EpisodeResult(task_index=1, final_distance=0.10, ever_success=True,
                      final_success=False, continuous_retention=False, exits=1, **defaults),
    ]
    result = aggregate(rows)
    assert result["ever_success_rate"] == 1.0
    assert result["final_success_rate"] == 0.5
    assert result["continuous_retention_count"] == 1
