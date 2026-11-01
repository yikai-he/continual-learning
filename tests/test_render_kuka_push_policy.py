"""Read-only KUKA push checkpoint replay helpers."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


SCRIPT = Path("scripts/render_kuka_push_policy.py")
SPEC = importlib.util.spec_from_file_location("render_kuka_push_policy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_checkpoint_aliases_and_seed_list() -> None:
    args = replay.parse_args(
        ["--checkpoint", "best-return", "--seeds", "10003", "10019"]
    )
    assert args.checkpoint == "best-return"
    assert args.seeds == [10003, 10019]
    assert args.control_hz == 20.0
    assert args.episodes == 1
    assert args.window_size == (1280, 900)
    assert args.env_name == "kuka-push-v3"
    assert args.reward_version == "v2"


@pytest.mark.parametrize(
    "reward_version", ("v1", "v2", "v3", "v3_1", "v3_2_retention")
)
def test_reward_version_argument_accepts_supported_values(reward_version) -> None:
    args = replay.parse_args(["--reward-version", reward_version])
    assert args.reward_version == reward_version


def test_reward_version_argument_rejects_unknown_value() -> None:
    with pytest.raises(SystemExit):
        replay.parse_args(["--reward-version", "unknown"])


def test_v3_3_reward_argument_is_cube_only() -> None:
    args = replay.parse_args(
        [
            "--env-name",
            "kuka-push-cube-v3",
            "--reward-version",
            "v3_3_contact_push_progress",
        ]
    )
    assert args.reward_version == "v3_3_contact_push_progress"
    with pytest.raises(SystemExit):
        replay.parse_args(["--reward-version", "v3_3_contact_push_progress"])


def test_v4_reward_argument_is_cube_only() -> None:
    args = replay.parse_args(
        [
            "--env-name",
            "kuka-push-cube-v3",
            "--reward-version",
            "v4_simple_approach_progress",
        ]
    )
    assert args.reward_version == "v4_simple_approach_progress"
    with pytest.raises(SystemExit):
        replay.parse_args(["--reward-version", "v4_simple_approach_progress"])


def test_v5_contact_start_arguments_are_cube_only() -> None:
    args = replay.parse_args([
        "--env-name", "kuka-push-cube-v3",
        "--reward-version", "v5_contact_start_progress",
        "--cube-push-curriculum-level", "2",
        "--cube-push-initialization", "contact",
    ])
    assert args.reward_version == "v5_contact_start_progress"
    assert args.cube_push_initialization == "contact"
    with pytest.raises(SystemExit):
        replay.parse_args(["--cube-push-initialization", "contact"])


def test_near_contact_common_distribution_arguments() -> None:
    args = replay.parse_args([
        "--env-name", "kuka-push-cube-v3",
        "--reward-version", "v6_rear_contact_push_progress",
        "--cube-push-curriculum-level", "2",
        "--cube-push-initialization", "near_contact",
        "--cube-y-min", "0.502",
        "--near-contact-gap-m", "0.005",
    ])
    assert args.cube_push_initialization == "near_contact"
    assert args.cube_y_min == pytest.approx(0.502)
    assert args.near_contact_gap_m == pytest.approx(0.005)


def test_stage1_reward_requires_level2_default_cube_environment() -> None:
    args = replay.parse_args([
        "--env-name", "kuka-push-cube-v3",
        "--reward-version", "v1_initial_to_contact",
        "--cube-push-curriculum-level", "2",
        "--cube-push-initialization", "default",
    ])
    assert args.reward_version == "v1_initial_to_contact"
    with pytest.raises(SystemExit):
        replay.parse_args([
            "--env-name", "kuka-push-cube-v3",
            "--reward-version", "v1_initial_to_contact",
            "--cube-push-curriculum-level", "3",
        ])


def test_stage1_v1_1_reward_requires_level2_default_cube_environment() -> None:
    args = replay.parse_args([
        "--env-name", "kuka-push-cube-v3",
        "--reward-version", "v1_1_initial_to_contact_aperture",
        "--cube-push-curriculum-level", "2",
        "--cube-push-initialization", "default",
    ])
    assert args.reward_version == "v1_1_initial_to_contact_aperture"
    with pytest.raises(SystemExit):
        replay.parse_args([
            "--env-name", "kuka-push-cube-v3",
            "--reward-version", "v1_1_initial_to_contact_aperture",
            "--cube-push-initialization", "contact",
        ])


def test_sawyer_adapted_reward_argument_is_cube_only() -> None:
    args = replay.parse_args(["--env-name", "kuka-push-cube-v3",
        "--reward-version", "sawyer_push_v2_adapted"])
    assert args.reward_version == "sawyer_push_v2_adapted"
    with pytest.raises(SystemExit):
        replay.parse_args(["--reward-version", "sawyer_push_v2_adapted"])


def test_environment_name_can_select_handle_press_side() -> None:
    args = replay.parse_args(
        ["--model", "model.zip", "--env-name", "kuka-handle-press-side-v3"]
    )
    assert args.env_name == "kuka-handle-press-side-v3"


def test_push_audit_case_rejects_another_environment() -> None:
    with pytest.raises(SystemExit):
        replay.parse_args(
            ["--audit-case", "45k-success", "--env-name", "kuka-handle-press-side-v3"]
        )


def test_requested_window_size_parsing_and_validation() -> None:
    assert replay.parse_args(["--window-size", "1200", "800"]).window_size == (
        1200,
        800,
    )
    with pytest.raises(SystemExit):
        replay.parse_args(["--window-size", "0", "900"])


def test_viewer_presentation_defaults_hud_cycle_and_help_toggle() -> None:
    state = replay.ViewerPresentation()
    assert (state.hud_mode, state.show_help, state.show_builtin_menu) == (
        "compact",
        False,
        False,
    )
    state.cycle_hud()
    assert state.hud_mode == "verbose"
    state.cycle_hud()
    assert state.hud_mode == "hidden"
    state.cycle_hud()
    assert state.hud_mode == "compact"
    state.toggle_help()
    assert state.show_help is True
    state.toggle_help()
    assert state.show_help is False


def test_audit_case_resolves_exact_checkpoint_and_task_identity() -> None:
    args = replay.parse_args(["--audit-case", "45k-success"])
    case = replay.AUDIT_CASES[args.audit_case]
    assert case == {
        "checkpoint": "best-success",
        "seed": 10003,
        "task_index": 29,
        "task_sha256": "1921052616856f4dd35a4913cc99c51a72d463e37f4c7851ae394aa53d4287cf",
    }


def test_seed_without_task_identity_is_rejected() -> None:
    args = replay.parse_args(["--seeds", "10003"])
    wrapper = SimpleNamespace(tasks=[])
    with pytest.raises(ValueError, match="reset seed alone"):
        replay.resolve_requests(args, wrapper)


def test_arbitrary_model_is_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        replay.parse_args(["--checkpoint", "final", "--model", "model.zip"])


@pytest.mark.parametrize(
    ("model_obs", "model_action", "env_obs", "env_action"),
    [((38,), (4,), (39,), (4,)), ((39,), (3,), (39,), (4,))],
)
def test_space_validation_rejects_wrong_shapes(
    model_obs, model_action, env_obs, env_action
) -> None:
    model = SimpleNamespace(
        observation_space=MagicMock(shape=model_obs),
        action_space=MagicMock(shape=model_action),
    )
    env = SimpleNamespace(
        observation_space=MagicMock(shape=env_obs),
        action_space=MagicMock(shape=env_action),
    )
    with pytest.raises(ValueError, match="shape"):
        replay.validate_spaces(model, env)


def test_source_loads_only_policy_parameters_and_predicts_deterministically() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SAC.load(model_path" in source
    assert "load_replay_buffer" not in source
    assert "model.learn" not in source
    assert "deterministic=True" in source
    assert "except KeyboardInterrupt" in source


def _run_args(**updates):
    values = {
        "device": "cpu",
        "env_name": "kuka-push-v3",
        "reward_version": "v2",
        "task_set_seed": 10_000,
        "smoke_test": 1,
        "output_video": None,
        "window_size": (8, 6),
        "control_hz": 20.0,
        "overwrite_video": False,
        "no_loop": True,
        "print_every": 20,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _run_dependencies(env, model):
    return (
        patch.object(replay, "resolve_model", return_value=(Path("model.zip"), "test")),
        patch.object(replay.SAC, "load", return_value=model),
        patch.object(replay, "make_env", return_value=env),
        patch.object(replay, "validate_spaces"),
        patch.object(replay, "find_task_wrapper", return_value=MagicMock()),
        patch.object(
            replay,
            "resolve_requests",
            return_value=[replay.TaskRequest(10_000, 0, "hash", "test")],
        ),
        patch.object(
            replay,
            "reset_exact_task",
            return_value=(replay.np.zeros(39), replay.np.zeros(3), replay.np.ones(3)),
        ),
    )


def test_environment_is_closed_after_normal_completion() -> None:
    env = MagicMock()
    env.step.return_value = (replay.np.zeros(39), 0.0, False, False, {})
    model = MagicMock()
    model.predict.return_value = (replay.np.zeros(4), None)
    dependencies = _run_dependencies(env, model)
    with dependencies[0], dependencies[1], dependencies[2], dependencies[3], \
        dependencies[4], dependencies[5], dependencies[6]:
        replay.run(_run_args())
    env.close.assert_called_once_with()


def test_selected_reward_version_reaches_environment_factory() -> None:
    env = MagicMock()
    env.step.return_value = (replay.np.zeros(39), 0.0, False, True, {})
    model = MagicMock()
    model.predict.return_value = (replay.np.zeros(4), None)
    dependencies = _run_dependencies(env, model)
    with dependencies[0], dependencies[1], dependencies[2] as make_env, \
        dependencies[3], dependencies[4], dependencies[5], dependencies[6]:
        replay.run(_run_args(reward_version="v3_1"))
    assert make_env.call_args.kwargs["reward_function_version"] == "v3_1"


def test_retention_reward_version_reaches_environment_factory() -> None:
    env = MagicMock()
    env.step.return_value = (replay.np.zeros(39), 0.0, False, True, {})
    model = MagicMock()
    model.predict.return_value = (replay.np.zeros(4), None)
    dependencies = _run_dependencies(env, model)
    with dependencies[0], dependencies[1], dependencies[2] as make_env, \
        dependencies[3], dependencies[4], dependencies[5], dependencies[6]:
        replay.run(_run_args(reward_version="v3_2_retention"))
    assert (
        make_env.call_args.kwargs["reward_function_version"]
        == "v3_2_retention"
    )


def test_v3_3_cube_reward_version_reaches_environment_factory() -> None:
    env = MagicMock()
    env.step.return_value = (replay.np.zeros(39), 0.0, False, True, {})
    model = MagicMock()
    model.predict.return_value = (replay.np.zeros(4), None)
    dependencies = _run_dependencies(env, model)
    with dependencies[0], dependencies[1], dependencies[2] as make_env, \
        dependencies[3], dependencies[4], dependencies[5], dependencies[6]:
        replay.run(
            _run_args(
                env_name="kuka-push-cube-v3",
                reward_version="v3_3_contact_push_progress",
            )
        )
    assert make_env.call_args.args[1] == "kuka-push-cube-v3"
    assert (
        make_env.call_args.kwargs["reward_function_version"]
        == "v3_3_contact_push_progress"
    )


def test_sawyer_adapted_reward_reaches_cube_environment_factory() -> None:
    env = MagicMock(); env.step.return_value=(replay.np.zeros(39),0.0,False,True,{})
    model = MagicMock(); model.predict.return_value=(replay.np.zeros(4),None)
    dependencies = _run_dependencies(env, model)
    with dependencies[0], dependencies[1], dependencies[2] as make_env, \
        dependencies[3], dependencies[4], dependencies[5], dependencies[6]:
        replay.run(_run_args(env_name="kuka-push-cube-v3",
            reward_version="sawyer_push_v2_adapted"))
    assert make_env.call_args.args[1] == "kuka-push-cube-v3"
    assert make_env.call_args.kwargs["reward_function_version"] == "sawyer_push_v2_adapted"


def test_contact_display_prefers_v3_3_physical_contact_diagnostic() -> None:
    base = MagicMock()
    assert replay.object_contact(
        base,
        {"physical_robot_object_contact": True, "named_pad_contact": False},
        replay.np.zeros(39),
    )
    base.pad_contacting_object.assert_not_called()


def test_environment_is_closed_when_rendering_fails() -> None:
    env = MagicMock()
    env.render.side_effect = RuntimeError("render failure")
    model = MagicMock()
    dependencies = _run_dependencies(env, model)
    with dependencies[0], dependencies[1], dependencies[2], dependencies[3], \
        dependencies[4], dependencies[5], dependencies[6], pytest.raises(
            RuntimeError, match="render failure"
        ):
        replay.run(_run_args(smoke_test=None))
    env.close.assert_called_once_with()


def _video_environment():
    camera = SimpleNamespace(
        type=0,
        fixedcamid=-1,
        trackbodyid=-1,
        lookat=replay.np.zeros(3),
        distance=1.0,
        azimuth=0.0,
        elevation=0.0,
    )
    viewer = SimpleNamespace(cam=camera, window=None)
    renderer = SimpleNamespace(width=0, height=0, viewer=viewer)
    base = SimpleNamespace(
        mujoco_renderer=renderer,
        model=SimpleNamespace(vis=SimpleNamespace(global_=SimpleNamespace())),
    )
    env = MagicMock()
    env.unwrapped = base
    frame = replay.np.zeros((6, 8, 3), dtype=replay.np.uint8)
    env.render.return_value = frame
    observation = replay.np.zeros(39)
    observation[36:39] = 1.0
    env.step.return_value = (observation, 0.0, False, True, {})
    return env


def _run_video(env, model, writer) -> None:
    dependencies = _run_dependencies(env, model)
    with dependencies[0], dependencies[1], dependencies[2], dependencies[3], \
        dependencies[4], dependencies[5], dependencies[6], patch.object(
            replay, "configure_camera", return_value="test"
        ), patch.object(replay, "workspace_would_clip", return_value=False), \
        patch.object(replay, "object_contact", return_value=False), patch.object(
            replay, "add_object_marker"
        ), patch.object(replay, "Mp4Writer", return_value=writer):
        replay.run(
            _run_args(smoke_test=None, output_video=Path("output.mp4"))
        )


def test_environment_is_closed_when_video_writing_fails() -> None:
    env = _video_environment()
    model = MagicMock()
    model.predict.return_value = (replay.np.zeros(4), None)
    writer = MagicMock()
    writer.write.side_effect = RuntimeError("write failure")
    with pytest.raises(RuntimeError, match="write failure"):
        _run_video(env, model, writer)
    writer.close.assert_called_once_with()
    env.close.assert_called_once_with()


def test_environment_is_closed_when_video_writer_close_fails() -> None:
    env = _video_environment()
    model = MagicMock()
    model.predict.return_value = (replay.np.zeros(4), None)
    writer = MagicMock()
    writer.close.side_effect = RuntimeError("close failure")
    with pytest.raises(RuntimeError, match="close failure"):
        _run_video(env, model, writer)
    env.close.assert_called_once_with()


def test_loop_restart_next_and_freeze_state_logic() -> None:
    command = replay.ReplayCommand()
    assert replay.advance_after_episode(auto_loop=True, command=command) == "restart"
    assert replay.advance_after_episode(auto_loop=False, command=command) == "freeze"
    command.restart = True
    assert replay.advance_after_episode(auto_loop=False, command=command) == "restart"
    assert command.restart is False
    command.next_seed = True
    assert replay.advance_after_episode(auto_loop=True, command=command) == "next"
    assert command.next_seed is False
    command.quit = True
    assert replay.advance_after_episode(auto_loop=True, command=command) == "quit"


def test_contact_falls_back_for_task_without_named_object_geometry() -> None:
    base = SimpleNamespace(
        _get_id_main_object=MagicMock(side_effect=KeyError("objGeom")),
        pad_contacting_object=MagicMock(),
    )
    observation = replay.np.ones(39)
    assert replay.object_contact(base, {"near_object": 1.0}, observation) is True
    assert replay.object_contact(base, {"near_object": 0.0}, observation) is False


def test_object_marker_is_optional() -> None:
    base = SimpleNamespace(data=SimpleNamespace(geom=MagicMock(side_effect=KeyError)))
    viewer = SimpleNamespace(add_marker=MagicMock())
    replay.add_object_marker(viewer, base)
    viewer.add_marker.assert_not_called()


def test_restart_reapplies_identical_serialized_task_and_reports_identity(capsys) -> None:
    data = pickle.dumps({"rand_vec": [0.1, 0.5, 0.02, -0.1, 0.7, 0.01]})
    task = SimpleNamespace(data=data)
    digest = replay.task_hash(task)

    class Base:
        _target_pos = [0.0, 0.7, 0.02]

        def __init__(self):
            self.selected = []

        def set_task(self, selected):
            self.selected.append(selected)

    class Env:
        def __init__(self):
            self.unwrapped = Base()

        def reset(self, seed=None):
            observation = [0.0] * 39
            observation[4:7] = [0.0, 0.5, 0.02]
            observation[36:39] = [0.0, 0.7, 0.02]
            return replay.np.asarray(observation), {}

    wrapper = SimpleNamespace(
        tasks=[task], toggle_sample_tasks_on_reset=MagicMock()
    )
    request = replay.TaskRequest(10003, 0, digest, "exact")
    env = Env()
    first = replay.reset_exact_task(env, wrapper, request)[0]
    second = replay.reset_exact_task(env, wrapper, request)[0]
    replay.np.testing.assert_array_equal(first, second)
    assert env.unwrapped.selected == [task, task]
    assert wrapper.toggle_sample_tasks_on_reset.call_count == 2
    output = capsys.readouterr().out
    assert "requested_seed=10003" in output
    assert "task_index=0" in output
    assert digest in output
    assert "tcp=[" in output
    assert "object=[" in output
    assert "goal=[" in output
    assert "task_vector=[" in output


def test_compact_hud_is_five_short_lines_without_coordinates_or_hash() -> None:
    request = replay.TaskRequest(10003, 29, "secret-sha", "45k-success")
    state = {
        "step": 11,
        "reward": 0.078,
        "return": 0.669,
        "success": False,
        "contact": False,
        "distance": 0.2116,
        "signed": 0.0,
        "max_progress": 0.0123,
        "clip": True,
        "contact_steps": 0,
        "progress": 0.001,
        "min_distance": 0.2,
        "max_displacement": 0.03,
    }
    lines = replay.hud_lines(
        label="best-success",
        request=request,
        state=state,
        paused=False,
        auto_loop=True,
        camera="close-up",
        verbose=False,
    )
    assert lines == [
        "best-success | task 29 | loop ON",
        "11/200 | R +0.08 | Sum 0.7",
        "Success NO | Contact NO | Clip YES",
        "Goal 0.2116 m",
        "dProg +0.0000 | Max +0.0123",
    ]
    assert "secret-sha" not in "\n".join(lines)
    assert "tcp" not in "\n".join(lines).lower()


def test_hammer_debug_info_reports_nail_progress() -> None:
    joint = SimpleNamespace(qpos=replay.np.array([0.034]))
    base = SimpleNamespace(data=SimpleNamespace(joint=MagicMock(return_value=joint)))

    lines = replay.get_task_specific_debug_info("kuka-hammer-v3", base)

    assert lines == ("Nail 0.034 / 0.090 m (38%)",)
    base.data.joint.assert_called_once_with("NailSlideJoint")


@pytest.mark.parametrize(
    "env_name",
    ("kuka-reach-v3", "kuka-push-v3", "kuka-button-press-v3"),
)
def test_non_hammer_debug_info_does_not_access_nail_joint(env_name) -> None:
    base = SimpleNamespace(data=SimpleNamespace(joint=MagicMock()))

    assert replay.get_task_specific_debug_info(env_name, base) == ()
    base.data.joint.assert_not_called()


def test_verbose_hud_adds_diagnostics_and_controls_stay_compact() -> None:
    request = replay.TaskRequest(7, 3, "sha", "task-3")
    state = {
        "step": 1,
        "reward": 0.0,
        "return": 0.0,
        "success": False,
        "contact": True,
        "distance": 0.1,
        "signed": 0.01,
        "max_progress": 0.02,
        "clip": False,
        "contact_steps": 1,
        "progress": 0.02,
        "min_distance": 0.09,
        "max_displacement": 0.03,
    }
    lines = replay.hud_lines(
        label="final",
        request=request,
        state=state,
        paused=True,
        auto_loop=False,
        camera="side",
        verbose=True,
    )
    assert len(lines) > 5
    assert "State PAUSED | Camera side" in lines
    assert "Contact steps 1" in lines
    assert replay.control_lines() == [
        "Space pause | R restart | N next | L loop",
        "I HUD | H MuJoCo menu | Q quit | 1-5 camera",
        "? / F1 help",
    ]


def test_window_configuration_falls_back_and_reports_actual_sizes(
    monkeypatch, capsys
) -> None:
    sizes = iter([(640, 480), (640, 480), (1280, 900)])
    monkeypatch.setattr(replay.glfw, "get_window_size", lambda window: next(sizes))
    monkeypatch.setattr(replay.glfw, "maximize_window", MagicMock())
    monkeypatch.setattr(replay.glfw, "poll_events", MagicMock())
    monkeypatch.setattr(replay.glfw, "get_window_attrib", lambda window, attr: 0)
    resize = MagicMock()
    monkeypatch.setattr(replay.glfw, "set_window_size", resize)
    monkeypatch.setattr(
        replay.glfw, "get_framebuffer_size", lambda window: (2560, 1800)
    )

    actual, framebuffer = replay.configure_viewer_window(object(), (1280, 900))

    resize.assert_called_once()
    assert resize.call_args.args[1:] == (1280, 900)
    assert actual == (1280, 900)
    assert framebuffer == (2560, 1800)
    assert "requested=1280x900" in capsys.readouterr().out
