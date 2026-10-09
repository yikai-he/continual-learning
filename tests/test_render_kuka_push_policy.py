"""Read-only KUKA push checkpoint replay helpers."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    assert "finally:\n        env.close()" in source


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
