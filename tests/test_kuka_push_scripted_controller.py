"""Calibrated KUKA Push scripted-controller and viewer integration tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.kuka_push_scripted_controller import CalibratedPushController


SCRIPT = Path("scripts/render_kuka_push_policy.py")
SPEC = importlib.util.spec_from_file_location("scripted_push_viewer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


def test_exact_calibrated_trajectory_and_action_conversion() -> None:
    controller = CalibratedPushController()
    obj = np.array([0.0, 0.50, 0.02])
    goal = np.array([0.0, 0.70, 0.02])
    controller.reset(obj, goal)
    assert (controller.rho, controller.tcp_z, controller.horizon) == (0.060, 0.042, 200)
    assert len(controller._targets) == 200
    np.testing.assert_allclose(controller._targets[0], [0.0, 0.44, 0.142])
    np.testing.assert_allclose(controller._targets[20], [0.0, 0.44, 0.042])
    np.testing.assert_allclose(controller._targets[-1], [0.0, 0.70, 0.042])
    base = SimpleNamespace(tcp_center=np.array([0.0, 0.45, 0.14]), action_scale=0.01)
    action, state = controller.predict(np.zeros(39), base)
    np.testing.assert_array_equal(action, np.array([0.0, -1.0, 0.2, 1.0], dtype=np.float32))
    assert state is None


def test_scripted_cli_rejects_checkpoint_and_non_push_task() -> None:
    args = viewer.parse_args([
        "--controller", "scripted", "--env-name", "kuka-push-cube-v3",
        "--reward-version", "v3_1", "--task-index", "19", "--seeds", "10019",
    ])
    assert args.controller == "scripted"
    with pytest.raises(SystemExit):
        viewer.parse_args(["--controller", "scripted", "--checkpoint", "final"])
    with pytest.raises(SystemExit):
        viewer.parse_args(["--controller", "scripted", "--env-name", "kuka-hammer-v3"])


def test_controller_selection_never_uses_policy_for_scripted() -> None:
    controller = CalibratedPushController()
    controller.reset(np.array([0.0, 0.50, 0.02]), np.array([0.0, 0.70, 0.02]))
    base = SimpleNamespace(tcp_center=np.array([0.0, 0.45, 0.14]), action_scale=0.01)
    action = viewer.controller_action("scripted", None, controller, np.zeros(39), base)
    assert action.shape == (4,)


def test_cube_contact_diagnostics_distinguishes_no_contact() -> None:
    base = SimpleNamespace(_get_id_main_object=lambda: (_ for _ in ()).throw(KeyError()))
    assert viewer.cube_contact_diagnostics(base) == {
        "physical": False,
        "labels": (),
        "tilt_deg": 0.0,
    }


def test_object_marker_uses_loaded_cube_geom_type_and_size() -> None:
    geom = SimpleNamespace(id=45, xpos=np.array([0.0, 0.5, 0.02]), xmat=np.eye(3))
    base = SimpleNamespace(
        data=SimpleNamespace(geom=MagicMock(return_value=geom)),
        model=SimpleNamespace(
            geom_type=np.array([0] * 45 + [int(viewer.mujoco.mjtGeom.mjGEOM_BOX)]),
            geom_size=np.array([[0.0, 0.0, 0.0]] * 45 + [[0.02, 0.02, 0.02]]),
        ),
    )
    mujoco_viewer = SimpleNamespace(add_marker=MagicMock())
    viewer.add_object_marker(mujoco_viewer, base)
    kwargs = mujoco_viewer.add_marker.call_args.kwargs
    assert kwargs["type"] == int(viewer.mujoco.mjtGeom.mjGEOM_BOX)
    np.testing.assert_allclose(kwargs["size"], [0.02, 0.02, 0.02])
