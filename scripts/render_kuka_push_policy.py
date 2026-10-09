#!/usr/bin/env python3
"""Interactively replay deterministic reward-v2 KUKA push SAC checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import glfw
import mujoco
import numpy as np
from stable_baselines3 import SAC
from metaworld.wrappers import RandomTaskSelectWrapper

from src.envs import make_env
from src.env_adapters.kuka_v3 import KUKA_V3_TASKS


RUN = Path("runs/reward_v2/kuka-push-v3/seed_0_150k_probe")
CHECKPOINTS = {
    "best-success": RUN / "best_success_model/best_success_model.zip",
    "best-return": RUN / "best_model/best_model.zip",
    "final": RUN / "final_model.zip",
}
AUDIT_CASES = {
    "45k-success": {
        "checkpoint": "best-success",
        "seed": 10003,
        "task_index": 29,
        "task_sha256": "1921052616856f4dd35a4913cc99c51a72d463e37f4c7851ae394aa53d4287cf",
    },
    "115k-wrong-direction": {
        "checkpoint": "best-return",
        "seed": 10019,
        "task_index": 6,
        "task_sha256": "e928bb522c358038bb6875fdbb60449b55700be5b6694a2d6632cfe322d66031",
    },
    "150k-stationary": {
        "checkpoint": "final",
        "seed": 10002,
        "task_index": 21,
        "task_sha256": "f5f74022fec28cf9b400905d82099633ac91457d28df26445f28a3abc5fb598f",
    },
}
CAMERA_NAMES = {1: "default", 2: "top-down", 3: "side", 4: "front", 5: "close-up"}
HUD_MODES = ("compact", "verbose", "hidden")
HAMMER_NAIL_SUCCESS_THRESHOLD = 0.09
CONTROL_KEYS = {
    glfw.KEY_SPACE,
    glfw.KEY_R,
    glfw.KEY_BACKSPACE,
    glfw.KEY_N,
    glfw.KEY_L,
    glfw.KEY_H,
    glfw.KEY_I,
    glfw.KEY_F1,
    glfw.KEY_SLASH,
    glfw.KEY_Q,
    glfw.KEY_ESCAPE,
    glfw.KEY_1,
    glfw.KEY_2,
    glfw.KEY_3,
    glfw.KEY_4,
    glfw.KEY_5,
}


class ViewerPresentation:
    def __init__(self) -> None:
        self.hud_mode = "compact"
        self.show_help = False
        self.show_builtin_menu = False

    def cycle_hud(self) -> None:
        index = (HUD_MODES.index(self.hud_mode) + 1) % len(HUD_MODES)
        self.hud_mode = HUD_MODES[index]

    def toggle_help(self) -> None:
        self.show_help = not self.show_help


class ReplayCommand:
    def __init__(self) -> None:
        self.quit = False
        self.restart = False
        self.next_seed = False


class TaskRequest:
    def __init__(
        self,
        seed: int,
        task_index: int,
        task_sha256: str,
        label: str,
    ) -> None:
        self.seed = seed
        self.task_index = task_index
        self.task_sha256 = task_sha256
        self.label = label


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", choices=tuple(CHECKPOINTS))
    source.add_argument("--model", type=Path, help="arbitrary SAC .zip checkpoint")
    parser.add_argument(
        "--env-name",
        choices=KUKA_V3_TASKS,
        default="kuka-push-v3",
        help="KUKA-v3 environment to replay (default: kuka-push-v3)",
    )
    parser.add_argument("--audit-case", choices=tuple(AUDIT_CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[10_000])
    parser.add_argument("--task-index", type=int, choices=range(50))
    parser.add_argument("--task-set-seed", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=1, help="repetitions per seed")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-name", default="corner")
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1280, 900),
        help="fallback GLFW window size if maximizing is unavailable",
    )
    parser.add_argument("--no-loop", action="store_true", help="freeze after an episode")
    parser.add_argument("--loop-delay", type=float, default=1.0)
    parser.add_argument(
        "--smoke-test",
        type=int,
        metavar="STEPS",
        help="run a headless deterministic prefix and exit",
    )
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        parser.error("--control-hz must be positive and finite")
    if args.print_every < 1:
        parser.error("--print-every must be positive")
    if not np.isfinite(args.loop_delay) or args.loop_delay < 0:
        parser.error("--loop-delay must be non-negative and finite")
    if args.audit_case and (args.model is not None or args.task_index is not None):
        parser.error("--audit-case cannot be combined with --model or --task-index")
    if args.audit_case and args.env_name != "kuka-push-v3":
        parser.error("--audit-case is only available with --env-name kuka-push-v3")
    if args.smoke_test is not None and not 1 <= args.smoke_test <= 200:
        parser.error("--smoke-test must be in [1, 200]")
    if any(dimension < 1 for dimension in args.window_size):
        parser.error("--window-size dimensions must be positive")
    args.window_size = tuple(args.window_size)
    return args


def configure_viewer_window(
    window: Any, requested: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Maximize a GLFW window, falling back to an explicit large size."""
    before = tuple(glfw.get_window_size(window))
    maximized = False
    try:
        glfw.maximize_window(window)
        glfw.poll_events()
        actual = tuple(glfw.get_window_size(window))
        maximized = bool(glfw.get_window_attrib(window, glfw.MAXIMIZED))
    except (glfw.GLFWError, AttributeError):
        actual = before
    if not maximized and actual == before and (
        actual[0] < requested[0] or actual[1] < requested[1]
    ):
        glfw.set_window_size(window, *requested)
        glfw.poll_events()
        actual = tuple(glfw.get_window_size(window))
    framebuffer = tuple(glfw.get_framebuffer_size(window))
    print(
        f"Viewer dimensions | requested={requested[0]}x{requested[1]} | "
        f"window={actual[0]}x{actual[1]} | framebuffer={framebuffer[0]}x{framebuffer[1]}",
        flush=True,
    )
    return actual, framebuffer


def resolve_model(args: argparse.Namespace) -> tuple[Path, str]:
    checkpoint = (
        AUDIT_CASES[args.audit_case]["checkpoint"]
        if args.audit_case
        else args.checkpoint or "best-success"
    )
    path = args.model if args.model is not None else CHECKPOINTS[checkpoint]
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    candidates = (path, path.with_suffix(".zip")) if path.suffix != ".zip" else (path,)
    for candidate in candidates:
        if candidate.is_file():
            label = checkpoint if args.model is None else candidate.stem
            return candidate.resolve(), label
    raise FileNotFoundError(f"SAC checkpoint not found: {path}")


def validate_spaces(model: SAC, env: Any) -> None:
    expected_obs, expected_action = (39,), (4,)
    values = {
        "model observation": model.observation_space.shape,
        "model action": model.action_space.shape,
        "environment observation": env.observation_space.shape,
        "environment action": env.action_space.shape,
    }
    for name, shape in values.items():
        expected = expected_obs if "observation" in name else expected_action
        if shape != expected:
            raise ValueError(f"{name} shape must be {expected}, got {shape}")
    if model.observation_space != env.observation_space:
        raise ValueError("model and environment observation spaces differ")
    if model.action_space != env.action_space:
        raise ValueError("model and environment action spaces differ")


def find_task_wrapper(env: Any) -> RandomTaskSelectWrapper:
    current = env
    while current is not None:
        if isinstance(current, RandomTaskSelectWrapper):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("KUKA environment has no RandomTaskSelectWrapper")


def task_hash(task: Any) -> str:
    return hashlib.sha256(task.data).hexdigest()


def resolve_requests(args: argparse.Namespace, wrapper: RandomTaskSelectWrapper) -> list[TaskRequest]:
    if args.audit_case:
        case = AUDIT_CASES[args.audit_case]
        requests = [
            TaskRequest(
                case["seed"],
                case["task_index"],
                case["task_sha256"],
                args.audit_case,
            )
        ]
    elif args.task_index is not None:
        digest = task_hash(wrapper.tasks[args.task_index])
        requests = [
            TaskRequest(seed, args.task_index, digest, f"task-{args.task_index}")
            for seed in args.seeds
        ]
    else:
        raise ValueError(
            "A reproducible replay requires --audit-case or --task-index; "
            "reset seed alone does not select a KUKA task."
        )
    return [request for request in requests for _ in range(args.episodes)]


def reset_exact_task(
    env: Any,
    wrapper: RandomTaskSelectWrapper,
    request: TaskRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = wrapper.tasks[request.task_index]
    digest = task_hash(task)
    if digest != request.task_sha256:
        raise RuntimeError(
            f"task {request.task_index} hash mismatch: {digest} != {request.task_sha256}"
        )
    env.unwrapped.set_task(task)
    wrapper.toggle_sample_tasks_on_reset(False)
    observation, _ = env.reset(seed=request.seed)
    base = env.unwrapped
    obj = np.asarray(observation[4:7], dtype=float).copy()
    goal = np.asarray(observation[36:39], dtype=float).copy()
    distance = float(np.linalg.norm(obj - goal))
    tcp = np.asarray(observation[:3], dtype=float).copy()
    task_vector = np.asarray(pickle.loads(task.data)["rand_vec"], dtype=float)
    print(
        f"RESET requested_seed={request.seed} | task_index={request.task_index} | "
        f"task_sha256={digest} | tcp={np.array2string(tcp, precision=5)} | "
        f"object={np.array2string(obj, precision=5)} | "
        f"goal={np.array2string(goal, precision=5)} | object_goal={distance:.6f} m | "
        f"task_vector={np.array2string(task_vector, precision=6)}",
        flush=True,
    )
    np.testing.assert_allclose(goal, np.asarray(base._target_pos), atol=1e-7)
    return np.asarray(observation), obj, goal


def hud_lines(
    *,
    label: str,
    request: TaskRequest,
    state: dict[str, Any],
    paused: bool,
    auto_loop: bool,
    camera: str,
    verbose: bool,
    task_specific_lines: tuple[str, ...] = (),
) -> list[str]:
    """Build narrow, single-column diagnostics without world coordinates."""
    success = "YES" if state["success"] else "NO"
    contact = "YES" if state["contact"] else "NO"
    clipping = "YES" if state["clip"] else "NO"
    looping = "ON" if auto_loop else "OFF"
    lines = [
        f"{label} | task {request.task_index} | loop {looping}",
        f"{state['step']}/200 | R {state['reward']:+.2f} | Sum {state['return']:.1f}",
        f"Success {success} | Contact {contact} | Clip {clipping}",
        f"Goal {state['distance']:.4f} m",
        f"dProg {state['signed']:+.4f} | Max {state['max_progress']:+.4f}",
    ]
    lines.extend(task_specific_lines)
    if verbose:
        lines.extend(
            [
                f"Replay {request.label} | Seed {request.seed}",
                f"State {'PAUSED' if paused else 'PLAYING'} | Camera {camera}",
                f"Contact steps {state['contact_steps']}",
                f"Total progress {state['progress']:+.4f} m",
                f"Min goal {state['min_distance']:.4f} m",
                f"Max displacement {state['max_displacement']:.4f} m",
            ]
        )
    return lines


def get_task_specific_debug_info(env_name: str, base: Any) -> tuple[str, ...]:
    """Return HUD diagnostics that are valid only for a specific task."""
    if env_name != "kuka-hammer-v3":
        return ()
    nail_qpos = float(base.data.joint("NailSlideJoint").qpos[0])
    progress = 100.0 * nail_qpos / HAMMER_NAIL_SUCCESS_THRESHOLD
    return (
        f"Nail {nail_qpos:.3f} / {HAMMER_NAIL_SUCCESS_THRESHOLD:.3f} m "
        f"({progress:.0f}%)",
    )


def control_lines() -> list[str]:
    return [
        "Space pause | R restart | N next | L loop",
        "I HUD | H MuJoCo menu | Q quit | 1-5 camera",
        "? / F1 help",
    ]


def advance_after_episode(*, auto_loop: bool, command: ReplayCommand) -> str:
    if command.quit:
        return "quit"
    if command.next_seed:
        command.next_seed = False
        return "next"
    if command.restart:
        command.restart = False
        return "restart"
    return "restart" if auto_loop else "freeze"


def workspace_would_clip(base: Any, action: np.ndarray) -> bool:
    proposed = np.asarray(base.data.mocap_pos[0], dtype=float) + np.clip(
        action[:3], -1.0, 1.0
    ) * float(base.action_scale)
    clipped = np.clip(proposed, base.mocap_low, base.mocap_high)
    return not np.allclose(proposed, clipped, atol=1e-12, rtol=0.0)


def object_contact(base: Any, info: dict[str, Any], observation: np.ndarray) -> bool:
    """Return task contact telemetry without assuming an ``objGeom`` exists."""
    try:
        return bool(
            base.pad_contacting_object(base._get_id_main_object())
            and observation[3] > 0
        )
    except KeyError:
        # Articulated tasks such as handle-press-side do not define objGeom.
        # Their task evaluator still exposes a useful near-object signal.
        return bool(info.get("near_object", False))


def add_object_marker(viewer: Any, base: Any) -> None:
    """Highlight objGeom when the selected task provides that named geometry."""
    try:
        obj_geom = base.data.geom("objGeom")
    except KeyError:
        return
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=base.model.geom_size[obj_geom.id].copy(),
        pos=obj_geom.xpos.copy(),
        mat=obj_geom.xmat.copy(),
        rgba=np.array([1.0, 0.0, 0.0, 1.0]),
    )


def episode_summary(
    *,
    seed: int,
    success: bool,
    first_success: int | None,
    episode_return: float,
    first_contact: int | None,
    contact_steps: int,
    maximum_displacement: float,
    maximum_progress: float,
    minimum_distance: float,
    final_distance: float,
) -> None:
    print(
        "EPISODE END | "
        f"seed={seed} | success={int(success)} | first_success={first_success} | "
        f"return={episode_return:.3f} | first_contact={first_contact} | "
        f"contact_steps={contact_steps} | max_displacement={maximum_displacement:.6f} | "
        f"max_goal_progress={maximum_progress:.6f} | min_goal_distance={minimum_distance:.6f} | "
        f"final_goal_distance={final_distance:.6f}",
        flush=True,
    )


def configure_camera(viewer: Any, base: Any, preset: int, default: dict[str, Any]) -> str:
    if preset == 1:
        for key, value in default.items():
            if key == "lookat":
                viewer.cam.lookat[:] = value
            else:
                setattr(viewer.cam, key, value)
        return CAMERA_NAMES[preset]
    obj = np.asarray(base._get_pos_objects(), dtype=float)[:3]
    goal = np.asarray(base._target_pos, dtype=float)[:3]
    tcp = np.asarray(base.tcp_center, dtype=float)[:3]
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = viewer.cam.trackbodyid = -1
    viewer.cam.lookat[:] = np.mean((obj, goal, tcp), axis=0)
    settings = {
        2: (1.15, 90.0, -90.0),
        3: (1.10, 0.0, -18.0),
        4: (1.10, 90.0, -18.0),
        5: (0.48, 135.0, -28.0),
    }
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = settings[preset]
    return CAMERA_NAMES[preset]


def run(args: argparse.Namespace) -> None:
    model_path, label = resolve_model(args)
    print(f"Loading {label}: {model_path}")
    model = SAC.load(model_path, device=args.device)
    env = make_env(
        "kuka-v3",
        args.env_name,
        args.task_set_seed,
        render_mode=None if args.smoke_test else "human",
        reward_function_version="v2",
    )
    validate_spaces(model, env)
    print("Spaces verified: model/env observation=(39,), action=(4,)")
    try:
        wrapper = find_task_wrapper(env)
        requested = resolve_requests(args, wrapper)
        if args.smoke_test:
            observation, initial_obj, goal = reset_exact_task(
                env, wrapper, requested[0]
            )
            for _ in range(args.smoke_test):
                action, _ = model.predict(observation, deterministic=True)
                observation, *_ = env.step(action)
            print(f"SMOKE TEST PASSED: deterministic steps={args.smoke_test}")
            return

        command = ReplayCommand()
        presentation = ViewerPresentation()
        window_configured = False
        auto_loop = not args.no_loop
        request_index = 0
        while request_index < len(requested) and not command.quit:
            request = requested[request_index]
            seed = request.seed
            observation, initial_obj, goal = reset_exact_task(
                env, wrapper, request
            )
            initial_distance = float(np.linalg.norm(initial_obj - goal))
            env.render()
            viewer = env.unwrapped.mujoco_renderer.viewer
            if viewer is None or viewer.window is None:
                raise RuntimeError("MuJoCo human viewer did not expose a GLFW window")
            window = viewer.window
            if not window_configured:
                configure_viewer_window(window, args.window_size)
                window_configured = True
            default_camera = {
                "type": viewer.cam.type,
                "fixedcamid": viewer.cam.fixedcamid,
                "trackbodyid": viewer.cam.trackbodyid,
                "lookat": np.asarray(viewer.cam.lookat).copy(),
                "distance": float(viewer.cam.distance),
                "azimuth": float(viewer.cam.azimuth),
                "elevation": float(viewer.cam.elevation),
            }
            camera = configure_camera(viewer, env.unwrapped, 5, default_camera)
            paused = False
            viewer_callback = viewer._key_callback

            def key_callback(win, key, scancode, action, mods):
                nonlocal paused, camera, auto_loop
                if action == glfw.PRESS and key in CONTROL_KEYS:
                    if key == glfw.KEY_SPACE:
                        paused = not paused
                    elif key in (glfw.KEY_R, glfw.KEY_BACKSPACE):
                        command.restart = True
                    elif key == glfw.KEY_N:
                        command.next_seed = True
                    elif key == glfw.KEY_L:
                        auto_loop = not auto_loop
                        print(f"Auto-loop: {'ON' if auto_loop else 'OFF'}")
                    elif key == glfw.KEY_H:
                        presentation.show_builtin_menu = not presentation.show_builtin_menu
                    elif key == glfw.KEY_I:
                        presentation.cycle_hud()
                    elif key in (glfw.KEY_F1, glfw.KEY_SLASH):
                        presentation.toggle_help()
                    elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                        command.quit = True
                    elif glfw.KEY_1 <= key <= glfw.KEY_5:
                        camera = configure_camera(
                            viewer, env.unwrapped, key - glfw.KEY_0, default_camera
                        )
                    return
                viewer_callback(win, key, scancode, action, mods)

            glfw.set_key_callback(window, key_callback)
            viewer._hide_menu = False
            original_overlay = viewer._create_overlay
            state: dict[str, Any] = {
                "step": 0,
                "reward": 0.0,
                "return": 0.0,
                "success": False,
                "distance": initial_distance,
                "progress": 0.0,
                "max_progress": 0.0,
                "signed": 0.0,
                "contact": False,
                "contact_steps": 0,
                "clip": False,
                "min_distance": initial_distance,
                "max_displacement": 0.0,
            }

            def overlay() -> None:
                if presentation.show_builtin_menu:
                    original_overlay()
                if presentation.hud_mode != "hidden":
                    diagnostics = hud_lines(
                        label=label,
                        request=request,
                        state=state,
                        paused=paused,
                        auto_loop=auto_loop,
                        camera=camera,
                        verbose=presentation.hud_mode == "verbose",
                        task_specific_lines=get_task_specific_debug_info(
                            args.env_name, env.unwrapped
                        ),
                    )
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(diagnostics), ""
                    )
                if presentation.show_help and presentation.hud_mode != "hidden":
                    viewer.add_overlay(
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        "\n".join(control_lines()),
                        "",
                    )

            viewer._create_overlay = overlay
            timestep = contact_steps = 0
            episode_return = 0.0
            first_success = first_contact = None
            minimum_distance = initial_distance
            maximum_progress = maximum_displacement = 0.0
            previous_distance = initial_distance
            next_step_at = time.monotonic()
            while timestep < 200 and not command.quit and not command.restart and not command.next_seed:
                if glfw.window_should_close(window):
                    command.quit = True
                    break
                if paused:
                    add_object_marker(viewer, env.unwrapped)
                    env.render()
                    time.sleep(0.02)
                    continue
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                clipping = workspace_would_clip(env.unwrapped, action)
                observation, reward, terminated, truncated, info = env.step(action)
                timestep += 1
                obj = np.asarray(observation[4:7], dtype=float)
                distance = float(np.linalg.norm(obj - goal))
                signed = previous_distance - distance
                contact = object_contact(env.unwrapped, info, observation)
                episode_return += float(reward)
                contact_steps += int(contact)
                success = bool(info.get("success", False))
                if contact and first_contact is None:
                    first_contact = timestep
                if success and first_success is None:
                    first_success = timestep
                minimum_distance = min(minimum_distance, distance)
                maximum_progress = max(maximum_progress, initial_distance - distance)
                maximum_displacement = max(
                    maximum_displacement, float(np.linalg.norm(obj - initial_obj))
                )
                state.update(
                    {
                        "step": timestep,
                        "reward": float(reward),
                        "return": episode_return,
                        "success": success,
                        "distance": distance,
                        "progress": initial_distance - distance,
                        "max_progress": maximum_progress,
                        "signed": signed,
                        "contact": contact,
                        "contact_steps": contact_steps,
                        "clip": clipping,
                        "min_distance": minimum_distance,
                        "max_displacement": maximum_displacement,
                    }
                )
                if timestep % args.print_every == 0 or success:
                    print(
                        f"t={timestep:03d} reward={reward:+.3f} return={episode_return:.3f} "
                        f"success={int(success)} contact={int(contact)} distance={distance:.6f} "
                        f"signed_progress={signed:+.6f} displacement={np.linalg.norm(obj-initial_obj):.6f}",
                        flush=True,
                    )
                add_object_marker(viewer, env.unwrapped)
                env.render()
                previous_distance = distance
                if terminated or truncated:
                    break
                next_step_at += 1.0 / args.control_hz
                delay = next_step_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_step_at = time.monotonic()
            if timestep:
                episode_summary(
                    seed=seed,
                    success=first_success is not None,
                    first_success=first_success,
                    episode_return=episode_return,
                    first_contact=first_contact,
                    contact_steps=contact_steps,
                    maximum_displacement=maximum_displacement,
                    maximum_progress=maximum_progress,
                    minimum_distance=minimum_distance,
                    final_distance=previous_distance,
                )
            disposition = advance_after_episode(
                auto_loop=auto_loop, command=command
            )
            if disposition == "quit":
                break
            if disposition == "next":
                request_index += 1
                continue
            if disposition == "restart":
                deadline = time.monotonic() + args.loop_delay
                while (
                    time.monotonic() < deadline
                    and not command.quit
                    and not command.restart
                    and not command.next_seed
                ):
                    env.render()
                    time.sleep(0.02)
                disposition = advance_after_episode(
                    auto_loop=auto_loop, command=command
                )
                if disposition == "next":
                    request_index += 1
                    continue
                elif disposition == "quit":
                    break
                elif disposition == "restart":
                    continue
            # Loop is off: preserve and render the final frame until directed.
            while not command.quit and not command.restart and not command.next_seed:
                if glfw.window_should_close(window):
                    command.quit = True
                    break
                env.render()
                time.sleep(0.02)
            disposition = advance_after_episode(
                auto_loop=auto_loop, command=command
            )
            if disposition == "next":
                request_index += 1
            elif disposition == "quit":
                break
            # restart keeps the same request index.
    except KeyboardInterrupt:
        print("\nInterrupted once; closing viewer cleanly.", flush=True)
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
