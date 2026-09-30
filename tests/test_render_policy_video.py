"""Offline policy-video rendering utility."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.visualization.render_policy_video import (
    parse_args,
    render_policy_video,
    resolve_model_path,
)


class _Policy:
    def __init__(self):
        self.calls = []

    def act(self, observation, deterministic=True):
        self.calls.append((observation.copy(), deterministic))
        return np.zeros(4, dtype=np.float32)


class _Unwrapped:
    def __init__(self):
        self.tasks = []

    def set_task(self, task):
        self.tasks.append(task)


class _Env:
    def __init__(self):
        self.unwrapped = _Unwrapped()
        self.steps = 0
        self.reset_seeds = []
        self.closed = False

    def reset(self, *, seed):
        self.steps = 0
        self.reset_seeds.append(seed)
        return np.zeros(39, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.full(39, self.steps, dtype=np.float32),
            1.5,
            False,
            self.steps == 2,
            {"success": self.steps == 1},
        )

    def render(self):
        return np.full((4, 6, 3), self.steps, dtype=np.uint8)


class _Writer:
    def __init__(self):
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class RenderPolicyVideoTests(unittest.TestCase):
    def test_rollout_writes_frames_and_returns_metrics(self):
        policy, env, writer = _Policy(), _Env(), _Writer()
        calls = []

        def factory(path, frame, fps, codec):
            calls.append((path, frame.shape, fps, codec))
            return writer

        result = render_policy_video(
            policy,
            env,
            Path("video.mp4"),
            episodes=2,
            seed=10,
            fps=24,
            codec="mp4v",
            fixed_tasks=["task-a", "task-b"],
            writer_factory=factory,
        )
        self.assertEqual(env.reset_seeds, [10, 11])
        self.assertEqual(env.unwrapped.tasks, ["task-a", "task-b"])
        self.assertEqual(len(policy.calls), 4)
        self.assertTrue(all(deterministic for _, deterministic in policy.calls))
        self.assertEqual(len(writer.frames), 6)
        self.assertTrue(writer.released)
        self.assertEqual(calls, [(Path("video.mp4"), (4, 6, 3), 24, "mp4v")])
        self.assertEqual(
            result,
            {
                "episodes": 2,
                "success_count": 2,
                "success_rate": 1.0,
                "mean_return": 3.0,
                "std_return": 0.0,
                "mean_episode_length": 2.0,
                "frame_count": 6,
            },
        )

    def test_cli_validation_and_sac_suffix_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.zip"
            model.touch()
            self.assertEqual(
                resolve_model_path(model.with_suffix(""), "sac"), model.resolve()
            )
            args = parse_args(
                [
                    "--task",
                    "hammer-v3",
                    "--model",
                    str(model),
                    "--policy-type",
                    "sac",
                    "--output",
                    str(Path(tmp) / "demo.mp4"),
                ]
            )
            self.assertEqual(
                (args.episodes, args.evaluation_mode, args.codec),
                (1, "fixed-tasks", "libx264"),
            )

    def test_invalid_frame_is_rejected(self):
        policy, env = _Policy(), _Env()
        env.render = lambda: np.zeros((4, 6), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            render_policy_video(
                policy,
                env,
                Path("video.mp4"),
                episodes=1,
                seed=0,
                fps=30,
                codec="mp4v",
            )


if __name__ == "__main__":
    unittest.main()
