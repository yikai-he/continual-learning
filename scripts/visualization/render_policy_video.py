"""Render deterministic MetaWorld policy rollouts to one MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np

from src.continual.evaluation import disable_task_sampling, fixed_mt1_tasks
from src.envs import make_metaworld_env


POLICY_TYPES = ("sac", "bc", "general-policy", "diffcrl")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy-type", choices=POLICY_TYPES, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--codec",
        default="libx264",
        help="FFmpeg video encoder (default: libx264).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--evaluation-mode",
        choices=("fixed-tasks", "sampled"),
        default="fixed-tasks",
    )
    parser.add_argument("--task-set-seed", type=int, default=10_000)
    parser.add_argument(
        "--reward-function-version", choices=("v1", "v2"), default="v2"
    )
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive.")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite.")
    if not args.codec or args.codec != args.codec.strip():
        parser.error("--codec must be a nonempty FFmpeg encoder name.")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4.")
    if args.output.exists():
        parser.error("--output must be a new file; refusing to overwrite it.")
    if args.evaluation_mode == "fixed-tasks" and args.episodes > 50:
        parser.error("Fixed-task rendering supports at most 50 MT1 configurations.")
    return args


def resolve_model_path(path: Path, policy_type: str) -> Path:
    candidates = [path]
    if policy_type == "sac" and path.suffix != ".zip":
        candidates.append(path.with_suffix(".zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Policy checkpoint not found: {path}")


def load_policy(policy_type: str, path: Path, device: str):
    """Load a repository policy behind the common unbatched act() interface."""
    if policy_type == "sac":
        from stable_baselines3 import SAC

        from src.continual.policy_adapter import SB3SACPolicy

        return SB3SACPolicy(SAC.load(path, device=device))

    from src.continual.bc_policy import GeneralPolicy

    policy = GeneralPolicy.load(path)
    policy.to(device)
    policy.eval()
    return policy


def _rgb_frame(frame) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected render() to return an HxWx3 uint8 RGB frame.")
    return frame


class _FFmpegWriter:
    """Stream fixed-size RGB frames to an H.264-compatible MP4 encoder."""

    def __init__(self, path: Path, frame: np.ndarray, fps: float, codec: str):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to encode MP4 videos.")
        self.height, self.width = frame.shape[:2]
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{self.width}x{self.height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError("Rendered frame dimensions changed during the rollout.")
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is unavailable.")
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            error = self.process.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg stopped while encoding: {error}") from exc

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace").strip()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(
                f"FFmpeg exited with status {return_code}: {error or 'unknown error'}"
            )


def _open_writer(path: Path, frame: np.ndarray, fps: float, codec: str):
    return _FFmpegWriter(path, frame, fps, codec)


def render_policy_video(
    policy,
    env,
    output: Path,
    *,
    episodes: int,
    seed: int,
    fps: float,
    codec: str,
    fixed_tasks=None,
    writer_factory=_open_writer,
) -> dict:
    """Roll out a policy and write initial plus post-step RGB frames."""
    writer = None
    returns, lengths, successes = [], [], []
    try:
        for episode in range(episodes):
            if fixed_tasks is not None:
                env.unwrapped.set_task(fixed_tasks[episode])
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            episode_length = 0
            episode_success = False

            frame = _rgb_frame(env.render())
            if writer is None:
                writer = writer_factory(output, frame, fps, codec)
            writer.write(frame)

            while True:
                action = policy.act(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                frame = _rgb_frame(env.render())
                writer.write(frame)
                episode_return += float(reward)
                episode_length += 1
                episode_success |= bool(info.get("success", False))
                if terminated or truncated:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(episode_success)
            print(
                f"Episode {episode + 1}/{episodes}: "
                f"return={episode_return:.3f}, success={episode_success}, "
                f"length={episode_length}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.release()

    return {
        "episodes": episodes,
        "success_count": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "frame_count": int(sum(lengths) + episodes),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    model_path = resolve_model_path(args.model, args.policy_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.policy_type, model_path, args.device)
    env = make_metaworld_env(
        args.task,
        args.seed,
        render_mode="rgb_array",
        reward_function_version=args.reward_function_version,
    )
    try:
        model = getattr(policy, "model", None)
        if model is not None and (
            model.observation_space != env.observation_space
            or model.action_space != env.action_space
        ):
            raise ValueError("SAC checkpoint and environment spaces differ.")
        fixed_tasks = None
        if args.evaluation_mode == "fixed-tasks":
            fixed_tasks, _ = fixed_mt1_tasks(args.task, args.task_set_seed)
            disable_task_sampling(env)
        result = render_policy_video(
            policy,
            env,
            args.output.resolve(),
            episodes=args.episodes,
            seed=args.seed,
            fps=args.fps,
            codec=args.codec,
            fixed_tasks=fixed_tasks,
        )
    finally:
        env.close()

    print(f"Video: {args.output.resolve()}")
    print(f"Success: {result['success_count']}/{result['episodes']}")
    print(f"Mean return: {result['mean_return']:.3f} ± {result['std_return']:.3f}")
    print(f"Mean episode length: {result['mean_episode_length']:.1f}")


if __name__ == "__main__":
    main()
