"""Evaluate a saved SAC policy on a MetaWorld MT1 task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from stable_baselines3 import SAC

from src.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--env-backend",
        choices=("metaworld-v3", "kuka-v2"),
        default="metaworld-v3",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reward-function-version", choices=("v1", "v2"), default="v2")
    return parser.parse_args()


def resolve_model_path(path: Path) -> Path:
    candidates = [path]
    if path.suffix != ".zip":
        candidates.append(path.with_suffix(".zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"SAC model not found: {path} (also checked .zip suffix)")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    model_path = resolve_model_path(args.model)
    env = make_env(
        args.env_backend,
        args.task,
        args.seed,
        render_mode="human" if args.render else None,
        reward_function_version=args.reward_function_version,
    )
    try:
        try:
            model = SAC.load(model_path, device="auto")
        except Exception as exc:
            raise RuntimeError(f"Could not load SAC model {model_path}: {exc}") from exc

        if model.observation_space != env.observation_space:
            raise ValueError(
                "Model/environment observation spaces are incompatible: "
                f"model={model.observation_space}, env={env.observation_space}."
            )
        if model.action_space != env.action_space:
            raise ValueError(
                "Model/environment action spaces are incompatible: "
                f"model={model.action_space}, env={env.action_space}."
            )

        returns: list[float] = []
        lengths: list[int] = []
        successes: list[bool] = []
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            episode_length = 0
            episode_success = False
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                episode_success |= bool(info.get("success", False))
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(episode_success)

        print(f"Task: {args.task}")
        print(f"Episodes: {args.episodes}")
        print(f"Mean return: {np.mean(returns):.6f}")
        print(f"Return standard deviation: {np.std(returns):.6f}")
        print(f"Success rate: {np.mean(successes):.6f}")
        print(f"Mean episode length: {np.mean(lengths):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
