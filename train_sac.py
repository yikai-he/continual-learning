"""Train a reproducible Stable-Baselines3 SAC baseline on MetaWorld MT1."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from src.config import (
    ExperimentConfig,
    output_directory,
    parse_config,
    save_resolved_config,
)
from src.envs import horizon_for_backend, make_env


class TrainingSuccessCallback(BaseCallback):
    """Log a rolling any-step success rate when vectorized episodes finish."""

    def __init__(self, window_size: int = 100) -> None:
        super().__init__()
        self._successes: deque[float] = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                self._successes.append(float(bool(info.get("is_success", False))))
                self.logger.record(
                    "rollout/success_rate", float(np.mean(self._successes))
                )
        return True


class FixedSeedEvalCallback(BaseCallback):
    """Evaluate on one fixed seed bank and retain return- and success-best models."""

    def __init__(
        self,
        eval_env,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int,
        log_path: Path,
        best_model_save_path: Path,
        best_success_model_save_path: Path,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seeds = tuple(range(seed, seed + n_eval_episodes))
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.best_success_model_save_path = best_success_model_save_path
        self.best_mean_return = -np.inf
        self.best_success_score = (-np.inf, -np.inf)
        self.timesteps: list[int] = []
        self.results: list[list[float]] = []
        self.ep_lengths: list[list[int]] = []
        self.successes: list[list[bool]] = []

    def _evaluate(self) -> tuple[list[float], list[int], list[bool]]:
        returns, lengths, successes = [], [], []
        for seed in self.seeds:
            observation, _ = self.eval_env.reset(seed=seed)
            episode_return = 0.0
            episode_length = 0
            episode_success = False
            while True:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.eval_env.step(
                    action
                )
                episode_return += float(reward)
                episode_length += 1
                episode_success |= bool(
                    info.get("success", info.get("is_success", False))
                )
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(episode_success)
        return returns, lengths, successes

    def _save_evaluations(self) -> None:
        np.savez(
            self.log_path / "evaluations.npz",
            timesteps=np.asarray(self.timesteps),
            results=np.asarray(self.results),
            ep_lengths=np.asarray(self.ep_lengths),
            successes=np.asarray(self.successes),
            seeds=np.asarray(self.seeds),
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq:
            return True
        returns, lengths, successes = self._evaluate()
        mean_return = float(np.mean(returns))
        success_rate = float(np.mean(successes))
        self.timesteps.append(self.num_timesteps)
        self.results.append(returns)
        self.ep_lengths.append(lengths)
        self.successes.append(successes)
        self._save_evaluations()
        self.logger.record("eval/mean_reward", mean_return)
        self.logger.record("eval/success_rate", success_rate)

        if mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            self.model.save(self.best_model_save_path / "best_model")
        score = (success_rate, mean_return)
        if score > self.best_success_score:
            self.best_success_score = score
            self.model.save(
                self.best_success_model_save_path / "best_success_model"
            )
        return True


def parse_args(argv=None) -> ExperimentConfig:
    """Parse and validate the SAC YAML configuration plus CLI overrides."""
    return parse_config("sac", __doc__, argv)


def create_run_directory(config: ExperimentConfig) -> Path:
    """Create a collision-safe run directory and return its absolute path.

    Canonical unnamed runs use reward/task/seed hierarchy; an existing target
    receives a timestamped sibling so prior checkpoints are never overwritten.
    """
    if config.runtime.run_name is not None:
        base = output_directory(config)
    else:
        base = (
            Path(config.runtime.output)
            / f"reward_{config.environment.reward_function_version}"
            / config.continual.tasks[0]
            / f"seed_{config.runtime.seed}"
        )
    if not base.exists():
        run_dir = base
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = base.parent / f"{base.name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def main(argv=None) -> None:
    """Train one independent single-task SB3 SAC expert.

    The evaluation callback saves the best deterministic-evaluation checkpoint;
    ``final_model`` stores the policy at the final training step. This script
    does not update the continual GeneralPolicy.
    """
    config = parse_args(argv)
    run_dir = create_run_directory(config)
    save_resolved_config(config, run_dir)
    for directory in (
        "tensorboard",
        "evaluations",
        "checkpoints",
        "best_model",
        "best_success_model",
    ):
        (run_dir / directory).mkdir()
    # Keep existing provenance keys for checkpoint audits and consumers.
    metadata = {
        "task": config.continual.tasks[0],
        "steps": config.sac.steps,
        "seed": config.runtime.seed,
        "device": config.runtime.device,
        "learning_starts": config.sac.learning_starts,
        "eval_freq": config.evaluation.frequency,
        "eval_episodes": config.evaluation.episodes,
        "eval_seeds": list(
            range(
                config.runtime.seed + config.evaluation.seed_offset,
                config.runtime.seed
                + config.evaluation.seed_offset
                + config.evaluation.episodes,
            )
        ),
        "checkpoint_freq": config.sac.checkpoint_freq,
        "run_dir": str(run_dir),
        "algorithm": "SAC",
        "env_backend": config.environment.backend,
        "effective_horizon": horizon_for_backend(config.environment.backend),
        "reward_function_version": config.environment.reward_function_version,
        "policy": "MlpPolicy",
        **{
            name: getattr(config.sac, name)
            for name in (
                "learning_rate",
                "buffer_size",
                "batch_size",
                "gamma",
                "tau",
                "train_freq",
                "gradient_steps",
                "ent_coef",
                "net_arch",
            )
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train_env = Monitor(
        make_env(
            config.environment.backend,
            config.continual.tasks[0],
            config.runtime.seed,
            reward_function_version=config.environment.reward_function_version,
        )
    )
    eval_env = Monitor(
        make_env(
            config.environment.backend,
            config.continual.tasks[0],
            config.runtime.seed + config.evaluation.seed_offset,
            reward_function_version=config.environment.reward_function_version,
        )
    )
    callbacks = CallbackList(
        [
            TrainingSuccessCallback(),
            FixedSeedEvalCallback(
                eval_env,
                eval_freq=config.evaluation.frequency,
                n_eval_episodes=config.evaluation.episodes,
                seed=config.runtime.seed + config.evaluation.seed_offset,
                log_path=run_dir / "evaluations",
                best_model_save_path=run_dir / "best_model",
                best_success_model_save_path=run_dir / "best_success_model",
            ),
            CheckpointCallback(
                save_freq=config.sac.checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="sac",
            ),
        ]
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=config.sac.learning_rate,
        buffer_size=config.sac.buffer_size,
        learning_starts=config.sac.learning_starts,
        batch_size=config.sac.batch_size,
        gamma=config.sac.gamma,
        tau=config.sac.tau,
        train_freq=config.sac.train_freq,
        gradient_steps=config.sac.gradient_steps,
        ent_coef=config.sac.ent_coef,
        policy_kwargs={"net_arch": config.sac.net_arch},
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=config.runtime.seed,
        device=config.runtime.device,
        verbose=0,
    )
    try:
        model.learn(
            total_timesteps=config.sac.steps,
            callback=callbacks,
            tb_log_name="SAC",
            progress_bar=True,
        )
        if model.replay_buffer is None:
            raise RuntimeError(
                "SAC training completed without creating a replay buffer."
            )
        print(f"Replay buffer transitions: {model.replay_buffer.size()}")
        model.save(run_dir / "final_model")
        print(f"Final model saved to: {run_dir / 'final_model.zip'}")
        print(f"Run directory: {run_dir}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
