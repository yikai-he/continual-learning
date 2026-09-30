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
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from src.config import (
    ExperimentConfig,
    output_directory,
    parse_config,
    save_resolved_config,
)
from src.envs import make_metaworld_env


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
    for directory in ("tensorboard", "evaluations", "checkpoints", "best_model"):
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
        "checkpoint_freq": config.sac.checkpoint_freq,
        "run_dir": str(run_dir),
        "algorithm": "SAC",
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
        make_metaworld_env(
            config.continual.tasks[0],
            config.runtime.seed,
            reward_function_version=config.environment.reward_function_version,
        )
    )
    eval_env = Monitor(
        make_metaworld_env(
            config.continual.tasks[0],
            config.runtime.seed + config.evaluation.seed_offset,
            reward_function_version=config.environment.reward_function_version,
        )
    )
    callbacks = CallbackList(
        [
            TrainingSuccessCallback(),
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "evaluations"),
                eval_freq=config.evaluation.frequency,
                n_eval_episodes=config.evaluation.episodes,
                deterministic=True,
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
