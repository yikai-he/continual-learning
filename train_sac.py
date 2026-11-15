"""Train a reproducible Stable-Baselines3 SAC baseline on MetaWorld MT1."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from src.config import (
    SAC_ENT_COEF,
    EarlyStoppingConfig,
    ExperimentConfig,
    output_directory,
    parse_config,
    save_resolved_config,
)
from src.sac_policy import LegacySACPolicy, SAC_POLICY_ARCHITECTURE
from continual_manipulation.evaluation import (
    disable_task_sampling,
    evaluate_task,
    fixed_mt1_tasks,
)
from src.continual.policy_adapter import SB3SACPolicy
from continual_manipulation.task_banks import backend_tasks, bank_hashes
from continual_manipulation.sequences import TaskSpec
from continual_manipulation.environments import (
    ACTION_SHAPE, OBSERVATION_SHAPE, horizon_for_backend, make_env,
)
from src.support.expert_manifest import qualification_record, write_expert_manifest
from src.support.reproducibility import file_hash
from src.support.run_manifest import complete_run, create_run_manifest
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


def target_entropy_from_output_std(output_std: float, action_dim: int) -> float:
    """Return the Legacy Gaussian differential-entropy target."""
    return float(
        action_dim * math.log(output_std * math.sqrt(2.0 * math.pi * math.e))
    )


class LegacyScheduleSAC(SAC):
    """SB3 SAC with the Legacy exploration and batched-update boundaries.

    The Legacy loop used the current zero-based environment-step index for both
    comparisons. Collecting one SB3 step at a time lets this small gate reproduce
    those boundaries without copying SB3's training loop.
    """

    def __init__(
        self,
        *args,
        random_exploration_steps: int = 10_000,
        update_interval: int = 50,
        **kwargs,
    ) -> None:
        self.random_exploration_steps = random_exploration_steps
        self.update_interval = update_interval
        super().__init__(*args, train_freq=1, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Legacy selects uniformly while current_timestep <= start_steps.
        return super()._sample_action(
            self.random_exploration_steps + 1, action_noise, n_envs
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        current_timestep = self.num_timesteps - 1
        if (
            current_timestep < self.learning_starts
            or current_timestep % self.update_interval != 0
        ):
            return
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)


def success_score(success_rate, mean_return) -> tuple:
    """Rank success first and mean return second."""
    values = np.asarray((success_rate, mean_return), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Checkpoint ranking metrics must be finite.")
    return float(values[0]), float(values[1])


def best_success_index(success_rates, mean_returns) -> int:
    """Return the first lexicographic success/return maximum."""
    if len(success_rates) != len(mean_returns) or not len(success_rates):
        raise ValueError("Checkpoint ranking inputs must be nonempty and aligned.")
    return max(
        range(len(success_rates)),
        key=lambda index: success_score(success_rates[index], mean_returns[index]),
    )


class FixedSeedEvalCallback(BaseCallback):
    """Evaluate on one fixed seed bank and retain the success-best expert."""

    def __init__(
        self,
        eval_env,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        seed: int,
        log_path: Path,
        best_success_model_save_path: Path,
        task_name: str | None = None,
        task_set_seed: int | None = None,
        evaluation_mode: str = "sampled",
        evaluate_on_training_start: bool = True,
        early_stopping: EarlyStoppingConfig | None = None,
        best_success_model_name: str = "best_success_model",
        atomic_model_replacement: bool = False,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seeds = tuple(range(seed, seed + n_eval_episodes))
        self.log_path = log_path
        self.best_success_model_save_path = best_success_model_save_path
        self.evaluation_mode = evaluation_mode
        self.task_set_seed = task_set_seed
        self.task_name = task_name
        self.evaluate_on_training_start = evaluate_on_training_start
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.best_success_model_name = best_success_model_name
        self.atomic_model_replacement = atomic_model_replacement
        self.consecutive_successful_evaluations = 0
        self.early_stopped = False
        self.stop_timestep: int | None = None
        self.trigger_success_rate: float | None = None
        self._next_evaluation_timestep: int | None = None
        if evaluation_mode == "fixed-tasks":
            if not task_name or task_set_seed is None:
                raise ValueError(
                    "Fixed-task SAC evaluation requires task_name and task_set_seed."
                )
            if task_name.startswith("kuka-"):
                current = eval_env
                while current is not None and not hasattr(current, "tasks"):
                    current = getattr(current, "env", None)
                if current is None:
                    raise ValueError("Fixed KUKA evaluation requires a task wrapper.")
                self.fixed_tasks = list(current.tasks)
                identities = [
                    {"task_data_sha256": digest}
                    for digest in bank_hashes(self.fixed_tasks)
                ]
            else:
                self.fixed_tasks, identities = fixed_mt1_tasks(task_name, task_set_seed)
            if n_eval_episodes > len(self.fixed_tasks):
                raise ValueError("Evaluation episodes exceed the fixed MT1 task bank.")
            self.task_hashes = tuple(
                item["task_data_sha256"] for item in identities[:n_eval_episodes]
            )
            disable_task_sampling(eval_env)
        elif evaluation_mode == "sampled":
            self.fixed_tasks = None
            self.task_hashes = ()
        else:
            raise ValueError("Unknown SAC evaluation mode.")
        self.best_success_score = (-np.inf, -np.inf)
        self.timesteps: list[int] = []
        self.results: list[list[float]] = []
        self.ep_lengths: list[list[int]] = []
        self.successes: list[list[bool]] = []
        self.return_stddevs: list[float] = []

    def _save_model(self, path: Path) -> None:
        """Save directly by default, or atomically replace one named archive."""
        path = Path(path)
        if not self.atomic_model_replacement:
            self.model.save(path)
            return
        target = path if path.suffix == ".zip" else path.with_suffix(".zip")
        atomic_replace_model(self.model, target)

    def _evaluate(self) -> tuple[list[float], list[int], list[bool]]:
        returns, lengths, successes = [], [], []
        for index, seed in enumerate(self.seeds):
            if self.fixed_tasks is not None:
                self.eval_env.unwrapped.set_task(self.fixed_tasks[index])
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
                step_success = bool(info.get("success", info.get("is_success", False)))
                episode_success |= step_success
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
            return_stddevs=np.asarray(self.return_stddevs),
            seeds=np.asarray(self.seeds),
            evaluation_mode=np.asarray(self.evaluation_mode),
            task_set_seed=np.asarray(
                -1 if self.task_set_seed is None else self.task_set_seed
            ),
            ordered_task_data_sha256=np.asarray(self.task_hashes),
        )

    def _record_evaluation(self, *, allow_early_stop: bool = True) -> bool:
        returns, lengths, successes = self._evaluate()
        mean_return = float(np.mean(returns))
        return_stddev = float(np.std(returns))
        success_rate = float(np.mean(successes))
        self.timesteps.append(self.num_timesteps)
        self.results.append(returns)
        self.ep_lengths.append(lengths)
        self.successes.append(successes)
        self.return_stddevs.append(return_stddev)
        self._save_evaluations()
        self.logger.record("eval/mean_reward", mean_return)
        self.logger.record("eval/std_reward", return_stddev)
        self.logger.record("eval/success_rate", success_rate)
        score = success_score(success_rate, mean_return)
        if score > self.best_success_score:
            self.best_success_score = score
            self._save_model(
                self.best_success_model_save_path / self.best_success_model_name
            )
        return self._update_early_stopping(success_rate) if allow_early_stop else True

    def _update_early_stopping(self, success_rate: float) -> bool:
        """Return False when the configured global success criterion is met."""
        if not self.early_stopping.enabled:
            return True
        if success_rate >= self.early_stopping.success_threshold:
            self.consecutive_successful_evaluations += 1
        else:
            self.consecutive_successful_evaluations = 0
        if (
            self.num_timesteps >= self.early_stopping.min_steps
            and self.consecutive_successful_evaluations >= self.early_stopping.patience
        ):
            self.early_stopped = True
            self.stop_timestep = self.num_timesteps
            self.trigger_success_rate = success_rate
            return False
        return True

    def _on_training_start(self) -> None:
        """Initialize the global schedule and optionally evaluate a fresh policy."""
        current = self.model.num_timesteps
        self._next_evaluation_timestep = next_global_boundary(current, self.eval_freq)
        if self.evaluate_on_training_start:
            self.num_timesteps = current
            self._record_evaluation(allow_early_stop=False)

    def _on_step(self) -> bool:
        if self._next_evaluation_timestep is None:
            self._next_evaluation_timestep = next_global_boundary(
                self.num_timesteps - 1, self.eval_freq
            )
        if self.num_timesteps < self._next_evaluation_timestep:
            return True
        continue_training = self._record_evaluation()
        self._next_evaluation_timestep = next_global_boundary(
            self.num_timesteps, self.eval_freq
        )
        return continue_training


def next_global_boundary(current_timestep: int, frequency: int) -> int:
    """Return the first strict global frequency multiple after ``current``."""
    if frequency <= 0:
        raise ValueError("Callback frequency must be positive.")
    return (current_timestep // frequency + 1) * frequency


def atomic_replace_model(model, path: Path) -> None:
    """Serialize and atomically replace one inference-only SB3 checkpoint."""
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.zip")
    try:
        model.save(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"SAC checkpoint serialization failed: {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def task_bank_provenance(
    task: str,
    training_seed: int,
    validation_seed: int,
    final_seeds: list[int],
    *,
    backend: str,
) -> dict:
    """Record task identities and prove all optimization/evaluation banks disjoint."""
    seeds = [training_seed, validation_seed, *final_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training, validation, and final task-bank seeds must differ.")
    banks = {
        str(seed): bank_hashes(backend_tasks(task, seed, backend=backend))
        for seed in seeds
    }
    for index, left in enumerate(seeds):
        for right in seeds[index + 1 :]:
            overlap = set(banks[str(left)]) & set(banks[str(right)])
            if overlap:
                raise ValueError(
                    f"Task banks {left} and {right} overlap by {len(overlap)} configuration(s)."
                )
    return {
        "training_task_set_seed": training_seed,
        "validation_task_set_seed": validation_seed,
        "final_test_task_set_seeds": list(final_seeds),
        "ordered_task_hashes": banks,
        "all_banks_pairwise_disjoint": True,
    }


def final_test(config: ExperimentConfig, model) -> list[dict]:
    """Evaluate a selected checkpoint once on final banks; never rank with results."""
    task = config.continual.tasks[0]
    policy = SB3SACPolicy(model)
    fixed = config.environment.backend in ("metaworld-v3", "kuka-v3")
    return [
        evaluate_task(
            policy,
            TaskSpec(task, task),
            episodes=config.evaluation.episodes,
            seed=(
                config.evaluation.final_reset_seed + index * config.evaluation.episodes
                if fixed
                else bank_seed
            ),
            evaluation_mode="fixed-tasks" if fixed else "sampled",
            task_set_seed=bank_seed,
            reward_function_version=config.environment.reward_function_version,
            backend=config.environment.backend,
        )
        for index, bank_seed in enumerate(config.evaluation.final_task_set_seeds)
    ]


def parse_args(argv=None) -> ExperimentConfig:
    """Parse and validate one fresh standalone SAC experiment."""
    return parse_config("sac", __doc__, argv)


def early_stopping_metadata(
    config: EarlyStoppingConfig, callback: FixedSeedEvalCallback | None = None
) -> dict:
    """Serialize configured and observed early-stopping state."""
    observed = callback if config.enabled else None
    return {
        "enabled": config.enabled,
        "min_steps": config.min_steps,
        "success_threshold": config.success_threshold,
        "patience": config.patience,
        "triggered": observed.early_stopped if observed is not None else False,
        "stop_timestep": observed.stop_timestep if observed is not None else None,
        "trigger_success_rate": (
            observed.trigger_success_rate if observed is not None else None
        ),
        "consecutive_success_count": (
            observed.consecutive_successful_evaluations if observed is not None else 0
        ),
    }


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

    The evaluation callback atomically saves the best deterministic-evaluation
    checkpoint. This script does not update the continual GeneralPolicy.
    """
    config = parse_args(argv)
    task_bank_checks = (
        {
            config.continual.tasks[0]: task_bank_provenance(
                config.continual.tasks[0],
                config.runtime.seed,
                config.evaluation.task_set_seed,
                config.evaluation.final_task_set_seeds,
                backend=config.environment.backend,
            )
        }
        if config.environment.backend in ("metaworld-v3", "kuka-v3")
        and config.evaluation.mode == "fixed-tasks"
        else {}
    )
    run_dir = create_run_directory(config)
    save_resolved_config(config, run_dir)
    create_run_manifest(run_dir, config, task_banks=task_bank_checks)
    for directory in ("tensorboard", "evaluations"):
        (run_dir / directory).mkdir()

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
            config.evaluation.task_set_seed
            if config.evaluation.mode == "fixed-tasks"
            else config.runtime.seed + config.evaluation.seed_offset,
            reward_function_version=config.environment.reward_function_version,
        )
    )
    evaluation_callback = FixedSeedEvalCallback(
        eval_env,
        eval_freq=config.evaluation.frequency,
        n_eval_episodes=config.evaluation.episodes,
        seed=config.runtime.seed + config.evaluation.seed_offset,
        log_path=run_dir / "evaluations",
        best_success_model_save_path=run_dir,
        task_name=config.continual.tasks[0],
        task_set_seed=config.evaluation.task_set_seed,
        evaluation_mode=config.evaluation.mode,
        early_stopping=config.sac.early_stopping,
        best_success_model_name="best_model",
        atomic_model_replacement=True,
    )
    target_entropy = target_entropy_from_output_std(
        config.sac.target_output_std, int(np.prod(train_env.action_space.shape))
    )
    model = LegacyScheduleSAC(
        LegacySACPolicy,
        train_env,
        learning_rate=config.sac.learning_rate,
        buffer_size=config.sac.buffer_size,
        learning_starts=config.sac.learning_starts,
        random_exploration_steps=config.sac.random_exploration_steps,
        batch_size=config.sac.batch_size,
        gamma=config.sac.gamma,
        tau=config.sac.tau,
        update_interval=config.sac.train_freq,
        gradient_steps=config.sac.gradient_steps,
        ent_coef=SAC_ENT_COEF,
        target_entropy=target_entropy,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=config.runtime.seed,
        device=config.runtime.device,
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=config.sac.steps,
            callback=evaluation_callback,
            tb_log_name="SAC",
            progress_bar=True,
        )
        best_path = run_dir / "best_model.zip"
        print(f"Best model saved to: {best_path}")
        if (
            isinstance(evaluation_callback.results, list)
            and evaluation_callback.results
        ):
            mean_returns = np.mean(evaluation_callback.results, axis=1)
            success_rates = np.mean(evaluation_callback.successes, axis=1)
            success_index = best_success_index(
                success_rates,
                mean_returns,
            )

            selected_qualification = qualification_record(
                success_rate=float(
                    np.mean(evaluation_callback.successes[success_index])
                ),
                mean_return=float(np.mean(evaluation_callback.results[success_index])),
                return_std=float(evaluation_callback.return_stddevs[success_index]),
                episodes=config.evaluation.episodes,
                task_set_seed=config.evaluation.task_set_seed,
                evaluation_seed=config.runtime.seed + config.evaluation.seed_offset,
                deterministic=True,
                task_hashes=list(evaluation_callback.task_hashes),
            )
            config_path = run_dir / "config.yaml"
            write_expert_manifest(
                best_path,
                task_name=config.continual.tasks[0],
                backend=config.environment.backend,
                reward_function_version=config.environment.reward_function_version,
                training_seed=config.runtime.seed,
                training_config_reference=str(config_path),
                training_config_sha256=file_hash(config_path),
                observation_shape=OBSERVATION_SHAPE,
                action_shape=ACTION_SHAPE,
                horizon=horizon_for_backend(config.environment.backend),
                qualification=selected_qualification,
                policy_architecture=SAC_POLICY_ARCHITECTURE,
            )
        else:
            raise RuntimeError("SAC evaluation produced no checkpoint-selection data.")

        selected = SAC.load(best_path, device=config.runtime.device)
        results = {
            "schema_version": 1,
            "actual_final_timestep": model.num_timesteps,
            "target_timestep": config.sac.steps,
            "early_stopping": early_stopping_metadata(
                config.sac.early_stopping, evaluation_callback
            ),
            "selected_expert": {
                "path": str(best_path.relative_to(run_dir)),
                "sha256": file_hash(best_path),
            },
            "final_evaluation": final_test(config, selected),
        }
        with (run_dir / "results.json").open("x", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2, allow_nan=False)
            stream.write("\n")
        complete_run(run_dir)
        print(f"Run directory: {run_dir}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
