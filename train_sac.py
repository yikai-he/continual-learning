"""Train a reproducible Stable-Baselines3 SAC baseline on MetaWorld MT1."""

from __future__ import annotations

import json
import math
import os
import zipfile
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
from src.sac_actor_init import initialize_sac_actor
from continual_manipulation.evaluation import (
    disable_task_sampling,
    evaluate_task,
    fixed_mt1_tasks,
    validate_active_task,
)
from src.continual.policy_adapter import SB3SACPolicy
from continual_manipulation.task_banks import backend_tasks, bank_hashes
from continual_manipulation.sequences import TaskSpec
from continual_manipulation.environments import (
    ACTION_SHAPE, OBSERVATION_SHAPE, horizon_for_backend, make_env,
)
from src.support.expert_manifest import qualification_record, write_expert_manifest
from src.support.expert_manifest import validate_expert_checkpoint, load_expert_manifest
from src.support.reproducibility import file_hash
from src.support.run_manifest import complete_run, create_run_manifest
from src.support.io import atomic_replace_json
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


def fixed_evaluation_schedule(tasks, identities, episodes: int, first_seed: int) -> dict:
    """Build a deterministic episode schedule that may repeat one fixed bank."""
    if episodes <= 0:
        raise ValueError("Fixed evaluation episode count must be positive.")
    if not tasks:
        raise ValueError("Fixed evaluation task bank is empty.")
    if len(tasks) != len(identities):
        raise ValueError("Fixed evaluation tasks and identities are misaligned.")
    hashes = []
    for identity in identities:
        digest = identity.get("task_data_sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Fixed evaluation task identity has no SHA-256 digest.")
        hashes.append(digest)
    indices = tuple(index % len(tasks) for index in range(episodes))
    return {
        "task_bank_size": len(tasks),
        "task_bank_repetitions": int(math.ceil(episodes / len(tasks))),
        "task_bank_indices": indices,
        "reset_seeds": tuple(range(first_seed, first_seed + episodes)),
        "unique_task_hashes": tuple(hashes),
        "episode_task_hashes": tuple(hashes[index] for index in indices),
    }


def validate_fixed_evaluation_preflight(
    eval_env,
    *,
    task_name: str,
    task_set_seed: int,
    episodes: int,
    first_reset_seed: int,
    backend: str = "metaworld-v3",
) -> dict:
    """Validate repeated fixed-bank selection and deterministic reset behavior."""
    tasks, identities = fixed_mt1_tasks(task_name, task_set_seed, backend=backend)
    schedule = fixed_evaluation_schedule(tasks, identities, episodes, first_reset_seed)
    disable_task_sampling(eval_env)
    first_index = schedule["task_bank_indices"][0]
    first_task = tasks[first_index]
    eval_env.unwrapped.set_task(first_task)
    first_observation, _ = eval_env.reset(seed=first_reset_seed)
    validate_active_task(eval_env, first_task, backend)
    eval_env.unwrapped.set_task(first_task)
    repeated_observation, _ = eval_env.reset(seed=first_reset_seed)
    validate_active_task(eval_env, first_task, backend)
    if not np.array_equal(first_observation, repeated_observation):
        raise ValueError(
            "Fixed evaluation reset is not reproducible for an identical "
            "task-configuration/reset-seed pair."
        )
    second_pass_seed_effective = None
    if episodes > len(tasks):
        repeated_index = len(tasks)
        repeated_task = tasks[schedule["task_bank_indices"][repeated_index]]
        eval_env.unwrapped.set_task(repeated_task)
        second_observation, _ = eval_env.reset(
            seed=schedule["reset_seeds"][repeated_index]
        )
        validate_active_task(eval_env, repeated_task, backend)
        second_pass_seed_effective = not np.array_equal(
            first_observation, second_observation
        )
    return {
        **schedule,
        "same_pair_reset_reproducible": True,
        "second_pass_reset_seed_changed_observation": second_pass_seed_effective,
    }


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
        checkpoint_context: dict | None = None,
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
        self.checkpoint_context = checkpoint_context
        self.best_checkpoint: dict | None = None
        self.consecutive_successful_evaluations = 0
        self.training_start_timestep: int | None = None
        self.best_plateau_success = -np.inf
        self.plateau_evaluations = 0
        self.stopping_reason: str | None = None
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
            schedule = fixed_evaluation_schedule(
                self.fixed_tasks, identities, n_eval_episodes, seed
            )
            self.task_bank_size = schedule["task_bank_size"]
            self.task_bank_repetitions = schedule["task_bank_repetitions"]
            self.task_bank_indices = schedule["task_bank_indices"]
            self.unique_task_hashes = schedule["unique_task_hashes"]
            self.task_hashes = schedule["episode_task_hashes"]
            disable_task_sampling(eval_env)
        elif evaluation_mode == "sampled":
            self.fixed_tasks = None
            self.task_hashes = ()
            self.unique_task_hashes = ()
            self.task_bank_size = None
            self.task_bank_repetitions = None
            self.task_bank_indices = ()
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
        if self.checkpoint_context is not None:
            self.best_checkpoint = save_sac_checkpoint(
                self.model,
                path.parent,
                "best",
                self.checkpoint_context,
            )
            return
        if not self.atomic_model_replacement:
            self.model.save(path)
            return
        target = path if path.suffix == ".zip" else path.with_suffix(".zip")
        atomic_replace_model(self.model, target)

    def _evaluate(self) -> tuple[list[float], list[int], list[bool]]:
        returns, lengths, successes = [], [], []
        for index, seed in enumerate(self.seeds):
            if self.fixed_tasks is not None:
                task_index = self.task_bank_indices[index]
                selected_task = self.fixed_tasks[task_index]
                self.eval_env.unwrapped.set_task(selected_task)
            observation, _ = self.eval_env.reset(seed=seed)
            if self.fixed_tasks is not None:
                validate_active_task(
                    self.eval_env,
                    selected_task,
                    "kuka-v3" if self.task_name.startswith("kuka-") else "metaworld-v3",
                )
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
            unique_task_data_sha256=np.asarray(self.unique_task_hashes),
            task_bank_size=np.asarray(
                -1 if self.task_bank_size is None else self.task_bank_size
            ),
            task_bank_repetitions=np.asarray(
                -1 if self.task_bank_repetitions is None else self.task_bank_repetitions
            ),
            task_bank_indices=np.asarray(self.task_bank_indices, dtype=np.int64),
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
        print(
            f"SAC progress step={self.num_timesteps} "
            f"success_rate={success_rate:.3f} mean_return={mean_return:.3f}",
            flush=True,
        )
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
        if success_rate > self.best_plateau_success + self.early_stopping.min_delta:
            self.best_plateau_success = success_rate
            self.plateau_evaluations = 0
        else:
            self.plateau_evaluations += 1
        start = self.training_start_timestep or 0
        additional_steps = self.num_timesteps - start
        if (
            additional_steps >= self.early_stopping.min_steps
            and self.consecutive_successful_evaluations >= self.early_stopping.patience
        ):
            self.early_stopped = True
            self.stop_timestep = self.num_timesteps
            self.trigger_success_rate = success_rate
            self.stopping_reason = "target-success"
            return False
        if (
            additional_steps >= self.early_stopping.min_steps
            and self.early_stopping.plateau_patience is not None
            and self.plateau_evaluations >= self.early_stopping.plateau_patience
        ):
            self.early_stopped = True
            self.stop_timestep = self.num_timesteps
            self.trigger_success_rate = success_rate
            self.stopping_reason = "validation-plateau"
            return False
        return True

    def record_maximum_budget_completion(self, maximum_additional_steps: int) -> None:
        """Persist normal completion at the configured additional-step ceiling."""
        if self.stopping_reason is not None:
            return
        start = self.training_start_timestep or 0
        additional_steps = self.num_timesteps - start
        if additional_steps < maximum_additional_steps:
            raise RuntimeError(
                "SAC training ended before its maximum budget without an Early "
                "Stopping reason."
            )
        self.stop_timestep = self.num_timesteps
        self.stopping_reason = "maximum-training-steps"

    def _on_training_start(self) -> None:
        """Initialize the global schedule and optionally evaluate a fresh policy."""
        current = self.model.num_timesteps
        if self.training_start_timestep is None:
            self.training_start_timestep = current
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


def _checkpoint_paths(run_dir: Path, name: str) -> tuple[Path, Path, Path]:
    """Return the stable model, replay, and metadata paths for one SAC snapshot."""
    return (
        run_dir / f"{name}_model.zip",
        run_dir / f"{name}_replay_buffer.pkl",
        run_dir / f"{name}_checkpoint.json",
    )


def _replay_buffer_metadata(model) -> dict:
    buffer = model.replay_buffer
    return {
        "class": f"{type(buffer).__module__}.{type(buffer).__name__}",
        "capacity": int(buffer.buffer_size),
        "stored_transitions": int(buffer.size()),
    }


def save_sac_checkpoint(
    model,
    run_dir: Path,
    name: str,
    context: dict,
    *,
    status: str = "complete",
) -> dict:
    """Atomically replace one synchronized model/replay/metadata checkpoint.

    Temporary files are fully serialized and hashed before publication. If a
    replacement fails, the previous complete trio is restored.
    """
    if name not in ("best", "final"):
        raise ValueError("SAC checkpoint name must be best or final.")
    run_dir = Path(run_dir)
    model_path, replay_path, metadata_path = _checkpoint_paths(run_dir, name)
    token = f"{os.getpid()}.{name}"
    staged_model = run_dir / f".{token}.model.tmp.zip"
    staged_replay = run_dir / f".{token}.replay.tmp.pkl"
    staged_metadata = run_dir / f".{token}.checkpoint.tmp.json"
    targets = (model_path, replay_path, metadata_path)
    staged = (staged_model, staged_replay, staged_metadata)
    backups = tuple(path.with_name(f".{path.name}.{token}.backup") for path in targets)
    try:
        model.save(staged_model)
        model.save_replay_buffer(staged_replay)
        if not zipfile.is_zipfile(staged_model):
            raise RuntimeError("SAC model checkpoint is not a valid ZIP archive.")
        if not staged_replay.is_file() or staged_replay.stat().st_size == 0:
            raise RuntimeError("SAC replay-buffer serialization produced no data.")
        metadata = {
            "schema_version": 1,
            "kind": "sac_model_replay_checkpoint_v1",
            "name": name,
            "status": status,
            "checkpoint_timestep": int(model.num_timesteps),
            "gradient_updates": int(getattr(model, "_n_updates", 0)),
            "task_name": context["task_name"],
            "backend": context["backend"],
            "reward_function_version": context["reward_function_version"],
            "observation_shape": list(model.observation_space.shape),
            "action_shape": list(model.action_space.shape),
            "replay_buffer": _replay_buffer_metadata(model),
            "model": {"path": model_path.name, "sha256": file_hash(staged_model)},
            "replay": {"path": replay_path.name, "sha256": file_hash(staged_replay)},
            "training_config_sha256": context["training_config_sha256"],
        }
        with staged_metadata.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")

        backed_up = []
        published = []
        try:
            for target, backup in zip(targets, backups):
                if target.exists():
                    os.replace(target, backup)
                    backed_up.append((target, backup))
            for source, target in zip(staged, targets):
                os.replace(source, target)
                published.append(target)
        except Exception:
            for target in published:
                target.unlink(missing_ok=True)
            for target, backup in reversed(backed_up):
                if backup.exists():
                    os.replace(backup, target)
            raise
        for backup in backups:
            backup.unlink(missing_ok=True)
        return metadata
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def validate_replay_checkpoint_pair(
    model_path: Path,
    replay_path: Path,
    metadata_path: Path,
    config,
    model,
) -> dict:
    """Verify and load a formal same-task model/replay checkpoint pair."""
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Replay-backed continuation requires checkpoint metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = (
        (metadata.get("kind") == "sac_model_replay_checkpoint_v1", "checkpoint kind"),
        (metadata.get("status") == "complete", "checkpoint status"),
        (metadata.get("task_name") == config.continual.tasks[0], "task identity"),
        (metadata.get("backend") == config.environment.backend, "backend"),
        (
            metadata.get("reward_function_version")
            == config.environment.reward_function_version,
            "reward version",
        ),
        (tuple(metadata.get("observation_shape", ())) == OBSERVATION_SHAPE, "observation shape"),
        (tuple(metadata.get("action_shape", ())) == ACTION_SHAPE, "action shape"),
        (metadata.get("model", {}).get("path") == model_path.name, "model path"),
        (metadata.get("replay", {}).get("path") == replay_path.name, "replay path"),
        (metadata.get("model", {}).get("sha256") == file_hash(model_path), "model SHA-256"),
        (metadata.get("replay", {}).get("sha256") == file_hash(replay_path), "replay SHA-256"),
        (metadata.get("checkpoint_timestep") == int(model.num_timesteps), "checkpoint timestep"),
    )
    for valid, label in checks:
        if not valid:
            raise ValueError(f"SAC model/replay checkpoint {label} mismatch.")
    source_config = model_path.parent / "config.yaml"
    if source_config.is_file() and metadata.get("training_config_sha256") != file_hash(source_config):
        raise ValueError("SAC model/replay checkpoint training configuration mismatch.")
    model.load_replay_buffer(replay_path)
    replay = metadata.get("replay_buffer", {})
    buffer = model.replay_buffer
    loaded_checks = (
        (
            f"{type(buffer).__module__}.{type(buffer).__name__}" == replay.get("class"),
            "class",
        ),
        (int(buffer.buffer_size) == replay.get("capacity"), "capacity"),
        (int(buffer.size()) == replay.get("stored_transitions"), "stored transition count"),
        (tuple(buffer.observation_space.shape) == OBSERVATION_SHAPE, "buffer observation shape"),
        (tuple(buffer.action_space.shape) == ACTION_SHAPE, "buffer action shape"),
    )
    for valid, label in loaded_checks:
        if not valid:
            raise ValueError(f"Loaded SAC replay buffer {label} mismatch.")
    return metadata


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
    episodes = config.evaluation.final_episodes or config.evaluation.episodes
    return [
        evaluate_task(
            policy,
            TaskSpec(task, task),
            episodes=episodes,
            seed=(
                config.evaluation.final_reset_seed + index * episodes
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
        "plateau_patience": config.plateau_patience,
        "min_delta": config.min_delta,
        "triggered": observed.early_stopped if observed is not None else False,
        "stop_timestep": observed.stop_timestep if observed is not None else None,
        "trigger_success_rate": (
            observed.trigger_success_rate if observed is not None else None
        ),
        "consecutive_success_count": (
            observed.consecutive_successful_evaluations if observed is not None else 0
        ),
        "stopping_reason": observed.stopping_reason if observed is not None else None,
        "additional_steps_at_stop": (
            observed.stop_timestep - (observed.training_start_timestep or 0)
            if observed is not None and observed.stop_timestep is not None
            else None
        ),
    }


def initialize_sac_training(config, train_env, run_dir: Path):
    """Create scratch SAC or restore the serialized model state for continuation."""
    if not config.resume.enabled:
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
        actor = initialize_sac_actor(
            model, config.actor_init, base_dir=Path(__file__).resolve().parent
        )
        return model, actor, {
            "enabled": False,
            "mode": "scratch",
            "restored": [],
            "reset": ["all SAC training state"],
            "source_checkpoint": None,
        }

    source = Path(config.resume.checkpoint).expanduser().resolve()
    manifest_path = source.parent / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("Resume requires a checkpoint-specific expert manifest.")
    manifest = load_expert_manifest(manifest_path)
    reward_configuration = manifest.get("reward_configuration")
    if reward_configuration not in (None, "v2 original hammer reward; nail progress weight 0.0"):
        raise ValueError(
            "Resume source uses a reward configuration that the SAC training factory "
            "cannot reconstruct safely."
        )
    validate_expert_checkpoint(
        source,
        expected_task=config.continual.tasks[0],
        expected_backend=config.environment.backend,
        expected_reward_function_version=config.environment.reward_function_version,
        expected_observation_shape=OBSERVATION_SHAPE,
        expected_action_shape=ACTION_SHAPE,
        expected_horizon=horizon_for_backend(config.environment.backend),
        manifest_path=manifest_path,
        allow_legacy=False,
    )
    model = LegacyScheduleSAC.load(
        source,
        env=train_env,
        device=config.runtime.device,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    source_timestep = int(model.num_timesteps)
    model.set_random_seed(config.runtime.seed)
    replay = config.resume.replay_buffer
    replay_metadata = None
    if replay is not None:
        replay_path = Path(replay).expanduser().resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"Resume replay buffer not found: {replay_path}")
        if replay_path.parent != source.parent:
            raise ValueError("Formal model-and-replay continuation requires colocated files.")
        expected = {
            "best_model.zip": ("best_replay_buffer.pkl", "best_checkpoint.json"),
            "final_model.zip": ("final_replay_buffer.pkl", "final_checkpoint.json"),
        }.get(source.name)
        if expected is None or replay_path.name != expected[0]:
            raise ValueError("Resume model and replay-buffer filenames do not form a matched pair.")
        metadata_path = source.parent / expected[1]
        replay_metadata = validate_replay_checkpoint_pair(
            source, replay_path, metadata_path, config, model
        )
        replay_record = str(replay_path)
        replay_restored = True
    else:
        replay_record = None
        replay_restored = False
    actor = {
        "schema_version": 1,
        "mode": "scratch",
        "reason": "same-task SAC continuation; actor initialization is not used",
    }
    provenance = {
        "enabled": True,
        "mode": "model-and-replay" if replay_restored else "model-state-only",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_hash(source),
        "source_manifest": str(manifest_path.resolve()),
        "source_policy_architecture": manifest.get("policy_architecture"),
        "source_timestep": source_timestep,
        "source_gradient_updates": int(getattr(model, "_n_updates", 0)),
        "replay_buffer": replay_record,
        "replay_checkpoint_metadata": replay_metadata,
        "restored": [
            "actor parameters", "critic parameters", "target critic parameters",
            "optimizer states", "entropy coefficient and optimizer", "timestep counters",
        ] + (["replay buffer"] if replay_restored else []),
        "reset": (
            [] if replay_restored else ["replay buffer"]
        ) + [
            "random-number-generator state (reseeded for continuation)",
            "environment state",
            "evaluation and checkpoint-selection callback state",
        ],
        "exact_process_resume": False,
        "effective_loaded_hyperparameters": {
            "learning_rate": model.learning_rate,
            "buffer_size": model.buffer_size,
            "learning_starts": model.learning_starts,
            "batch_size": model.batch_size,
            "gamma": model.gamma,
            "tau": model.tau,
            "gradient_steps": model.gradient_steps,
            "random_exploration_steps": getattr(model, "random_exploration_steps", None),
            "update_interval": getattr(model, "update_interval", None),
        },
    }
    return model, actor, provenance


def refill_resume_replay_buffer(model, steps: int, *, additional_budget: int) -> dict:
    """Collect stochastic transitions without invoking SAC optimization."""
    if steps < 0 or steps >= additional_budget:
        raise ValueError("Refill steps must be nonnegative and smaller than the budget.")
    before_steps = int(model.num_timesteps)
    before_updates = int(getattr(model, "_n_updates", 0))
    before_size = int(model.replay_buffer.size())
    if steps == 0:
        return {
            "requested_steps": 0,
            "collected_steps": 0,
            "buffer_size_before": before_size,
            "buffer_size_after": before_size,
            "completed": True,
        }
    env = model.get_env()
    if env is None or env.num_envs != 1:
        raise ValueError("Replay refill requires one attached vectorized environment.")
    model._last_obs = env.reset()
    for _ in range(steps):
        actions, buffer_actions = model._sample_action(
            learning_starts=0, action_noise=None, n_envs=env.num_envs
        )
        new_obs, rewards, dones, infos = env.step(actions)
        model.num_timesteps += env.num_envs
        model._store_transition(
            model.replay_buffer,
            buffer_actions,
            new_obs,
            rewards,
            dones,
            infos,
        )
    if int(getattr(model, "_n_updates", 0)) != before_updates:
        raise RuntimeError("SAC optimizer updates occurred during replay refill.")
    collected = int(model.num_timesteps) - before_steps
    if collected != steps:
        raise RuntimeError("Replay refill timestep accounting mismatch.")
    return {
        "requested_steps": steps,
        "collected_steps": collected,
        "buffer_size_before": before_size,
        "buffer_size_after": int(model.replay_buffer.size()),
        "completed": True,
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
    checkpoint_context = {
        "task_name": config.continual.tasks[0],
        "backend": config.environment.backend,
        "reward_function_version": config.environment.reward_function_version,
        "training_config_sha256": file_hash(run_dir / "config.yaml"),
    }
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
        checkpoint_context=checkpoint_context,
    )
    model, actor_initialization, resume_provenance = initialize_sac_training(
        config, train_env, run_dir
    )
    source_timestep = int(model.num_timesteps)
    refill = (
        refill_resume_replay_buffer(
            model,
            config.resume.refill_steps,
            additional_budget=config.sac.steps,
        )
        if config.resume.enabled and config.resume.replay_buffer is None
        else {
            "requested_steps": 0,
            "collected_steps": 0,
            "buffer_size_before": int(model.replay_buffer.size()),
            "buffer_size_after": int(model.replay_buffer.size()),
            "completed": True,
        }
    )
    resume_provenance["refill"] = refill
    evaluation_callback.training_start_timestep = source_timestep
    with (run_dir / "actor_initialization.json").open("x", encoding="utf-8") as stream:
        json.dump(actor_initialization, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Actor initialization: {config.actor_init.mode}")

    try:
        learn_kwargs = dict(
            total_timesteps=config.sac.steps - refill["collected_steps"],
            callback=evaluation_callback,
            tb_log_name="SAC",
            progress_bar=False,
        )
        if config.resume.enabled:
            learn_kwargs["reset_num_timesteps"] = False
        model.learn(**learn_kwargs)
        evaluation_callback.num_timesteps = int(model.num_timesteps)
        evaluation_callback.record_maximum_budget_completion(config.sac.steps)
        final_checkpoint = save_sac_checkpoint(
            model,
            run_dir,
            "final",
            checkpoint_context,
            status="pending-validation",
        )
        if (
            evaluation_callback.timesteps
            and evaluation_callback.timesteps[-1] == int(model.num_timesteps)
        ):
            final_returns = evaluation_callback.results[-1]
            final_lengths = evaluation_callback.ep_lengths[-1]
            final_successes = evaluation_callback.successes[-1]
        else:
            final_returns, final_lengths, final_successes = evaluation_callback._evaluate()
        final_policy_validation = {
            "success_rate": float(np.mean(final_successes)),
            "success_count": int(np.sum(final_successes)),
            "mean_return": float(np.mean(final_returns)),
            "std_return": float(np.std(final_returns)),
            "mean_episode_length": float(np.mean(final_lengths)),
        }
        architecture = resume_provenance.get("source_policy_architecture")
        if isinstance(architecture, dict):
            architecture = architecture.get("identity") or architecture.get("policy_class")
        config_path = run_dir / "config.yaml"
        write_expert_manifest(
            run_dir / "final_model.zip",
            task_name=config.continual.tasks[0],
            backend=config.environment.backend,
            reward_function_version=config.environment.reward_function_version,
            training_seed=config.runtime.seed,
            training_config_reference=str(config_path),
            training_config_sha256=file_hash(config_path),
            observation_shape=OBSERVATION_SHAPE,
            action_shape=ACTION_SHAPE,
            horizon=horizon_for_backend(config.environment.backend),
            qualification=None,
            policy_architecture=architecture or SAC_POLICY_ARCHITECTURE,
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
                policy_architecture=architecture or SAC_POLICY_ARCHITECTURE,
            )
        else:
            raise RuntimeError("SAC evaluation produced no checkpoint-selection data.")

        selected = SAC.load(best_path, device=config.runtime.device)
        results = {
            "schema_version": 1,
            "actor_initialization": actor_initialization,
            "resume": resume_provenance,
            "source_policy_validation": (
                {
                    "success_rate": float(np.mean(evaluation_callback.successes[0])),
                    "mean_return": float(np.mean(evaluation_callback.results[0])),
                    "timestep": int(resume_provenance["source_timestep"]),
                }
                if config.resume.enabled else None
            ),
            "final_policy_validation": final_policy_validation,
            "actual_final_timestep": model.num_timesteps,
            "additional_training_steps": model.num_timesteps - resume_provenance.get("source_timestep", 0),
            "total_effective_environment_steps": model.num_timesteps,
            "additional_gradient_updates": int(getattr(model, "_n_updates", 0))
            - int(resume_provenance.get("source_gradient_updates", 0)),
            "target_additional_timesteps": config.sac.steps,
            "target_timestep": (
                resume_provenance.get("source_timestep", 0) + config.sac.steps
            ),
            "early_stopping": early_stopping_metadata(
                config.sac.early_stopping, evaluation_callback
            ),
            "selected_expert": {
                "path": str(best_path.relative_to(run_dir)),
                "sha256": file_hash(best_path),
            },
            "best_checkpoint": evaluation_callback.best_checkpoint,
            "final_checkpoint": {**final_checkpoint, "status": "complete"},
            "final_evaluation": final_test(config, selected),
        }
        with (run_dir / "results.json").open("x", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2, allow_nan=False)
            stream.write("\n")
        final_checkpoint["status"] = "complete"
        atomic_replace_json(
            run_dir / "final_checkpoint.json", final_checkpoint, allow_nan=False
        )
        complete_run(run_dir)
        print(f"Run directory: {run_dir}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
