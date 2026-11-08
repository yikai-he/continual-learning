"""Train a reproducible Stable-Baselines3 SAC baseline on MetaWorld MT1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import deque
from datetime import datetime
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
from src.config import (
    EarlyStoppingConfig,
    ExperimentConfig,
    SAC_ENT_COEF,
    SAC_NET_ARCH,
    output_directory,
    parse_config,
    save_resolved_config,
)
from src.continual.evaluation import disable_task_sampling, evaluate_task, fixed_mt1_tasks
from src.continual.policy_adapter import SB3SACPolicy
from src.continual.task_bank import backend_tasks, bank_hashes
from src.continual.task_sequence import TaskSpec
from src.envs import ACTION_SHAPE, OBSERVATION_SHAPE, horizon_for_backend, make_env
from src.support.reproducibility import git_output
from src.sac.checkpoint_ranking import (
    best_return_index,
    best_success_index,
    success_score,
)
from src.support.expert_manifest import qualification_record, write_expert_manifest
from src.support.reproducibility import file_hash
from src.support.run_manifest import complete_run, create_run_manifest
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor


def export_run_summaries(run_dir: Path) -> None:
    """Write compact reports from an existing SAC evaluation archive."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    archive_path = run_dir / "evaluations" / "evaluations.npz"
    with np.load(archive_path) as archive:
        steps = np.asarray(archive["timesteps"], dtype=int)
        returns = np.asarray(archive["results"], dtype=float)
        successes = np.asarray(archive["successes"], dtype=float)
        return_stds = (
            np.asarray(archive["return_stddevs"], dtype=float)
            if "return_stddevs" in archive
            else np.std(returns, axis=1)
        )

    if (
        not len(steps)
        or returns.shape[0] != len(steps)
        or successes.shape[0] != len(steps)
    ):
        raise ValueError("Evaluation archive is empty or has inconsistent row counts.")
    mean_returns = np.mean(returns, axis=1)
    success_rates = np.mean(successes, axis=1)
    rows = [
        {
            "step": int(steps[i]),
            "success_rate": float(success_rates[i]),
            "mean_return": float(mean_returns[i]),
            "return_std": float(return_stds[i]),
        }
        for i in range(len(steps))
    ]
    return_index = best_return_index(mean_returns)
    success_index = best_success_index(success_rates, mean_returns)

    with (run_dir / "evaluation_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def checkpoint(index: int, model_path: str) -> dict:
        return {**rows[index], "model_path": model_path}

    summary = {
        "task": metadata["task"],
        "backend": metadata["env_backend"],
        "seed": metadata["seed"],
        "total_timesteps": metadata["steps"],
        "reward_function_version": metadata["reward_function_version"],
        "horizon": metadata["effective_horizon"],
        "initial": {key: rows[0][key] for key in rows[0] if key != "step"},
        "final": rows[-1],
        "best_success": checkpoint(
            success_index, "best_success_model/best_success_model.zip"
        ),
        "best_return": checkpoint(return_index, "best_model/best_model.zip"),
        "artifacts": {
            "final_model": "final_model.zip",
            "evaluations_npz": "evaluations/evaluations.npz",
            "tensorboard": "tensorboard",
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    table = [
        "| Step | Success | Mean Return | Std |",
        "|---:|---:|---:|---:|",
        *(
            f"| {row['step']} | {row['success_rate']:.1%} | "
            f"{row['mean_return']:.4f} | {row['return_std']:.4f} |"
            for row in rows
        ),
    ]
    markdown = [
        "# SAC Experiment Summary",
        "",
        f"- Task: {summary['task']}",
        f"- Backend: {summary['backend']}",
        f"- Seed: {summary['seed']}",
        f"- Steps: {summary['total_timesteps']}",
        f"- Reward: {summary['reward_function_version']}",
        f"- Horizon: {summary['horizon']}",
        "",
        "## Results",
        "",
        *table,
        "",
        "## Best Checkpoints",
        "",
        f"- Best success: step {rows[success_index]['step']} "
        f"({rows[success_index]['success_rate']:.1%}), "
        "`best_success_model/best_success_model.zip`",
        f"- Best return: step {rows[return_index]['step']} "
        f"({rows[return_index]['mean_return']:.4f}), `best_model/best_model.zip`",
        "- Final model: `final_model.zip`",
        "",
    ]
    (run_dir / "EXPERIMENT_SUMMARY.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )


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
        task_name: str | None = None,
        task_set_seed: int | None = None,
        evaluation_mode: str = "sampled",
        evaluate_on_training_start: bool = True,
        early_stopping: EarlyStoppingConfig | None = None,
        save_best_return: bool = True,
        best_success_model_name: str = "best_success_model",
        atomic_model_replacement: bool = False,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seeds = tuple(range(seed, seed + n_eval_episodes))
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.best_success_model_save_path = best_success_model_save_path
        self.evaluation_mode = evaluation_mode
        self.task_set_seed = task_set_seed
        self.task_name = task_name
        self.evaluate_on_training_start = evaluate_on_training_start
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.save_best_return = save_best_return
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
                    {"task_data_sha256": hashlib.sha256(task.data).hexdigest()}
                    for task in self.fixed_tasks
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
        self.best_mean_return = -np.inf
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
        if self.save_best_return and mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            self._save_model(self.best_model_save_path / "best_model")
        elif mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
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


class GlobalCheckpointCallback(CheckpointCallback):
    """Save SB3 checkpoints on global rather than callback-local boundaries."""

    def _on_training_start(self) -> None:
        self._next_checkpoint_timestep = next_global_boundary(
            self.model.num_timesteps, self.save_freq
        )

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_checkpoint_timestep:
            return True
        model_path = self._checkpoint_path(extension="zip")
        self.model.save(model_path)
        if (
            self.save_replay_buffer
            and hasattr(self.model, "replay_buffer")
            and self.model.replay_buffer is not None
        ):
            replay_path = self._checkpoint_path("replay_buffer_", extension="pkl")
            self.model.save_replay_buffer(replay_path)
        self._next_checkpoint_timestep = next_global_boundary(
            self.num_timesteps, self.save_freq
        )
        return True


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


class LatestModelCallback(BaseCallback):
    """Atomically retain only the latest inference checkpoint."""

    def __init__(self, save_freq: int, path: Path) -> None:
        super().__init__()
        self.save_freq = save_freq
        self.path = Path(path)
        self._next_checkpoint_timestep = None

    def _on_training_start(self) -> None:
        self._next_checkpoint_timestep = next_global_boundary(
            self.model.num_timesteps, self.save_freq
        )

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_checkpoint_timestep:
            return True
        atomic_replace_model(self.model, self.path)
        self._next_checkpoint_timestep = next_global_boundary(
            self.num_timesteps, self.save_freq
        )
        return True


def task_bank_provenance(task: str, training_seed: int, validation_seed: int,
                         final_seeds: list[int], *, backend: str) -> dict:
    """Record task identities and prove all optimization/evaluation banks disjoint."""
    seeds = [training_seed, validation_seed, *final_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training, validation, and final task-bank seeds must differ.")
    banks = {
        str(seed): bank_hashes(backend_tasks(task, seed, backend=backend))
        for seed in seeds
    }
    for index, left in enumerate(seeds):
        for right in seeds[index + 1:]:
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
            policy, TaskSpec(task, task), episodes=config.evaluation.episodes,
            seed=(config.evaluation.final_reset_seed + index * config.evaluation.episodes
                  if fixed else bank_seed),
            evaluation_mode="fixed-tasks" if fixed else "sampled",
            task_set_seed=bank_seed,
            reward_function_version=config.environment.reward_function_version,
            backend=config.environment.backend,
        )
        for index, bank_seed in enumerate(config.evaluation.final_task_set_seeds)
    ]


def _git_identity(path: Path) -> dict[str, str | None]:
    """Read local Git provenance without changing repository state."""
    return {
        "branch": git_output(path, "branch", "--show-current"),
        "commit": git_output(path, "rev-parse", "HEAD"),
    }


def _runtime_provenance(backend: str) -> dict:
    """Capture package, registration, and editable-checkout provenance."""
    import metaworld

    checkout = Path(metaworld.__file__).resolve().parent.parent
    registration_ids = {
        "metaworld-v3": "Meta-World/MT1",
        "kuka-v3": "Meta-World-KUKA/MT1",
    }
    return {
        "environment_registration_id": registration_ids[backend],
        "metaworld_version": package_metadata.version("metaworld"),
        "gymnasium_version": package_metadata.version("gymnasium"),
        "mujoco_version": package_metadata.version("mujoco"),
        "metaworld_checkout": str(checkout),
        "metaworld_git": _git_identity(checkout),
    }


def parse_args(argv=None) -> tuple[ExperimentConfig, object]:
    """Parse and validate the SAC YAML configuration plus CLI overrides."""

    def configure(parser) -> None:
        parser.add_argument(
            "--resume-from",
            type=Path,
            default=None,
            metavar="PATH",
            help=(
                "Resume from an SB3 SAC checkpoint and its matching replay "
                "buffer; sac.steps remains the target total timestep count."
            ),
        )

    return parse_config(
        "sac",
        __doc__,
        argv,
        configure_parser=configure,
        return_args=True,
    )


def replay_buffer_path(checkpoint_path: Path) -> Path:
    """Return the replay-buffer sidecar produced by SB3 CheckpointCallback."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.suffix != ".zip":
        raise ValueError("Resume checkpoint must be an SB3 .zip model checkpoint.")
    match = re.fullmatch(
        r"(?P<prefix>.+)_(?P<step>\d+)_steps\.zip", checkpoint_path.name
    )
    if match is None:
        raise ValueError(
            "Resume checkpoint name must match '<prefix>_<step>_steps.zip' so "
            "its SB3 replay buffer can be identified safely."
        )
    expected = checkpoint_path.with_name(
        f"{match['prefix']}_replay_buffer_{match['step']}_steps.pkl"
    )
    candidates = list(
        checkpoint_path.parent.glob(
            f"{match['prefix']}_replay_buffer_{match['step']}_steps*.pkl"
        )
    )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple ambiguous replay-buffer candidates found for checkpoint "
            f"{checkpoint_path}: {', '.join(str(path) for path in candidates)}"
        )
    if candidates and candidates[0] != expected:
        raise ValueError(
            "Replay-buffer candidate does not use SB3's standard checkpoint "
            f"name; expected: {expected}"
        )
    return expected


def remaining_timesteps(target: int, completed: int) -> int:
    """Compute a positive remaining budget for total-timestep resume semantics."""
    if completed >= target:
        raise ValueError(
            f"Checkpoint already has {completed} timesteps, which meets or exceeds "
            f"the target total of {target}; refusing to train additional steps."
        )
    return target - completed


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


def load_resumable_sac(checkpoint_path: Path, env, *, device: str):
    """Load a SAC checkpoint and require its matching replay-buffer sidecar."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    buffer_path = replay_buffer_path(checkpoint_path)
    if not buffer_path.is_file():
        raise FileNotFoundError(
            "Cannot faithfully resume SAC: matching replay buffer is missing: "
            f"{buffer_path}. Model-only checkpoints are not accepted."
        )
    try:
        model = SAC.load(checkpoint_path, env=env, device=device)
    except ValueError as error:
        raise ValueError(
            "Resume checkpoint is incompatible with the current task's "
            f"observation or action space: {error}"
        ) from error
    if getattr(model.policy, "net_arch", None) != list(SAC_NET_ARCH):
        raise ValueError(
            "Resume checkpoint uses an unsupported SAC architecture; only "
            "net_arch=[256, 256, 256, 256] is accepted."
        )
    model.load_replay_buffer(buffer_path)
    if model.replay_buffer is None:
        raise RuntimeError("Replay buffer loading completed without a replay buffer.")
    return model, buffer_path


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
    config, cli_args = parse_args(argv)
    resume_from = cli_args.resume_from
    low_disk = config.sac.checkpoint_mode == "low_disk"
    if low_disk and resume_from is not None:
        raise ValueError(
            "Low-disk checkpoints are inference-only and cannot faithfully resume; "
            "use sac.checkpoint_mode: full for replay-backed resume."
        )
    task_bank_checks = (
        {config.continual.tasks[0]: task_bank_provenance(
            config.continual.tasks[0], config.runtime.seed,
            config.evaluation.task_set_seed,
            config.evaluation.final_task_set_seeds,
            backend=config.environment.backend,
        )}
        if config.environment.backend in ("metaworld-v3", "kuka-v3")
        and config.evaluation.mode == "fixed-tasks"
        else {}
    )
    run_dir = create_run_directory(config)
    save_resolved_config(config, run_dir)
    create_run_manifest(run_dir, config, task_banks=task_bank_checks)
    directories = ("tensorboard", "evaluations") if low_disk else (
        "tensorboard", "evaluations", "checkpoints", "best_model", "best_success_model",
    )
    for directory in directories:
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
        "evaluation_mode": config.evaluation.mode,
        "task_set_seed": config.evaluation.task_set_seed
        if config.evaluation.mode == "fixed-tasks"
        else None,
        "eval_seeds": list(
            range(
                config.runtime.seed + config.evaluation.seed_offset,
                config.runtime.seed
                + config.evaluation.seed_offset
                + config.evaluation.episodes,
            )
        ),
        "checkpoint_freq": config.sac.checkpoint_freq,
        "checkpoint_mode": config.sac.checkpoint_mode,
        "checkpoint_selection": {
            "primary": "success_rate", "secondary": "mean_return",
            "tie_policy": "keep_previous",
        },
        "best_checkpoint": "best_model.zip" if low_disk else "best_success_model/best_success_model.zip",
        "latest_checkpoint": "latest_model.zip" if low_disk else "final_model.zip",
        "low_disk_resume_supported": False if low_disk else None,
        "run_dir": str(run_dir),
        "algorithm": "SAC",
        "env_backend": config.environment.backend,
        "effective_horizon": horizon_for_backend(config.environment.backend),
        "reward_function_version": config.environment.reward_function_version,
        "policy": "MlpPolicy",
        "resumed": resume_from is not None,
        "resume_checkpoint": str(resume_from.expanduser().resolve())
        if resume_from is not None
        else None,
        **_runtime_provenance(config.environment.backend),
        "ent_coef": SAC_ENT_COEF,
        "net_arch": list(SAC_NET_ARCH),
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
        best_model_save_path=run_dir / "best_model",
        best_success_model_save_path=run_dir if low_disk else run_dir / "best_success_model",
        task_name=config.continual.tasks[0],
        task_set_seed=config.evaluation.task_set_seed,
        evaluation_mode=config.evaluation.mode,
        evaluate_on_training_start=resume_from is None,
        early_stopping=config.sac.early_stopping,
        save_best_return=not low_disk,
        best_success_model_name="best_model" if low_disk else "best_success_model",
        atomic_model_replacement=low_disk,
    )
    checkpoint_callback = (
        LatestModelCallback(config.sac.checkpoint_freq, run_dir / "latest_model.zip")
        if low_disk else
        GlobalCheckpointCallback(
            save_freq=config.sac.checkpoint_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="sac", save_replay_buffer=True,
        )
    )
    callbacks = CallbackList(
        [TrainingSuccessCallback(), evaluation_callback, checkpoint_callback]
    )

    if resume_from is None:
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
            ent_coef=SAC_ENT_COEF,
            policy_kwargs={"net_arch": list(SAC_NET_ARCH)},
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=config.runtime.seed,
            device=config.runtime.device,
            verbose=0,
        )
        start_timesteps = 0
        training_timesteps = config.sac.steps
        reset_num_timesteps = True
        replay_path = None
    else:
        model, replay_path = load_resumable_sac(
            resume_from, train_env, device=config.runtime.device
        )
        start_timesteps = model.num_timesteps
        training_timesteps = remaining_timesteps(config.sac.steps, start_timesteps)
        reset_num_timesteps = False
        # Send new TensorBoard events to this collision-safe child run.
        model.tensorboard_log = str(run_dir / "tensorboard")

    metadata.update(
        {
            "starting_timesteps": start_timesteps,
            "target_timesteps": config.sac.steps,
            "remaining_timesteps": training_timesteps,
            "resume_replay_buffer": str(replay_path) if replay_path else None,
            "early_stopping": early_stopping_metadata(config.sac.early_stopping),
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        model.learn(
            total_timesteps=training_timesteps,
            callback=callbacks,
            tb_log_name="SAC",
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
        metadata["actual_final_timestep"] = model.num_timesteps
        metadata["target_maximum_timestep"] = config.sac.steps
        metadata["early_stopping"] = early_stopping_metadata(
            config.sac.early_stopping, evaluation_callback
        )
        (run_dir / "config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if model.replay_buffer is None:
            raise RuntimeError(
                "SAC training completed without creating a replay buffer."
            )
        print(f"Replay buffer transitions: {model.replay_buffer.size()}")
        if low_disk:
            atomic_replace_model(model, run_dir / "latest_model.zip")
            best_path = run_dir / "best_model.zip"
            selected = SAC.load(best_path, device=config.runtime.device)
            metadata["final_test"] = final_test(config, selected)
            metadata["best_checkpoint_sha256"] = file_hash(best_path)
            metadata["latest_checkpoint_sha256"] = file_hash(run_dir / "latest_model.zip")
            print(f"Best model saved to: {best_path}")
            print(f"Latest model saved to: {run_dir / 'latest_model.zip'}")
        else:
            model.save(run_dir / "final_model")
            print(f"Final model saved to: {run_dir / 'final_model.zip'}")
        if (
            isinstance(evaluation_callback.results, list)
            and evaluation_callback.results
        ):
            mean_returns = np.mean(evaluation_callback.results, axis=1)
            success_rates = np.mean(evaluation_callback.successes, axis=1)
            return_index = best_return_index(mean_returns)
            success_index = best_success_index(
                success_rates,
                mean_returns,
            )

            def qualification(index):
                return qualification_record(
                    success_rate=float(np.mean(evaluation_callback.successes[index])),
                    mean_return=float(np.mean(evaluation_callback.results[index])),
                    return_std=float(evaluation_callback.return_stddevs[index]),
                    episodes=config.evaluation.episodes,
                    task_set_seed=config.evaluation.task_set_seed,
                    evaluation_seed=config.runtime.seed + config.evaluation.seed_offset,
                    deterministic=True,
                    task_hashes=list(evaluation_callback.task_hashes),
                )

            config_path = run_dir / "config.yaml"
            checkpoint_rows = (
                ((run_dir / "best_model.zip", success_index),
                 (run_dir / "latest_model.zip", -1))
                if low_disk else
                ((run_dir / "final_model.zip", -1),
                 (run_dir / "best_model" / "best_model.zip", return_index),
                 (run_dir / "best_success_model" / "best_success_model.zip", success_index))
            )
            for checkpoint, index in checkpoint_rows:
                write_expert_manifest(
                    checkpoint,
                    task_name=config.continual.tasks[0],
                    backend=config.environment.backend,
                    reward_function_version=config.environment.reward_function_version,
                    training_seed=config.runtime.seed,
                    training_config_reference=str(config_path),
                    training_config_sha256=file_hash(config_path),
                    observation_shape=OBSERVATION_SHAPE,
                    action_shape=ACTION_SHAPE,
                    horizon=horizon_for_backend(config.environment.backend),
                    qualification=qualification(index),
                )
        (run_dir / "config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            if not low_disk:
                export_run_summaries(run_dir)
        except Exception as error:
            print(f"WARNING: Could not write SAC result summaries: {error}")
        complete_run(run_dir)
        print(f"Run directory: {run_dir}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
