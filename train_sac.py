"""Train a reproducible Stable-Baselines3 SAC baseline on MetaWorld MT1."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import deque
from datetime import datetime
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
from src.config import (
    EarlyStoppingConfig,
    ExperimentConfig,
    output_directory,
    parse_config,
    save_resolved_config,
)
from src.continual.evaluation import disable_task_sampling, fixed_mt1_tasks
from src.continual.task_bank import verify_disjoint_task_banks
from src.envs import ACTION_SHAPE, OBSERVATION_SHAPE, horizon_for_backend, make_env
from src.support.reproducibility import git_output
from src.hammer_diagnostics import (
    HAMMER_TASK,
    HammerEpisodeTracker,
    summarize_hammer_episodes,
)
from src.sac.checkpoint_ranking import (
    best_return_index,
    best_success_index,
    success_score,
)
from src.sac.kuka_diagnostics import KukaEpisodeDiagnostics
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


def summarize_success_trace(successes, *, sustained_steps: int = 10) -> dict:
    """Summarize observational success retention without changing success semantics."""
    trace = [bool(value) for value in successes]
    if not trace:
        raise ValueError("Success trace must contain at least one environment step.")
    if sustained_steps <= 0:
        raise ValueError("sustained_steps must be positive.")
    longest_run = run = 0
    for success in trace:
        run = run + 1 if success else 0
        longest_run = max(longest_run, run)
    first_success = next((i for i, success in enumerate(trace) if success), None)
    retained_steps = sum(trace[first_success:]) if first_success is not None else 0
    post_success_steps = len(trace) - first_success if first_success is not None else 0
    return {
        "any_step_success": any(trace),
        "final_step_success": trace[-1],
        "sustained_success": longest_run >= sustained_steps,
        "longest_success_run": longest_run,
        "first_success_step": first_success,
        "success_retention_steps": retained_steps,
        "success_retention_fraction": (
            retained_steps / post_success_steps if post_success_steps else 0.0
        ),
    }


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
        self.behavior_metrics: list[list[dict]] = []
        self.success_diagnostics: list[list[dict]] = []
        self.hammer_metrics: list[list[dict]] = []
        self.handoff_metrics: list[list[dict]] = []

    def _evaluate(self) -> tuple[list[float], list[int], list[bool]]:
        returns, lengths, successes = [], [], []
        evaluation_behavior = []
        evaluation_success_diagnostics = []
        evaluation_hammer = []
        evaluation_handoff = []
        for index, seed in enumerate(self.seeds):
            if self.fixed_tasks is not None:
                self.eval_env.unwrapped.set_task(self.fixed_tasks[index])
            observation, _ = self.eval_env.reset(seed=seed)
            diagnostics = KukaEpisodeDiagnostics(
                self.task_name,
                self.eval_env,
                observation,
                HammerEpisodeTracker,
            )
            episode_return = 0.0
            episode_length = 0
            episode_success = False
            episode_success_trace = []
            while True:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.eval_env.step(
                    action
                )
                episode_return += float(reward)
                episode_length += 1
                step_success = bool(info.get("success", info.get("is_success", False)))
                episode_success |= step_success
                episode_success_trace.append(step_success)
                diagnostics.observe_step(
                    action, observation, reward, info, episode_length
                )
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(episode_success)
            success_diagnostic = summarize_success_trace(episode_success_trace)
            evaluation_success_diagnostics.append({"seed": seed, **success_diagnostic})
            behavior, hammer, handoff = diagnostics.finish(
                seed=seed,
                episode_return=episode_return,
                episode_length=episode_length,
                episode_success=episode_success,
                success_diagnostic=success_diagnostic,
                final_info=info,
            )
            if behavior is not None:
                evaluation_behavior.append(behavior)
            if hammer is not None:
                evaluation_hammer.append(hammer)
            if handoff is not None:
                evaluation_handoff.append(handoff)
        self._last_behavior_metrics = evaluation_behavior
        self._last_success_diagnostics = evaluation_success_diagnostics
        self._last_hammer_metrics = evaluation_hammer
        self._last_handoff_metrics = evaluation_handoff
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
        if self.behavior_metrics:
            (self.log_path / "behavior_metrics.json").write_text(
                json.dumps(
                    [
                        {"step": step, "episodes": episodes}
                        for step, episodes in zip(self.timesteps, self.behavior_metrics)
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if self.success_diagnostics:
            (self.log_path / "success_diagnostics.json").write_text(
                json.dumps(
                    [
                        {"step": step, "episodes": episodes}
                        for step, episodes in zip(
                            self.timesteps, self.success_diagnostics
                        )
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if self.hammer_metrics:
            (self.log_path / "hammer_metrics.json").write_text(
                json.dumps(
                    [
                        {"step": step, "episodes": episodes}
                        for step, episodes in zip(self.timesteps, self.hammer_metrics)
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if self.handoff_metrics:
            (self.log_path / "handoff_metrics.json").write_text(
                json.dumps(
                    [
                        {"step": step, "episodes": episodes}
                        for step, episodes in zip(self.timesteps, self.handoff_metrics)
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
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
        if getattr(self, "_last_behavior_metrics", None):
            self.behavior_metrics.append(self._last_behavior_metrics)
        if getattr(self, "_last_success_diagnostics", None):
            self.success_diagnostics.append(self._last_success_diagnostics)
        hammer_metrics = getattr(self, "_last_hammer_metrics", None)
        if hammer_metrics:
            self.hammer_metrics.append(hammer_metrics)
        handoff_metrics = getattr(self, "_last_handoff_metrics", None)
        if handoff_metrics:
            self.handoff_metrics.append(handoff_metrics)
        self._save_evaluations()
        self.logger.record("eval/mean_reward", mean_return)
        self.logger.record("eval/std_reward", return_stddev)
        self.logger.record("eval/success_rate", success_rate)
        if hammer_metrics:
            for name, value in summarize_hammer_episodes(hammer_metrics).items():
                self.logger.record(f"eval/{name}", value)
        if handoff_metrics:
            self.logger.record(
                "eval/handoff_instantaneous_success_rate",
                float(
                    np.mean([item["instantaneous_success"] for item in handoff_metrics])
                ),
            )
            self.logger.record("eval/handoff_sustained_success_rate", success_rate)
            for metric in (
                "valid_rear_contact_episode",
                "first_valid_rear_contact_step",
                "initial_tcp_target_distance",
                "minimum_tcp_target_distance",
                "final_tcp_target_distance",
                "mean_physical_aperture",
                "final_physical_aperture",
                "mean_aperture_error",
                "maximum_cube_displacement",
                "maximum_cube_tilt",
                "reward_tcp_shaping_total",
                "reward_aperture_shaping_total",
                "reward_handoff_total",
                "final_tcp_potential",
                "final_aperture_potential",
                "final_total_potential",
                "discounted_episode_return_gamma_0_99",
            ):
                values = [
                    float(item[metric])
                    for item in handoff_metrics
                    if item.get(metric) is not None
                ]
                if values:
                    self.logger.record(f"eval/handoff_{metric}", float(np.mean(values)))

        if mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            self.model.save(self.best_model_save_path / "best_model")
        score = success_score(success_rate, mean_return)
        if score > self.best_success_score:
            self.best_success_score = score
            self.model.save(self.best_success_model_save_path / "best_success_model")
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
        "kuka-v2": None,
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
    task_bank_checks = (
        {
            config.continual.tasks[0]: verify_disjoint_task_banks(
                config.continual.tasks[0],
                config.runtime.seed,
                config.evaluation.task_set_seed,
            )
        }
        if config.environment.backend == "metaworld-v3"
        and config.evaluation.mode == "fixed-tasks"
        else {}
    )
    run_dir = create_run_directory(config)
    save_resolved_config(config, run_dir)
    create_run_manifest(run_dir, config, task_banks=task_bank_checks)
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
        "run_dir": str(run_dir),
        "algorithm": "SAC",
        "env_backend": config.environment.backend,
        "effective_horizon": horizon_for_backend(config.environment.backend),
        "reward_function_version": config.environment.reward_function_version,
        "hammer_reward_variant": config.environment.hammer_reward_variant,
        "hammer_nail_progress_weight": (config.environment.hammer_nail_progress_weight),
        "peg_unplug_reward_variant": config.environment.peg_unplug_reward_variant,
        "peg_unplug_progress_weight": config.environment.peg_unplug_progress_weight,
        "peg_unplug_progress_gamma": config.environment.peg_unplug_progress_gamma,
        "cube_push_curriculum_level": config.environment.cube_push_curriculum_level,
        "cube_push_initialization": config.environment.cube_push_initialization,
        "cube_y_min": config.environment.cube_y_min,
        "near_contact_gap_m": config.environment.near_contact_gap_m,
        "policy": "MlpPolicy",
        "resumed": resume_from is not None,
        "resume_checkpoint": str(resume_from.expanduser().resolve())
        if resume_from is not None
        else None,
        **_runtime_provenance(config.environment.backend),
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
            hammer_reward_variant=config.environment.hammer_reward_variant,
            hammer_nail_progress_weight=(
                config.environment.hammer_nail_progress_weight
            ),
            peg_unplug_reward_variant=config.environment.peg_unplug_reward_variant,
            peg_unplug_progress_weight=config.environment.peg_unplug_progress_weight,
            peg_unplug_progress_gamma=config.environment.peg_unplug_progress_gamma,
            cube_push_curriculum_level=config.environment.cube_push_curriculum_level,
            cube_push_initialization=config.environment.cube_push_initialization,
            cube_y_min=config.environment.cube_y_min,
            near_contact_gap_m=config.environment.near_contact_gap_m,
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
            hammer_reward_variant=config.environment.hammer_reward_variant,
            hammer_nail_progress_weight=(
                config.environment.hammer_nail_progress_weight
            ),
            peg_unplug_reward_variant=config.environment.peg_unplug_reward_variant,
            peg_unplug_progress_weight=config.environment.peg_unplug_progress_weight,
            peg_unplug_progress_gamma=config.environment.peg_unplug_progress_gamma,
            cube_push_curriculum_level=config.environment.cube_push_curriculum_level,
            cube_push_initialization=config.environment.cube_push_initialization,
            cube_y_min=config.environment.cube_y_min,
            near_contact_gap_m=config.environment.near_contact_gap_m,
        )
    )
    evaluation_callback = FixedSeedEvalCallback(
        eval_env,
        eval_freq=config.evaluation.frequency,
        n_eval_episodes=config.evaluation.episodes,
        seed=config.runtime.seed + config.evaluation.seed_offset,
        log_path=run_dir / "evaluations",
        best_model_save_path=run_dir / "best_model",
        best_success_model_save_path=run_dir / "best_success_model",
        task_name=config.continual.tasks[0],
        task_set_seed=config.evaluation.task_set_seed,
        evaluation_mode=config.evaluation.mode,
        evaluate_on_training_start=resume_from is None,
        early_stopping=config.sac.early_stopping,
    )
    callbacks = CallbackList(
        [
            TrainingSuccessCallback(),
            evaluation_callback,
            GlobalCheckpointCallback(
                save_freq=config.sac.checkpoint_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="sac",
                save_replay_buffer=True,
            ),
        ]
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
            ent_coef=config.sac.ent_coef,
            policy_kwargs={"net_arch": config.sac.net_arch},
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
            for checkpoint, index in (
                (run_dir / "final_model.zip", -1),
                (run_dir / "best_model" / "best_model.zip", return_index),
                (
                    run_dir / "best_success_model" / "best_success_model.zip",
                    success_index,
                ),
            ):
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
                    hammer_reward_variant=(
                        config.environment.hammer_reward_variant
                        if config.continual.tasks[0] == HAMMER_TASK
                        else None
                    ),
                    hammer_nail_progress_weight=(
                        config.environment.hammer_nail_progress_weight
                        if config.continual.tasks[0] == HAMMER_TASK
                        else None
                    ),
                    qualification=qualification(index),
                )
        try:
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
