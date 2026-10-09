"""Train a reproducible Stable-Baselines3 SAC baseline on MetaWorld MT1."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import deque
from datetime import datetime
from importlib import metadata as package_metadata
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
    EarlyStoppingConfig,
    ExperimentConfig,
    output_directory,
    parse_config,
    save_resolved_config,
)
from src.envs import ACTION_SHAPE, OBSERVATION_SHAPE, horizon_for_backend, make_env
from src.continual.evaluation import disable_task_sampling, fixed_mt1_tasks
from src.continual.task_bank import verify_disjoint_task_banks
from src.support.run_manifest import complete_run, create_run_manifest
from src.support.expert_manifest import qualification_record, write_expert_manifest
from src.support.reproducibility import file_hash
from src.hammer_diagnostics import (
    HAMMER_TASK,
    HammerEpisodeTracker,
    summarize_hammer_episodes,
)


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

    if not len(steps) or returns.shape[0] != len(steps) or successes.shape[0] != len(steps):
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
    best_return_index = int(np.argmax(mean_returns))
    best_success_index = max(
        range(len(rows)), key=lambda i: (success_rates[i], mean_returns[i])
    )

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
            best_success_index, "best_success_model/best_success_model.zip"
        ),
        "best_return": checkpoint(best_return_index, "best_model/best_model.zip"),
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
        f"- Best success: step {rows[best_success_index]['step']} "
        f"({rows[best_success_index]['success_rate']:.1%}), "
        "`best_success_model/best_success_model.zip`",
        f"- Best return: step {rows[best_return_index]['step']} "
        f"({rows[best_return_index]['mean_return']:.4f}), `best_model/best_model.zip`",
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
                self.fixed_tasks, identities = fixed_mt1_tasks(
                    task_name, task_set_seed
                )
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
        self.hammer_metrics: list[list[dict]] = []

    def _evaluate(self) -> tuple[list[float], list[int], list[bool]]:
        returns, lengths, successes = [], [], []
        evaluation_behavior = []
        evaluation_hammer = []
        for index, seed in enumerate(self.seeds):
            if self.fixed_tasks is not None:
                self.eval_env.unwrapped.set_task(self.fixed_tasks[index])
            observation, _ = self.eval_env.reset(seed=seed)
            episode_return = 0.0
            episode_length = 0
            episode_success = False
            hammer_tracker = (
                HammerEpisodeTracker(self.eval_env)
                if self.task_name == HAMMER_TASK
                else None
            )
            if hammer_tracker is not None:
                hammer_tracker.sample(0)
            track_push = self.task_name == "kuka-push-v3" and observation.shape == (39,)
            if track_push:
                initial_object = np.asarray(observation[4:7], dtype=float).copy()
                goal = np.asarray(observation[36:39], dtype=float).copy()
                initial_distance = float(np.linalg.norm(initial_object - goal))
                minimum_distance = initial_distance
                maximum_displacement = 0.0
                contact_steps = stationary_contact_steps = 0
                clipped_contact_without_progress_steps = 0
                positive_progress_steps = negative_progress_steps = 0
                progress_contribution = baseline_contribution = 0.0
                pre_contact_caging_contribution = 0.0
                contact_baseline_contribution = 0.0
                first_success_count = 0
                ever_near_object = False
                had_contact = False
                contact_lost = False
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
                if hammer_tracker is not None:
                    hammer_tracker.sample(episode_length)
                if track_push:
                    reward_prefix = (
                        "reward_v3_1"
                        if "reward_v3_1_total" in info
                        else "reward_v3"
                    )
                    obj = np.asarray(observation[4:7], dtype=float)
                    current_distance = float(np.linalg.norm(obj - goal))
                    signed_progress = float(
                        info.get(
                            f"{reward_prefix}_signed_progress",
                            initial_distance - current_distance
                            if episode_length == 1
                            else previous_distance - current_distance,
                        )
                    )
                    raw_env = self.eval_env.unwrapped
                    physical_contact = (
                        raw_env.pad_contacting_object(raw_env._get_id_main_object())
                        and float(observation[3]) > 0
                    )
                    contact = bool(
                        info.get(f"{reward_prefix}_contact", physical_contact)
                    )
                    at_workspace_boundary = bool(
                        np.any(
                            np.isclose(
                                np.asarray(raw_env.data.mocap_pos[0], dtype=float),
                                np.asarray(raw_env.mocap_low, dtype=float),
                                atol=1e-9,
                            )
                            | np.isclose(
                                np.asarray(raw_env.data.mocap_pos[0], dtype=float),
                                np.asarray(raw_env.mocap_high, dtype=float),
                                atol=1e-9,
                            )
                        )
                    )
                    if had_contact and not contact:
                        contact_lost = True
                    had_contact |= contact
                    contact_steps += int(contact)
                    stationary_contact_steps += int(
                        contact and abs(signed_progress) <= 1e-5
                    )
                    clipped_contact_without_progress_steps += int(
                        contact
                        and at_workspace_boundary
                        and abs(signed_progress) <= 1e-5
                    )
                    positive_progress_steps += int(contact and signed_progress > 1e-5)
                    negative_progress_steps += int(contact and signed_progress < -1e-5)
                    progress_contribution += 1000.0 * float(contact) * float(
                        info.get(
                            f"{reward_prefix}_clipped_progress",
                            np.clip(signed_progress, -0.005, 0.005),
                        )
                    )
                    caging = float(
                        info.get(
                            f"{reward_prefix}_caging",
                            info.get("grasp_reward", 0.0),
                        )
                    )
                    if reward_prefix == "reward_v3_1":
                        pre_contact_caging_contribution += caging * float(not contact)
                        contact_baseline_contribution += 0.05 * float(contact)
                        baseline_contribution += (
                            caging * float(not contact) + 0.05 * float(contact)
                        )
                    else:
                        pre_contact_caging_contribution += 0.1 * caging
                        contact_baseline_contribution += 0.05 * float(contact)
                        baseline_contribution += 0.1 * caging + 0.05 * float(contact)
                    first_success_count += int(
                        info.get(f"{reward_prefix}_first_success", False)
                    )
                    ever_near_object |= bool(info.get("near_object", False))
                    minimum_distance = min(minimum_distance, current_distance)
                    maximum_displacement = max(
                        maximum_displacement,
                        float(np.linalg.norm(obj - initial_object)),
                    )
                    previous_distance = current_distance
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(episode_success)
            if hammer_tracker is not None:
                evaluation_hammer.append(
                    hammer_tracker.finish(episode_success).to_dict()
                )
            if track_push:
                final_progress = initial_distance - previous_distance
                maximum_progress = initial_distance - minimum_distance
                if episode_success and previous_distance > 0.05:
                    category = "overshoot"
                elif episode_success:
                    category = "success"
                elif not ever_near_object:
                    category = "no_meaningful_approach"
                elif not had_contact:
                    category = "approach_no_contact"
                elif maximum_displacement <= 0.001:
                    category = "stationary_contact"
                elif final_progress < -0.005:
                    category = "wrong_direction"
                elif minimum_distance <= 0.075:
                    category = "near_threshold_miss"
                elif maximum_progress >= 0.03:
                    category = "useful_progress_insufficient"
                elif contact_lost:
                    category = "contact_loss"
                else:
                    category = "contact_low_progress"
                evaluation_behavior.append(
                    {
                        "seed": seed,
                        "success": episode_success,
                        "category": category,
                        "initial_goal_distance": initial_distance,
                        "maximum_goal_progress": maximum_progress,
                        "minimum_goal_distance": minimum_distance,
                        "final_goal_distance": previous_distance,
                        "maximum_puck_displacement": maximum_displacement,
                        "contact_rate": contact_steps / episode_length,
                        "stationary_contact_rate": stationary_contact_steps
                        / episode_length,
                        "workspace_clipped_contact_without_progress_rate": (
                            clipped_contact_without_progress_steps / episode_length
                        ),
                        "wrong_direction_step_rate": negative_progress_steps
                        / episode_length,
                        "correct_direction_step_rate": positive_progress_steps
                        / episode_length,
                        "signed_progress_contribution": progress_contribution,
                        "caging_contact_baseline_contribution": baseline_contribution,
                        "pre_contact_caging_contribution": (
                            pre_contact_caging_contribution
                        ),
                        "contact_baseline_contribution": contact_baseline_contribution,
                        "success_bonus_occurrences": first_success_count,
                    }
                )
        self._last_behavior_metrics = evaluation_behavior
        self._last_hammer_metrics = evaluation_hammer
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
                        for step, episodes in zip(
                            self.timesteps, self.behavior_metrics
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
        hammer_metrics = getattr(self, "_last_hammer_metrics", None)
        if hammer_metrics:
            self.hammer_metrics.append(hammer_metrics)
        self._save_evaluations()
        self.logger.record("eval/mean_reward", mean_return)
        self.logger.record("eval/std_reward", return_stddev)
        self.logger.record("eval/success_rate", success_rate)
        if hammer_metrics:
            for name, value in summarize_hammer_episodes(hammer_metrics).items():
                self.logger.record(f"eval/{name}", value)

        if mean_return > self.best_mean_return:
            self.best_mean_return = mean_return
            self.model.save(self.best_model_save_path / "best_model")
        score = (success_rate, mean_return)
        if score > self.best_success_score:
            self.best_success_score = score
            self.model.save(
                self.best_success_model_save_path / "best_success_model"
            )
        return (
            self._update_early_stopping(success_rate)
            if allow_early_stop
            else True
        )

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
            and self.consecutive_successful_evaluations
            >= self.early_stopping.patience
        ):
            self.early_stopped = True
            self.stop_timestep = self.num_timesteps
            self.trigger_success_rate = success_rate
            return False
        return True

    def _on_training_start(self) -> None:
        """Initialize the global schedule and optionally evaluate a fresh policy."""
        current = self.model.num_timesteps
        self._next_evaluation_timestep = next_global_boundary(
            current, self.eval_freq
        )
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
            replay_path = self._checkpoint_path(
                "replay_buffer_", extension="pkl"
            )
            self.model.save_replay_buffer(replay_path)
        self._next_checkpoint_timestep = next_global_boundary(
            self.num_timesteps, self.save_freq
        )
        return True


def _git_identity(path: Path) -> dict[str, str | None]:
    """Read local Git provenance without changing repository state."""
    result: dict[str, str | None] = {"branch": None, "commit": None}
    for key, args in (
        ("branch", ("branch", "--show-current")),
        ("commit", ("rev-parse", "HEAD")),
    ):
        try:
            result[key] = subprocess.check_output(
                ("git", "-C", str(path), *args), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return result


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
    match = re.fullmatch(r"(?P<prefix>.+)_(?P<step>\d+)_steps\.zip", checkpoint_path.name)
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
            observed.consecutive_successful_evaluations
            if observed is not None
            else 0
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
        "hammer_nail_progress_weight": (
            config.environment.hammer_nail_progress_weight
        ),
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
            "early_stopping": early_stopping_metadata(
                config.sac.early_stopping
            ),
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
        if isinstance(evaluation_callback.results, list) and evaluation_callback.results:
            best_return_index = int(
                np.argmax(np.mean(evaluation_callback.results, axis=1))
            )
            best_success_index = max(
                range(len(evaluation_callback.results)),
                key=lambda index: (
                    np.mean(evaluation_callback.successes[index]),
                    np.mean(evaluation_callback.results[index]),
                ),
            )

            def qualification(index):
                return qualification_record(
                    success_rate=float(np.mean(evaluation_callback.successes[index])),
                    mean_return=float(np.mean(evaluation_callback.results[index])),
                    return_std=float(evaluation_callback.return_stddevs[index]),
                    episodes=config.evaluation.episodes,
                    task_set_seed=config.evaluation.task_set_seed,
                    evaluation_seed=config.runtime.seed
                    + config.evaluation.seed_offset,
                    deterministic=True,
                    task_hashes=list(evaluation_callback.task_hashes),
                )

            config_path = run_dir / "config.yaml"
            for checkpoint, index in (
                (run_dir / "final_model.zip", -1),
                (run_dir / "best_model" / "best_model.zip", best_return_index),
                (
                    run_dir / "best_success_model" / "best_success_model.zip",
                    best_success_index,
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
