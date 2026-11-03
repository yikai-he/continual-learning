#!/usr/bin/env python3
"""Run independent four-layer LayerNorm SAC benchmarks (no policy transfer)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from src.config import EarlyStoppingConfig
from src.continual.evaluation import evaluate_task
from src.continual.policy_adapter import SB3SACPolicy
from src.continual.task_bank import backend_tasks, bank_hashes
from src.continual.task_sequence import TaskSpec
from src.envs import make_env
from src.sac.transferable_actor import TransferableSACPolicy
from src.sac.architecture_benchmark import (
    LOW_DISK_EXPERIMENTS, environment_task_id, load_benchmark_config,
    run_directory, success_auc,
    stopping_reason, validate_installed_tasks,
)
from src.support.io import atomic_replace_json, atomic_write_json
from src.support.reproducibility import file_hash, git_output
from train_sac import (
    FixedSeedEvalCallback, GlobalCheckpointCallback, TrainingSuccessCallback,
    load_resumable_sac, next_global_boundary, remaining_timesteps,
)


class BenchmarkEvalCallback(FixedSeedEvalCallback):
    """Existing fixed-bank evaluator plus benchmark-specific logger fields."""

    def _record_evaluation(self, *, allow_early_stop=True):
        result = super()._record_evaluation(allow_early_stop=allow_early_stop)
        self.logger.record(
            "eval/consecutive_success_count",
            self.consecutive_successful_evaluations,
        )
        self.logger.record("eval/best_success_rate", self.best_success_score[0])
        self.logger.record("time/actual_training_timesteps", self.num_timesteps)
        self._write_evaluation_csv()
        return result

    def _write_evaluation_csv(self):
        """Atomically retain the complete evaluation history in tabular form."""
        path = Path(self.log_path) / "evaluation_history.csv"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        best_success = (-np.inf, -np.inf)
        best_return = -np.inf
        try:
            with temporary.open("w", encoding="utf-8", newline="") as output:
                fields = [
                    "timestep", "success_rate", "mean_return", "return_std",
                    "best_success_rate", "best_mean_return",
                ]
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for step, returns, successes, return_std in zip(
                    self.timesteps,
                    self.results,
                    self.successes,
                    self.return_stddevs,
                ):
                    success_rate = float(np.mean(successes))
                    mean_return = float(np.mean(returns))
                    best_success = max(best_success, (success_rate, mean_return))
                    best_return = max(best_return, mean_return)
                    writer.writerow(
                        {
                            "timestep": step,
                            "success_rate": success_rate,
                            "mean_return": mean_return,
                            "return_std": return_std,
                            "best_success_rate": best_success[0],
                            "best_mean_return": best_return,
                        }
                    )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _atomic_replace_model(model, path: Path) -> None:
    """Replace one SB3 zip without exposing a partially written destination."""
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.zip")
    try:
        model.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class LatestModelCallback(BaseCallback):
    """Atomically replace a single inference checkpoint on global boundaries."""

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
        _atomic_replace_model(self.model, self.path)
        self._next_checkpoint_timestep = next_global_boundary(
            self.num_timesteps, self.save_freq
        )
        return True


def _write_metadata(path, value, *, initial):
    if initial:
        atomic_write_json(path, value)
    else:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Benchmark metadata is missing: {path}")
        atomic_replace_json(path, value, allow_nan=False)


def _bank_provenance(
    task, training_seed, validation_seed, final_seeds, *, backend="metaworld-v3"
):
    seeds = [training_seed, validation_seed, *final_seeds]
    banks = {
        str(seed): bank_hashes(
            backend_tasks(task, seed)
            if backend == "metaworld-v3"
            else backend_tasks(task, seed, backend=backend)
        )
        for seed in seeds
    }
    for i, left in enumerate(seeds):
        for right in seeds[i + 1:]:
            if set(banks[str(left)]) & set(banks[str(right)]):
                raise ValueError(f"Task banks {left} and {right} overlap.")
    return {
        "training_task_set_seed": training_seed,
        "validation_task_set_seed": validation_seed,
        "final_test_task_set_seeds": list(final_seeds),
        "ordered_task_hashes": banks,
        "all_banks_pairwise_disjoint": True,
    }


def _hydrate(callback, run_dir, maximum_step, settings):
    path = run_dir / "evaluations" / "evaluations.npz"
    if not path.is_file():
        raise FileNotFoundError("Resume run has no evaluation history.")
    with np.load(path) as archive:
        mask = archive["timesteps"] <= maximum_step
        callback.timesteps = archive["timesteps"][mask].astype(int).tolist()
        callback.results = archive["results"][mask].astype(float).tolist()
        callback.ep_lengths = archive["ep_lengths"][mask].astype(int).tolist()
        callback.successes = archive["successes"][mask].astype(bool).tolist()
        callback.return_stddevs = archive["return_stddevs"][mask].astype(float).tolist()
    if not callback.timesteps:
        raise ValueError("No evaluation history precedes the resume checkpoint.")
    for attribute, filename in (
        ("behavior_metrics", "behavior_metrics.json"),
        ("success_diagnostics", "success_diagnostics.json"),
        ("hammer_metrics", "hammer_metrics.json"),
        ("handoff_metrics", "handoff_metrics.json"),
    ):
        detail = run_dir / "evaluations" / filename
        if detail.is_file():
            rows = json.loads(detail.read_text(encoding="utf-8"))
            setattr(
                callback,
                attribute,
                [row["episodes"] for row in rows if row["step"] <= maximum_step],
            )
    rates = np.mean(callback.successes, axis=1).tolist()
    returns = np.mean(callback.results, axis=1).tolist()
    callback.best_mean_return = max(returns)
    callback.best_success_score = max(zip(rates, returns))
    count = 0
    for rate in reversed(rates):
        if rate < settings.threshold:
            break
        count += 1
    callback.consecutive_successful_evaluations = count
    if (
        getattr(settings, "enabled", True)
        and
        callback.timesteps[-1] >= settings.min_timesteps
        and count >= settings.consecutive_evaluations
    ):
        callback.early_stopped = True
        callback.stop_timestep = callback.timesteps[-1]
        callback.trigger_success_rate = rates[-1]
    callback._save_evaluations()


def _final_test(config, task, model):
    backend = config.environment.env_backend
    env_task = environment_task_id(config, task)
    policy = SB3SACPolicy(model)
    results = []
    for index, bank_seed in enumerate(config.evaluation.final_task_set_seeds):
        results.append(
            evaluate_task(
                policy,
                TaskSpec(task_id=env_task, task_name=env_task),
                episodes=50,
                seed=config.evaluation.final_reset_seed + 50 * index,
                evaluation_mode="fixed-tasks",
                task_set_seed=bank_seed,
                reward_function_version=config.reward_function_version,
                hammer_reward_variant=config.environment.hammer_reward_variant,
                hammer_nail_progress_weight=(
                    config.environment.hammer_nail_progress_weight
                ),
                peg_unplug_reward_variant=(
                    config.environment.peg_unplug_reward_variant
                ),
                peg_unplug_progress_weight=(
                    config.environment.peg_unplug_progress_weight
                ),
                peg_unplug_progress_gamma=(
                    config.environment.peg_unplug_progress_gamma
                ),
                backend=backend,
                progress=True,
            )
        )
    returns = results[0]["episode_returns"] + results[1]["episode_returns"]
    successes = results[0]["episode_successes"] + results[1]["episode_successes"]
    return {
        "episodes": 100,
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "task_set_seeds": list(config.evaluation.final_task_set_seeds),
        "reset_seeds": results[0]["seeds"] + results[1]["seeds"],
        "ordered_task_hashes": (
            results[0]["ordered_task_hashes"] + results[1]["ordered_task_hashes"]
        ),
        "banks": results,
    }


def _validate_backend_identity(env, config, env_task):
    """Fail before training if a KUKA profile resolved to a Sawyer class."""
    base = env.unwrapped
    if config.environment.env_backend == "kuka-v3":
        from metaworld_kuka.registration import KUKA_ENVIRONMENTS

        expected = KUKA_ENVIRONMENTS[env_task]
        if not isinstance(base, expected) or not type(base).__module__.startswith(
            "metaworld_kuka."
        ):
            raise RuntimeError("KUKA profile resolved to a non-KUKA environment.")
    return type(base).__module__, type(base).__name__


def _git_provenance():
    import metaworld

    continual_root = Path(__file__).resolve().parents[1]
    kuka_root = Path(metaworld.__file__).resolve().parent.parent
    return {
        "continual_learning": git_output(continual_root, "rev-parse", "HEAD"),
        "metaworld_kuka_v3": git_output(kuka_root, "rev-parse", "HEAD"),
    }


def run_one(config, task, seed, *, resume=False):
    low_disk = config.experiment in LOW_DISK_EXPERIMENTS
    if low_disk and resume:
        raise ValueError(
            "Low-disk experiments cannot resume training because "
            "replay buffers are intentionally not persisted. Restart from scratch."
        )
    run_dir = run_directory(config, task, seed)
    backend = config.environment.env_backend
    env_task = environment_task_id(config, task)
    if resume:
        if not (run_dir / "RUNNING").is_file() or (run_dir / "RUN_COMPLETE").exists():
            raise ValueError("Resume requires one incomplete RUNNING benchmark.")
        saved = load_benchmark_config(run_dir / "config.yaml")
        if asdict(saved) != asdict(config):
            raise ValueError("Resume configuration differs from the saved benchmark.")
        checkpoints = sorted(
            (run_dir / "checkpoints").glob("sac_*_steps.zip"),
            key=lambda path: int(path.stem.split("_")[-2]),
        )
        if not checkpoints:
            raise FileNotFoundError("Resume run has no paired SAC checkpoint.")
        checkpoint = checkpoints[-1]
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if (
            metadata.get("kind") != f"{config.experiment}_run_v1"
            or metadata.get("task") != task
            or metadata.get("seed") != seed
            or metadata.get("config") != asdict(config)
        ):
            raise ValueError("Resume metadata differs from the requested run.")
    else:
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark run: {run_dir}")
        run_dir.mkdir(parents=True)
        import yaml
        with (run_dir / "config.yaml").open("x", encoding="utf-8") as output:
            yaml.safe_dump(asdict(config), output, sort_keys=False)
        (run_dir / "RUNNING").write_text("running\n", encoding="utf-8")
        directories = ("tensorboard", "evaluations") if low_disk else (
            "tensorboard", "evaluations", "checkpoints", "best_model",
            "best_success_model",
        )
        for name in directories:
            (run_dir / name).mkdir()
        checkpoint = None
        metadata = {
            "kind": f"{config.experiment}_run_v1", "task": task, "seed": seed,
            "config": asdict(config),
            "task_banks": _bank_provenance(
                env_task, seed, config.evaluation.validation_task_set_seed,
                config.evaluation.final_task_set_seeds, backend=backend,
            ),
            "initialization": "random; no GeneralPolicy or expert transfer",
            "training_status": "running",
            "architecture": {
                "actor_hidden_sizes": list(config.policy.net_arch),
                "critic_hidden_sizes": list(config.policy.net_arch),
                "actor_layer_norm_after_hidden": [0],
            },
            "git_commits": _git_provenance(),
            "environment": {
                "backend": backend,
                "task_label": task,
                "registered_task_id": env_task,
                "robot": "KUKA iiwa" if backend == "kuka-v3" else "Sawyer",
                "gripper": (
                    "Robotiq 2F-85" if backend == "kuka-v3" else "Sawyer gripper"
                ),
                "reward_function_version": config.reward_function_version,
                "hammer_reward_variant": config.environment.hammer_reward_variant,
                "hammer_nail_progress_weight": (
                    config.environment.hammer_nail_progress_weight
                ),
                "peg_unplug_reward_variant": (
                    config.environment.peg_unplug_reward_variant
                ),
                "peg_unplug_progress_weight": (
                    config.environment.peg_unplug_progress_weight
                ),
                "peg_unplug_progress_gamma": (
                    config.environment.peg_unplug_progress_gamma
                ),
            },
            "training_duration_seconds": 0.0,
            "checkpoint_strategy": (
                "low-disk: atomically replaced best_model.zip and latest_model.zip; "
                "no replay persistence or exact resume"
                if low_disk
                else "periodic paired model and replay-buffer checkpoints"
            ),
        }
        _write_metadata(run_dir / "metadata.json", metadata, initial=True)

    torch.set_num_threads(config.training.torch_num_threads)
    env_options = dict(
        reward_function_version=config.reward_function_version,
        hammer_reward_variant=config.environment.hammer_reward_variant,
        hammer_nail_progress_weight=config.environment.hammer_nail_progress_weight,
        peg_unplug_reward_variant=config.environment.peg_unplug_reward_variant,
        peg_unplug_progress_weight=config.environment.peg_unplug_progress_weight,
        peg_unplug_progress_gamma=config.environment.peg_unplug_progress_gamma,
    )
    train_env = Monitor(make_env(backend, env_task, seed, **env_options))
    eval_env = Monitor(make_env(
        backend, env_task, config.evaluation.validation_task_set_seed, **env_options,
    ))
    try:
        module_name, class_name = _validate_backend_identity(train_env, config, env_task)
        _validate_backend_identity(eval_env, config, env_task)
        metadata["environment"].update(
            environment_class=f"{module_name}.{class_name}"
        )
        _write_metadata(run_dir / "metadata.json", metadata, initial=False)
        stop = config.early_stopping
        callback = BenchmarkEvalCallback(
            eval_env, eval_freq=config.evaluation.frequency,
            n_eval_episodes=config.evaluation.episodes,
            seed=config.evaluation.validation_reset_seed,
            log_path=run_dir / "evaluations",
            best_model_save_path=run_dir / "best_model",
            best_success_model_save_path=(
                run_dir if low_disk else run_dir / "best_success_model"
            ),
            task_name=env_task,
            task_set_seed=config.evaluation.validation_task_set_seed,
            evaluation_mode="fixed-tasks", evaluate_on_training_start=not resume,
            early_stopping=EarlyStoppingConfig(
                enabled=stop.enabled, min_steps=stop.min_timesteps,
                success_threshold=stop.threshold,
                patience=stop.consecutive_evaluations,
            ),
            save_best_return=not low_disk,
            best_success_model_name=("best_model" if low_disk else "best_success_model"),
            atomic_model_replacement=low_disk,
        )
        checkpoint_callback = (
            LatestModelCallback(config.checkpoint_frequency, run_dir / "latest_model.zip")
            if low_disk
            else GlobalCheckpointCallback(
                save_freq=config.checkpoint_frequency,
                save_path=str(run_dir / "checkpoints"), name_prefix="sac",
                save_replay_buffer=True,
            )
        )
        callbacks = CallbackList(
            [TrainingSuccessCallback(), callback, checkpoint_callback]
        )
        if resume:
            model, _ = load_resumable_sac(checkpoint, train_env, device=config.training.device)
            _hydrate(callback, run_dir, model.num_timesteps, stop)
            if model.num_timesteps > config.training.max_timesteps:
                raise ValueError("Resume checkpoint exceeds the configured budget.")
            remaining = (
                0
                if callback.early_stopped
                or model.num_timesteps == config.training.max_timesteps
                else
                remaining_timesteps(config.training.max_timesteps, model.num_timesteps)
            )
            reset = False
        else:
            t = config.training
            model = SAC(
                TransferableSACPolicy, train_env, learning_rate=t.learning_rate,
                buffer_size=t.buffer_size, learning_starts=t.learning_starts,
                batch_size=t.batch_size, gamma=t.gamma, tau=t.tau,
                train_freq=t.train_freq, gradient_steps=t.gradient_steps,
                ent_coef=t.ent_coef,
                policy_kwargs={
                    "net_arch": config.policy.net_arch,
                    "log_std_init": config.policy.log_std_init,
                },
                tensorboard_log=str(run_dir / "tensorboard"), seed=seed,
                device=t.device, verbose=0,
            )
            remaining, reset = t.max_timesteps, True
        started = time.monotonic()
        if remaining:
            model.learn(
                total_timesteps=remaining, callback=callbacks,
                tb_log_name=config.experiment, progress_bar=True,
                reset_num_timesteps=reset,
            )
        duration = float(metadata.get("training_duration_seconds", 0.0)) + (
            time.monotonic() - started
        )
        if not stop.enabled and model.num_timesteps != config.training.max_timesteps:
            raise RuntimeError(
                "Full-budget training ended without exactly reaching max_timesteps."
            )
        if low_disk:
            _atomic_replace_model(model, run_dir / "latest_model.zip")
        else:
            model.save(run_dir / "final_model")
            model.save_replay_buffer(run_dir / "final_replay_buffer.pkl")
        best_path = (
            run_dir / "best_model.zip"
            if low_disk
            else run_dir / "best_success_model" / "best_success_model.zip"
        )
        best = SAC.load(best_path, device=config.training.device)
        final_test = _final_test(config, task, best)
        with np.load(run_dir / "evaluations" / "evaluations.npz") as archive:
            steps = archive["timesteps"].astype(int).tolist()
            rates = np.mean(archive["successes"], axis=1).tolist()
            returns = np.mean(archive["results"], axis=1).tolist()
            return_stddevs = archive["return_stddevs"].astype(float).tolist()
        result = {
            "kind": f"{config.experiment}_result_v1", "task": task, "seed": seed,
            "actual_training_steps": model.num_timesteps,
            "stopping_reason": stopping_reason(
                callback.early_stopped,
                model.num_timesteps,
                config.training.max_timesteps,
            ),
            "best_validation_success": max(rates),
            "best_validation_mean_return_at_best_success": max(zip(rates, returns))[1],
            "best_validation_mean_return": max(returns),
            "best_success_checkpoint": str(best_path),
            "best_success_checkpoint_sha256": file_hash(best_path),
            "latest_model": str(
                run_dir / ("latest_model.zip" if low_disk else "final_model.zip")
            ),
            "latest_model_sha256": file_hash(
                run_dir / ("latest_model.zip" if low_disk else "final_model.zip")
            ),
            "training_duration_seconds": duration,
            "evaluation_steps": steps, "validation_success_rates": rates,
            "validation_mean_returns": returns,
            "validation_return_standard_deviations": return_stddevs,
            "success_auc": success_auc(steps, rates), "final_test": final_test,
        }
        if not low_disk:
            result.update(
                best_return_checkpoint=str(
                    run_dir / "best_model" / "best_model.zip"
                ),
                best_return_checkpoint_sha256=file_hash(
                    run_dir / "best_model" / "best_model.zip"
                ),
            )
        atomic_write_json(run_dir / "result.json", result)
        atomic_write_json(run_dir / "summary.json", result)
        metadata["result"] = result
        metadata["training_duration_seconds"] = duration
        metadata["training_status"] = "complete"
        _write_metadata(run_dir / "metadata.json", metadata, initial=False)
        (run_dir / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
        (run_dir / "RUNNING").unlink()
    finally:
        train_env.close(); eval_env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sac_4layer_benchmark.yaml"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=None)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_benchmark_config(args.config)
    validate_installed_tasks(
        config.tasks, backend=config.environment.env_backend
    )
    tasks = config.tasks if args.all else [args.task]
    seeds = args.seeds or ([args.seed] if args.seed is not None else config.seeds)
    if args.resume and (len(tasks) != 1 or len(seeds) != 1):
        parser.error("--resume requires exactly one task and one seed.")
    for task in tasks:
        for seed in seeds:
            run_one(config, task, seed, resume=args.resume)


if __name__ == "__main__":
    main()
