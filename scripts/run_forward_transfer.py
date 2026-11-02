#!/usr/bin/env python3
"""Run one isolated Random-vs-Transferred Immediate SAC condition."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from src.continual.evaluation import disable_task_sampling
from src.continual.task_bank import verify_disjoint_task_banks
from src.envs import make_env
from src.sac.forward_transfer import (
    evaluation_step_grid,
    learning_curve_metrics,
    load_forward_transfer_config,
    load_temporally_valid_general_policy,
    resolved_run_directory,
    save_resolved_forward_transfer_config,
)
from src.sac.policy_transfer import transfer_general_policy_to_sac
from src.sac.transferable_actor import TransferableSACPolicy
from src.support.io import atomic_replace_json, atomic_write_json
from src.support.reproducibility import file_hash, state_hash
from train_sac import (
    FixedSeedEvalCallback,
    GlobalCheckpointCallback,
    TrainingSuccessCallback,
    load_resumable_sac,
    remaining_timesteps,
)


def _hydrate_evaluations(callback: FixedSeedEvalCallback, run_dir: Path) -> None:
    """Restore evaluation history so resume preserves the full learning curve."""
    path = run_dir / "evaluations" / "evaluations.npz"
    if not path.is_file():
        raise FileNotFoundError("Resume run has no evaluation history.")
    with np.load(path) as archive:
        callback.timesteps = archive["timesteps"].astype(int).tolist()
        callback.results = archive["results"].astype(float).tolist()
        callback.ep_lengths = archive["ep_lengths"].astype(int).tolist()
        callback.successes = archive["successes"].astype(bool).tolist()
        callback.return_stddevs = archive["return_stddevs"].astype(float).tolist()
    for attribute, filename in (
        ("behavior_metrics", "behavior_metrics.json"),
        ("success_diagnostics", "success_diagnostics.json"),
        ("hammer_metrics", "hammer_metrics.json"),
        ("handoff_metrics", "handoff_metrics.json"),
    ):
        detail = run_dir / "evaluations" / filename
        if detail.is_file():
            rows = json.loads(detail.read_text(encoding="utf-8"))
            setattr(callback, attribute, [row["episodes"] for row in rows])
    success_rates = np.mean(callback.successes, axis=1)
    mean_returns = np.mean(callback.results, axis=1)
    callback.best_mean_return = float(np.max(mean_returns))
    callback.best_success_score = max(zip(success_rates, mean_returns))


def _summary(config, run_dir: Path, source: dict[str, object]) -> dict[str, object]:
    with np.load(run_dir / "evaluations" / "evaluations.npz") as archive:
        steps = archive["timesteps"].astype(int).tolist()
        success_rates = np.mean(archive["successes"], axis=1).tolist()
        mean_returns = np.mean(archive["results"], axis=1).tolist()
        task_hashes = archive["ordered_task_data_sha256"].tolist()
    expected = evaluation_step_grid(config.sac.steps, config.evaluation.frequency)
    if steps != expected:
        raise ValueError(f"Incomplete evaluation grid: expected {expected}, got {steps}")
    return {
        "kind": "forward_transfer_run_summary_v1",
        "condition": config.condition,
        "target_task": config.target_task,
        "training_seed": config.runtime.seed,
        "source_general_policy": source,
        "evaluation_steps": steps,
        "success_rates": success_rates,
        "mean_returns": mean_returns,
        "ordered_evaluation_task_hashes": task_hashes,
        "metrics": learning_curve_metrics(
            steps,
            success_rates,
            mean_returns,
            success_threshold=config.evaluation.success_threshold,
        ),
    }


def _write_run_metadata(
    path: Path, metadata: dict[str, object], *, initial: bool
) -> None:
    """Create metadata once, then atomically replace only that owned artifact."""
    path = Path(path)
    if initial:
        atomic_write_json(path, metadata)
        return
    if not path.is_file():
        raise FileNotFoundError(f"Run metadata is missing: {path}")
    atomic_replace_json(path, metadata, allow_nan=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the derived incomplete run from its latest paired checkpoint.",
    )
    args = parser.parse_args(argv)
    config = load_forward_transfer_config(args.config, seed=args.seed)
    run_dir = resolved_run_directory(config)
    policy, source_identity = load_temporally_valid_general_policy(config)

    if args.resume:
        if not run_dir.is_dir() or not (run_dir / "RUNNING").is_file():
            raise ValueError("Resume requires an existing incomplete RUNNING directory.")
        if (run_dir / "RUN_COMPLETE").exists():
            raise ValueError("Completed forward-transfer runs cannot be resumed.")
        saved = load_forward_transfer_config(run_dir / "config.yaml")
        if asdict(saved) != asdict(config):
            raise ValueError("Resume configuration differs from the saved run.")
        checkpoints = sorted(
            (run_dir / "checkpoints").glob("sac_*_steps.zip"),
            key=lambda path: int(path.stem.split("_")[-2]),
        )
        if not checkpoints:
            raise FileNotFoundError("Resume run has no SAC checkpoint.")
        resume_checkpoint = checkpoints[-1]
    else:
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
        run_dir.mkdir(parents=True)
        save_resolved_forward_transfer_config(config, run_dir / "config.yaml")
        (run_dir / "RUNNING").write_text("running\n", encoding="utf-8")
        for name in (
            "tensorboard", "evaluations", "checkpoints", "best_model",
            "best_success_model",
        ):
            (run_dir / name).mkdir()
        resume_checkpoint = None

    task_bank = verify_disjoint_task_banks(
        config.target_task,
        config.runtime.seed,
        config.evaluation.task_set_seed,
    )
    if args.resume:
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            metadata.get("kind") != "forward_transfer_run_v1"
            or metadata.get("config") != asdict(config)
            or metadata.get("source_general_policy") != source_identity
        ):
            raise ValueError("Resume metadata differs from the requested run.")
    else:
        metadata = {
            "kind": "forward_transfer_run_v1",
            "config": asdict(config),
            "source_general_policy": source_identity,
            "task_bank_check": task_bank,
            "initialization": config.condition,
            "actor_architecture": "TransferableSACPolicy LayerNorm [256, 256]",
        }
        _write_run_metadata(run_dir / "metadata.json", metadata, initial=True)

    torch.set_num_threads(1)
    train_env = Monitor(
        make_env(
            config.environment_backend,
            config.target_task,
            config.runtime.seed,
            reward_function_version=config.reward_function_version,
        )
    )
    eval_env = Monitor(
        make_env(
            config.environment_backend,
            config.target_task,
            config.evaluation.task_set_seed,
            reward_function_version=config.reward_function_version,
        )
    )
    disable_task_sampling(eval_env)
    try:
        evaluation = FixedSeedEvalCallback(
            eval_env,
            eval_freq=config.evaluation.frequency,
            n_eval_episodes=config.evaluation.episodes,
            seed=config.evaluation.seed,
            log_path=run_dir / "evaluations",
            best_model_save_path=run_dir / "best_model",
            best_success_model_save_path=run_dir / "best_success_model",
            task_name=config.target_task,
            task_set_seed=config.evaluation.task_set_seed,
            evaluation_mode="fixed-tasks",
            evaluate_on_training_start=not args.resume,
        )
        callbacks = CallbackList(
            [
                TrainingSuccessCallback(),
                evaluation,
                GlobalCheckpointCallback(
                    save_freq=config.sac.checkpoint_frequency,
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix="sac",
                    save_replay_buffer=True,
                ),
            ]
        )
        if args.resume:
            model, _ = load_resumable_sac(
                resume_checkpoint, train_env, device=config.runtime.device
            )
            _hydrate_evaluations(evaluation, run_dir)
            remaining = remaining_timesteps(config.sac.steps, model.num_timesteps)
            reset_timesteps = False
        else:
            model = SAC(
                TransferableSACPolicy,
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
                policy_kwargs={
                    "net_arch": [256, 256],
                    "log_std_init": config.sac.log_std_init,
                },
                tensorboard_log=str(run_dir / "tensorboard"),
                seed=config.runtime.seed,
                device=config.runtime.device,
                verbose=0,
            )
            metadata["initial_actor_sha256"] = state_hash(model.policy.actor)
            metadata["initial_critic_sha256"] = state_hash(model.policy.critic)
            if config.condition == "transferred":
                transfer_general_policy_to_sac(model, policy)
                metadata["transferred_actor_sha256"] = state_hash(model.policy.actor)
            _write_run_metadata(run_dir / "metadata.json", metadata, initial=False)
            remaining = config.sac.steps
            reset_timesteps = True
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            tb_log_name="forward_transfer",
            progress_bar=True,
            reset_num_timesteps=reset_timesteps,
        )
        model.save(run_dir / "final_model")
        model.save_replay_buffer(run_dir / "final_replay_buffer.pkl")
        metadata.update(
            {
                "actual_final_timestep": model.num_timesteps,
                "final_model": {
                    "path": "final_model.zip",
                    "sha256": file_hash(run_dir / "final_model.zip"),
                },
                "final_replay_buffer": {
                    "path": "final_replay_buffer.pkl",
                    "sha256": file_hash(run_dir / "final_replay_buffer.pkl"),
                },
            }
        )
        _write_run_metadata(run_dir / "metadata.json", metadata, initial=False)
        atomic_write_json(
            run_dir / "summary.json", _summary(config, run_dir, source_identity)
        )
        (run_dir / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
        (run_dir / "RUNNING").unlink()
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
