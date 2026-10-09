"""One persistent SAC, sequential tasks, no cross-task replay. No expert loading.

Each task gets a fixed step budget. Boundary warm-up uses random actions and
withholds updates; initial learning_starts is configured separately. Global timesteps persist.
ReplayBuffer.reset logically discards all prior-task transitions; old storage is
not sampled. An unfinished episode at a budget boundary is discarded with it.
"""

import random
from collections import deque

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.config import (
    ExperimentConfig,
    output_directory,
    resolve_device,
    save_resolved_config,
    validate_config,
)
from src.continual.evaluation import EvaluationMatrix, evaluate_stage
from src.continual.policy_adapter import SB3SACPolicy
from src.continual.reporting import print_final_report, print_stage_report
from src.continual.task_sequence import TaskSequence
from src.envs import make_metaworld_env
from src.support.io import write_json
from src.support.reproducibility import file_hash
from src.support.run_manifest import complete_run, create_run_manifest
from src.continual.task_bank import verify_disjoint_task_banks


class FiniteTransitions(BaseCallback):
    """Abort sequential SAC if SB3 emits non-finite transition values."""

    def _on_step(self):
        for key in ("new_obs", "actions", "rewards"):
            if not np.isfinite(self.locals[key]).all():
                raise ValueError(f"Nonfinite training {key}.")
        return True


def switch_task(model, task_name, seed, warmup_steps, *, reward_function_version="v2"):
    """Retain all learning state; replace env and logically empty standard replay."""
    if model.replay_buffer is None or model.optimize_memory_usage:
        raise ValueError("Require standard non-memory-optimized replay.")
    old_env = model.get_env()
    new_env = Monitor(
        make_metaworld_env(
            task_name, seed, reward_function_version=reward_function_version
        )
    )
    try:
        model.set_env(new_env, force_reset=True)
    except Exception:
        new_env.close()
        raise
    old_env.close()
    model.replay_buffer.reset()
    assert model.replay_buffer.size() == 0
    model.learning_starts = model.num_timesteps + warmup_steps
    model.get_env().seed(seed)  # Applied to the next reset; no model RNG reseeding.
    model.ep_info_buffer = deque(maxlen=model._stats_window_size)
    model.ep_success_buffer = deque(maxlen=model._stats_window_size)


def run(config: ExperimentConfig, *, observer=None):
    """Run the baseline. Optional observer(event,stage,model) supports boundary tests."""
    validate_config(config)
    if config.experiment != "continual_sac":
        raise ValueError("Sequential SAC requires experiment: continual_sac.")
    config = resolve_device(config)
    sequence = TaskSequence.from_names(config.continual.tasks)
    output = output_directory(config)
    task_bank_checks = (
        {
            task.task_name: verify_disjoint_task_banks(
                task.task_name,
                config.runtime.seed + stage,
                config.evaluation.task_set_seed,
            )
            for stage, task in enumerate(sequence.tasks)
        }
        if config.evaluation.mode == "fixed-tasks"
        else {}
    )
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output)
    create_run_manifest(output, config, task_banks=task_bank_checks)
    write_json(
        output / "config.json",
        {
            "tasks": config.continual.tasks,
            "steps_per_task": config.continual.steps_per_task,
            "seed": config.runtime.seed,
            "output": str(output),
            "algorithm": "sequential SAC / no replay",
            "eval_episodes": config.evaluation.episodes,
            "eval_seed": config.evaluation.seed,
            "learning_starts": config.sac.learning_starts,
            "boundary_warmup_steps": config.continual.boundary_warmup_steps,
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
            "boundary": "retain policy/critics/targets/optimizers/entropy; reset replay only",
            "task_seed": "seed + stage",
            "warmup": "random actions each stage; global threshold=start+warmup",
        },
    )
    torch.set_num_threads(1)
    env = Monitor(
        make_metaworld_env(
            sequence.task(0).task_name,
            config.runtime.seed,
            reward_function_version=config.environment.reward_function_version,
        )
    )
    model = None
    try:
        model = SAC(
            "MlpPolicy",
            env,
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
            optimize_memory_usage=False,
            seed=config.runtime.seed,
            device=config.runtime.device,
            verbose=0,
        )
        matrix = EvaluationMatrix(sequence)
        stages = []
        for stage, task in enumerate(sequence.tasks):
            start = model.num_timesteps
            if stage:
                switch_task(
                    model,
                    task.task_name,
                    config.runtime.seed + stage,
                    config.continual.boundary_warmup_steps,
                    reward_function_version=config.environment.reward_function_version,
                )
            if observer:
                observer("boundary", stage, model)
            print(
                f"\nStage {stage + 1}/{sequence.num_tasks}: {task.task_name}",
                flush=True,
            )
            model.learn(
                total_timesteps=config.continual.steps_per_task,
                reset_num_timesteps=False,
                callback=FiniteTransitions(),
                progress_bar=True,
            )
            if model.num_timesteps != start + config.continual.steps_per_task:
                raise RuntimeError("Unexpected stage timestep count.")
            if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
                raise ValueError("Nonfinite model parameters.")
            if not torch.isfinite(model.log_ent_coef).all():
                raise ValueError("Nonfinite entropy state.")
            stage_dir = output / f"stage_{stage}_{task.task_name}"
            stage_dir.mkdir()
            checkpoint = output / f"after_task_{stage}_{task.task_name}.zip"
            model.save(checkpoint)
            # Evaluation must not consume training RNG streams.
            rng = (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            )
            try:
                evaluate_stage(
                    SB3SACPolicy(model),
                    matrix,
                    stage,
                    episodes=config.evaluation.episodes,
                    seed=config.evaluation.seed,
                    evaluation_mode=config.evaluation.mode,
                    task_set_seed=config.evaluation.task_set_seed,
                    reward_function_version=config.environment.reward_function_version,
                    progress=True,
                )
            finally:
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                if rng[3] is not None:
                    torch.cuda.set_rng_state_all(rng[3])
            record = {
                "stage": stage,
                "task": task.task_name,
                "task_seed": config.runtime.seed + stage,
                "seed": config.runtime.seed,
                "local_start": 0,
                "local_end": model.num_timesteps - start,
                "global_start": start,
                "global_end": model.num_timesteps,
                "task_steps": model.num_timesteps - start,
                "replay_size": model.replay_buffer.size(),
                "learning_starts_global": model.learning_starts,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_hash(checkpoint),
            }
            matrix.stage_provenance[str(stage)] = record
            write_json(stage_dir / "evaluation_matrix.json", matrix.to_dict())
            write_json(stage_dir / "stage.json", record)
            stages.append(record)
            if observer:
                observer("stage_end", stage, model)
            print_stage_report(
                matrix,
                stage,
                task_training_steps=record["task_steps"],
                global_timesteps=model.num_timesteps,
            )
        model.save(output / "final_model.zip")
        write_json(output / "evaluation_matrix.json", matrix.to_dict())
        write_json(output / "stages.json", stages)
        print_final_report(matrix)
        complete_run(output)
        return model
    finally:
        if model is not None:
            model.get_env().close()
        else:
            env.close()
