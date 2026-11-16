"""Scientific validation for typed experiment configurations."""

from __future__ import annotations

import math
import re

from .schema import (
    ContinualEvaluationConfig,
    DiffCRLExperimentConfig,
    EnvironmentConfig,
    RuntimeConfig,
    SACExperimentConfig,
)


def _positive(section, names: tuple[str, ...], prefix: str) -> None:
    for name in names:
        value = getattr(section, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{prefix}.{name} must be positive.")


def _validate_runtime(runtime: RuntimeConfig) -> None:
    if not 0 <= runtime.seed < 2**32:
        raise ValueError("runtime.seed must be in [0, 2**32).")
    if not runtime.output or not runtime.output.strip():
        raise ValueError("Missing required field: runtime.output.")
    if runtime.run_name is not None and (
        not runtime.run_name.strip()
        or runtime.run_name in (".", "..")
        or "/" in runtime.run_name
        or "\\" in runtime.run_name
    ):
        raise ValueError("runtime.run_name must be a nonempty directory name, not a path.")
    if re.fullmatch(r"(cpu|auto|cuda(?::[0-9]+)?)", runtime.device) is None:
        raise ValueError("runtime.device must be cpu, auto, cuda, or cuda:N.")


def _validate_environment(environment: EnvironmentConfig) -> None:
    if environment.backend not in ("metaworld-v3", "kuka-v3"):
        raise ValueError("environment.backend must be metaworld-v3 or kuka-v3.")
    if environment.reward_function_version not in ("v1", "v2"):
        raise ValueError("environment.reward_function_version must be v1 or v2.")


def _validate_tasks(tasks: list[str], backend: str) -> None:
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("continual.tasks must be a nonempty list of unique task names.")
    if backend == "metaworld-v3" and any(not task.endswith("-v3") for task in tasks):
        raise ValueError("continual.tasks must contain MetaWorld v3 task names.")
    if backend == "kuka-v3":
        from continual_manipulation.environments.kuka_v3 import KUKA_V3_TASKS

        unsupported = [task for task in tasks if task not in KUKA_V3_TASKS]
        if unsupported:
            raise ValueError(
                "The kuka-v3 backend supports only registered KUKA task names. "
                f"Unsupported tasks: {', '.join(unsupported)}."
            )


def _validate_continual_evaluation(evaluation: ContinualEvaluationConfig) -> None:
    _positive(evaluation, ("episodes",), "evaluation")
    for name in ("seed", "task_set_seed"):
        if not 0 <= getattr(evaluation, name) < 2**32:
            raise ValueError(f"evaluation.{name} must be in [0, 2**32).")
    if evaluation.mode not in ("fixed-tasks", "sampled"):
        raise ValueError("evaluation.mode must be fixed-tasks or sampled.")
    if evaluation.mode == "fixed-tasks" and evaluation.episodes > 50:
        raise ValueError("evaluation.episodes cannot exceed 50 in fixed-tasks mode.")


def _validate_sac_values(sac, prefix: str = "sac") -> None:
    _positive(
        sac,
        (
            "learning_rate",
            "buffer_size",
            "batch_size",
            "train_freq",
            "gradient_steps",
        ),
        prefix,
    )
    for name in ("learning_starts", "random_exploration_steps"):
        if getattr(sac, name) < 0:
            raise ValueError(f"{prefix}.{name} must be nonnegative.")
    if not math.isfinite(sac.target_output_std) or sac.target_output_std <= 0:
        raise ValueError(f"{prefix}.target_output_std must be positive.")
    for name in ("gamma", "tau"):
        if not 0 < getattr(sac, name) <= 1:
            raise ValueError(f"{prefix}.{name} must be in (0, 1].")


def _validate_sac(config: SACExperimentConfig) -> None:
    _validate_environment(config.environment)
    _validate_sac_values(config.sac)
    _positive(config.sac, ("steps",), "sac")
    actor_init = config.actor_init
    if actor_init.mode not in ("scratch", "sac_actor", "general_actor"):
        raise ValueError(
            "actor_init.mode must be scratch, sac_actor, or general_actor."
        )
    if actor_init.mode == "scratch" and actor_init.checkpoint is not None:
        raise ValueError("actor_init.checkpoint must be null in scratch mode.")
    if actor_init.mode != "scratch" and (
        actor_init.checkpoint is None or not actor_init.checkpoint.strip()
    ):
        raise ValueError(
            f"actor_init.checkpoint is required in {actor_init.mode} mode."
        )
    resume = config.resume
    if resume.enabled and actor_init.mode != "scratch":
        raise ValueError("resume and Actor Warm-Start are mutually exclusive.")
    if resume.enabled and (resume.checkpoint is None or not resume.checkpoint.strip()):
        raise ValueError("resume.checkpoint is required when resume.enabled is true.")
    if not resume.enabled and (resume.checkpoint is not None or resume.replay_buffer is not None):
        raise ValueError("resume paths must be null when resume.enabled is false.")
    if resume.refill_steps < 0:
        raise ValueError("resume.refill_steps must be nonnegative.")
    if not resume.enabled and resume.refill_steps != 0:
        raise ValueError("resume.refill_steps must be zero when resume is disabled.")
    if resume.replay_buffer is not None and resume.refill_steps != 0:
        raise ValueError("resume.refill_steps must be zero when a replay buffer is restored.")
    if resume.enabled and resume.replay_buffer is None and resume.refill_steps >= config.sac.steps:
        raise ValueError("resume.refill_steps must be smaller than the additional SAC step budget.")
    stop = config.sac.early_stopping
    if stop.min_steps < 0:
        raise ValueError("sac.early_stopping.min_steps must be nonnegative.")
    if stop.patience <= 0:
        raise ValueError("sac.early_stopping.patience must be positive.")
    if stop.plateau_patience is not None and stop.plateau_patience <= 0:
        raise ValueError("sac.early_stopping.plateau_patience must be positive or null.")
    if not math.isfinite(stop.min_delta) or stop.min_delta < 0:
        raise ValueError("sac.early_stopping.min_delta must be nonnegative.")
    if not 0 <= stop.success_threshold <= 1:
        raise ValueError("sac.early_stopping.success_threshold must be in [0, 1].")
    _validate_tasks(config.continual.tasks, config.environment.backend)
    if len(config.continual.tasks) != 1:
        raise ValueError("continual.tasks must contain exactly one task for sac.")
    evaluation = config.evaluation
    _positive(evaluation, ("episodes", "frequency"), "evaluation")
    if evaluation.final_episodes is not None and evaluation.final_episodes <= 0:
        raise ValueError("evaluation.final_episodes must be positive or null.")
    if evaluation.seed_offset < 0:
        raise ValueError("evaluation.seed_offset must be nonnegative.")
    if evaluation.mode not in ("fixed-tasks", "sampled"):
        raise ValueError("evaluation.mode must be fixed-tasks or sampled.")
    if not 0 <= evaluation.task_set_seed < 2**32:
        raise ValueError("evaluation.task_set_seed must be in [0, 2**32).")
    if not evaluation.final_task_set_seeds or len(set(evaluation.final_task_set_seeds)) != len(
        evaluation.final_task_set_seeds
    ):
        raise ValueError("evaluation.final_task_set_seeds must be nonempty and unique.")
    for seed in (*evaluation.final_task_set_seeds, evaluation.final_reset_seed):
        if not 0 <= seed < 2**32:
            raise ValueError("Final evaluation seeds must be in [0, 2**32).")


def _validate_diffcrl(config: DiffCRLExperimentConfig) -> None:
    _validate_environment(config.environment)
    _validate_tasks(config.continual.tasks, config.environment.backend)
    _positive(config.bc, ("epochs", "batch_size", "learning_rate"), "bc")
    if config.bc.loss not in (
        "squashed-gaussian-nll",
        "squashed_gaussian_nll",
        "deterministic_mse",
    ):
        raise ValueError(
            "bc.loss must select squashed_gaussian_nll or deterministic_mse."
        )
    if config.bc.task_conditioning not in ("none", "one_hot"):
        raise ValueError("bc.task_conditioning must be none or one_hot.")
    if config.bc.activation not in ("relu", "leaky_relu"):
        raise ValueError("bc.activation must be relu or leaky_relu.")
    if config.bc.hidden_dim not in (256, 512):
        raise ValueError("bc.hidden_dim must be 256 or 512.")
    if not -20 <= config.bc.initial_log_std <= 2:
        raise ValueError("bc.initial_log_std must be in [-20, 2].")
    if config.bc.dual_checkpoint_selection and config.bc.loss not in (
        "squashed-gaussian-nll",
        "squashed_gaussian_nll",
    ):
        raise ValueError(
            "bc.dual_checkpoint_selection requires squashed_gaussian_nll."
        )
    if config.bc.checkpoint_selection not in (
        "validation_nll",
        "validation_physical_mse",
    ):
        raise ValueError(
            "bc.checkpoint_selection must be validation_nll or "
            "validation_physical_mse."
        )
    if (
        config.bc.checkpoint_selection == "validation_physical_mse"
        and config.bc.loss not in (
            "squashed-gaussian-nll",
            "squashed_gaussian_nll",
        )
    ):
        raise ValueError(
            "validation_physical_mse checkpoint selection requires "
            "squashed_gaussian_nll."
        )
    if (
        config.bc.dual_checkpoint_selection
        and config.bc.checkpoint_selection != "validation_nll"
    ):
        raise ValueError(
            "Dual checkpoint evaluation requires validation_nll to remain "
            "the authoritative checkpoint selection."
        )

    diffusion = config.diffusion
    _positive(diffusion, ("epochs", "batch_size", "learning_rate"), "diffusion")
    if diffusion.backbone not in ("temporal_cnn", "temporal_unet1d"):
        raise ValueError(
            "diffusion.backbone must be temporal_cnn or temporal_unet1d."
        )
    if diffusion.steps < 2 or diffusion.width < 4 or diffusion.width % 4:
        raise ValueError(
            "diffusion.steps must be >=2; diffusion.width must be a positive multiple of 4."
        )
    if diffusion.action_space not in ("raw", "pre-tanh"):
        raise ValueError("diffusion.action_space must be raw or pre-tanh.")
    if not 0 < diffusion.action_clamp_epsilon < 1:
        raise ValueError("diffusion.action_clamp_epsilon must be in (0, 1).")
    if diffusion.generated_action_projection not in ("none", "clip"):
        raise ValueError("diffusion.generated_action_projection must be none or clip.")

    flow = config.flow_matching
    _positive(
        flow,
        ("epochs", "batch_size", "learning_rate", "integration_steps"),
        "flow_matching",
    )
    if flow.backbone not in ("temporal_cnn", "temporal_unet1d"):
        raise ValueError(
            "flow_matching.backbone must be temporal_cnn or temporal_unet1d."
        )
    if flow.width < 4 or flow.width % 4:
        raise ValueError(
            "flow_matching.width must be a positive multiple of 4 so every "
            "GroupNorm scale is valid."
        )
    if not math.isfinite(flow.time_embedding_scale) or flow.time_embedding_scale <= 0:
        raise ValueError("flow_matching.time_embedding_scale must be positive and finite.")
    if not all(math.isfinite(value) and value > 0 for value in (flow.rtol, flow.atol)):
        raise ValueError("flow_matching.rtol and flow_matching.atol must be positive and finite.")
    if flow.ode_solver not in ("euler", "midpoint", "heun2", "rk4", "dopri5"):
        raise ValueError("flow_matching.ode_solver is unsupported.")

    continual = config.continual
    if continual.trajectories_per_task < 2:
        raise ValueError("continual.trajectories_per_task must be >=2 for a disjoint holdout.")
    schedule = continual.current_trajectory_schedule
    if schedule:
        if len(schedule) != len(continual.tasks):
            raise ValueError(
                "continual.current_trajectory_schedule must contain one count per task."
            )
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 2
            for count in schedule
        ):
            raise ValueError(
                "continual.current_trajectory_schedule counts must be integers >=2."
            )
    if continual.replay_mode == "real":
        raise ValueError(
            "continual.replay_mode 'real' is a retired stored-trajectory protocol; "
            "historical YAML remains archival and is not executable. Use "
            "'expert-recollection' for fresh historical expert rollouts."
        )
    if continual.replay_mode not in (
        "diffusion",
        "flow-matching",
        "expert-recollection",
        "none",
    ):
        raise ValueError("continual.replay_mode is unsupported.")
    if continual.replay_allocation not in (
        "generator-default",
        "fixed-total",
        "fixed-weighted",
    ):
        raise ValueError(
            "continual.replay_allocation must be generator-default, fixed-total, "
            "or fixed-weighted."
        )
    if (
        isinstance(continual.replay_budget, bool)
        or not isinstance(continual.replay_budget, int)
        or continual.replay_budget < 0
    ):
        raise ValueError("continual.replay_budget must be a nonnegative integer.")
    weights = continual.replay_task_weights
    if continual.replay_allocation == "fixed-weighted":
        if continual.replay_mode not in ("diffusion", "flow-matching"):
            raise ValueError(
                "fixed-weighted allocation is supported only for diffusion and "
                "flow-matching replay."
            )
        if len(weights) != len(continual.tasks):
            raise ValueError(
                "continual.replay_task_weights must contain one weight per task."
            )
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in weights
        ):
            raise ValueError(
                "continual.replay_task_weights must be finite nonnegative numbers."
            )
        if len(continual.tasks) > 1 and sum(weights[:1]) <= 0:
            raise ValueError(
                "Historical task weights must have a positive sum when replay is required."
            )
    elif weights:
        raise ValueError(
            "continual.replay_task_weights is valid only with fixed-weighted allocation."
        )
    if continual.replay_mode == "none" and not (
        continual.replay_allocation == "generator-default" and continual.replay_budget == 200
        or continual.replay_allocation == "fixed-total"
    ):
        raise ValueError("No Replay accepts only the default or a nominal fixed-total budget.")
    if continual.historical_replay_training_only and continual.replay_mode == "none":
        raise ValueError("historical_replay_training_only requires replay.")
    if continual.collection_mode not in ("all", "success_preferred"):
        raise ValueError("continual.collection_mode must be all or success_preferred.")
    if continual.expert_action_mode not in ("deterministic", "stochastic"):
        raise ValueError(
            "continual.expert_action_mode must be deterministic or stochastic."
        )
    if (
        isinstance(continual.collection_max_attempt_multiplier, bool)
        or not isinstance(continual.collection_max_attempt_multiplier, int)
        or continual.collection_max_attempt_multiplier < 1
    ):
        raise ValueError(
            "continual.collection_max_attempt_multiplier must be an integer >= 1."
        )
    for task in continual.tasks:
        if not continual.experts.get(task):
            raise ValueError(f"Missing required field: continual.experts.{task}.")

    _validate_continual_evaluation(config.evaluation)
    collection_seed = config.runtime.seed
    evaluation_seed = config.evaluation.seed
    collection_count = max(schedule) if schedule else continual.trajectories_per_task
    if max(collection_seed, evaluation_seed) < min(
        collection_seed + collection_count,
        evaluation_seed + config.evaluation.episodes,
    ):
        raise ValueError("runtime.seed collection range and evaluation.seed range must be disjoint.")


def validate_config(config) -> None:
    if isinstance(config, SACExperimentConfig):
        _validate_sac(config)
    elif isinstance(config, DiffCRLExperimentConfig):
        _validate_diffcrl(config)
    else:
        raise TypeError("Unsupported configuration type.")
    _validate_runtime(config.runtime)
