"""Typed experiment settings, strict YAML loading, and runtime-only CLI overrides.

Relative output and checkpoint paths are relative to the working directory.
Omitted fields use the defaults defined for the selected experiment entry point.
"""

from __future__ import annotations

import argparse
import math
import re
import types
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml

SAC_ENT_COEF = "auto"
SAC_NET_ARCH = (256, 256, 256, 256)
BC_HIDDEN_SIZES = (256, 256, 256, 256)
FLOW_RESIDUAL_BLOCKS = 6
FLOW_TIME_EMBEDDING = "sinusoidal"
GENERATOR_GRADIENT_CLIP_NORM = 1.0
GENERATOR_VALIDATION_DRAWS = 4
FLOW_CHECKPOINT_SELECTION = "validation-loss"
FLOW_PROBABILITY_PATH = "linear"
FLOW_SOURCE_DISTRIBUTION = "standard-normal"
REPLAY_PRIORITY_STRATEGY = "uniform"


@dataclass(frozen=True)
class EnvironmentConfig:
    """MetaWorld environment semantics shared by training and evaluation."""

    backend: str = "metaworld-v3"
    reward_function_version: str = "v2"


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Optional development-only success-based stopping criterion."""

    enabled: bool = False
    min_steps: int = 100_000
    success_threshold: float = 0.95
    patience: int = 3


@dataclass(frozen=True)
class SACConfig:
    """SAC settings shared by standalone experts and the sequential baseline."""

    steps: int = 10_000
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    checkpoint_freq: int = 10_000
    checkpoint_mode: str = "low_disk"
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True)
class BCConfig:
    """Persistent GeneralPolicy architecture and stage-update objective."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    loss: str = "post-tanh-mse"  # Physical-action or pre-tanh latent-space MSE.
    task_conditioning: str = "none"  # "none" or explicit task-order one-hot.


@dataclass(frozen=True)
class DiffusionConfig:
    """Experiment settings; the model checkpoint schema remains separate."""

    epochs: int = 300
    batch_size: int = 8
    learning_rate: float = 0.001
    steps: int = 100
    width: int = 64
    action_space: str = "raw"
    action_clamp_epsilon: float = 1e-4
    generated_action_projection: str = "none"


@dataclass(frozen=True)
class FlowMatchingConfig:
    """Flow Matching training, temporal-network, and fixed-step ODE settings."""

    epochs: int = 300
    batch_size: int = 8
    learning_rate: float = 0.001
    width: int = 64
    time_embedding_scale: float = 100.0
    integration_steps: int = 100
    ode_solver: str = "euler"
    rtol: float = 1e-5
    atol: float = 1e-5


@dataclass(frozen=True)
class DiffusionModelConfig:
    """Existing serialized model dimensions and action-coordinate contract."""

    horizon: int = 200
    features: int = 22
    steps: int = 100
    width: int = 64
    num_tasks: int = 0
    diffusion_action_space: str = "raw"
    diffusion_action_clamp_epsilon: float = 1e-4


def _default_experts():
    return {
        "reach-v3": "runs/experts/metaworld-v3/reach-v3/policy.zip",
        "button-press-v3": "runs/experts/metaworld-v3/button-press-v3/policy.zip",
        "hammer-v3": "runs/experts/metaworld-v3/hammer-v3/policy.zip",
    }


@dataclass(frozen=True)
class ContinualConfig:
    """Ordered curriculum, expert sources, stage budget, and replay behavior."""

    tasks: List[str] = field(
        default_factory=lambda: ["reach-v3", "button-press-v3", "hammer-v3"]
    )
    experts: Dict[str, str] = field(default_factory=_default_experts)
    trajectories_per_task: int = 200
    replay_mode: str = "diffusion"  # Old-task generation or current-task-only BC.
    # ``generator-default`` preserves legacy behavior: DDPM uses a per-task
    # count while Flow Matching uses one fixed total.  New matched experiments
    # select ``fixed-total`` explicitly for either generator.
    replay_allocation: str = "generator-default"
    replay_budget: int = 200
    # Opt-in comparison protocol: historical groups are used only for training,
    # leaving the current task's real holdout as the common validation set.
    historical_replay_training_only: bool = False
    steps_per_task: int = 1_000_000
    # Update-free random-action steps after each sequential-SAC task switch.
    boundary_warmup_steps: int = 10_000


@dataclass(frozen=True)
class EvaluationConfig:
    """Deterministic reset and MT1 task-selection protocol for evaluation."""

    episodes: int = 50
    seed: int = 10_000
    mode: str = "fixed-tasks"  # Fixed MT1 bank or sampling anew on each reset.
    task_set_seed: int = 10_000  # Identifies the ordered fixed MT1 task bank.
    frequency: int = 5_000
    seed_offset: int = 10_000  # Single-task SAC uses runtime.seed + this offset.
    final_task_set_seeds: List[int] = field(default_factory=lambda: [20_000, 20_001])
    final_reset_seed: int = 30_000


@dataclass(frozen=True)
class RuntimeConfig:
    """Run identity, execution device, and output-location settings."""

    seed: int = 0
    device: str = "cpu"
    output: Optional[str] = None
    run_name: Optional[str] = None
    run_dir: Optional[str] = None  # Filled with the actual directory in saved configs.


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level typed configuration projected to the selected entry point."""

    experiment: str = "diffcrl"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    continual: ContinualConfig = field(default_factory=ContinualConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


_COMMON_SAC_FIELDS = (
    "learning_rate",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "gamma",
    "tau",
    "train_freq",
    "gradient_steps",
)
_CONTINUAL_EVALUATION_FIELDS = ("episodes", "seed", "mode", "task_set_seed")
_RUNTIME_FIELDS = ("seed", "device", "output", "run_name", "run_dir")
# Per-entry-point allowlist for validation and resolved-config serialization.
_EXPERIMENT_FIELDS = {
    "sac": {
        "environment": (
            "backend",
            "reward_function_version",
        ),
        "sac": (
            "steps", *_COMMON_SAC_FIELDS, "checkpoint_freq", "checkpoint_mode",
            "early_stopping",
        ),
        "continual": ("tasks",),
        "evaluation": (
            "episodes",
            "frequency",
            "seed_offset",
            "mode",
            "task_set_seed",
            "final_task_set_seeds",
            "final_reset_seed",
        ),
        "runtime": _RUNTIME_FIELDS,
    },
    "continual_sac": {
        "environment": ("reward_function_version",),
        "sac": _COMMON_SAC_FIELDS,
        "continual": ("tasks", "steps_per_task", "boundary_warmup_steps"),
        "evaluation": _CONTINUAL_EVALUATION_FIELDS,
        "runtime": _RUNTIME_FIELDS,
    },
    "diffcrl": {
        "environment": (
            "backend",
            "reward_function_version",
        ),
        "bc": tuple(BCConfig.__annotations__),
        "diffusion": tuple(DiffusionConfig.__annotations__),
        "flow_matching": tuple(FlowMatchingConfig.__annotations__),
        "continual": (
            "tasks", "experts", "trajectories_per_task", "replay_mode",
            "replay_allocation", "replay_budget",
            "historical_replay_training_only",
        ),
        "evaluation": _CONTINUAL_EVALUATION_FIELDS,
        "runtime": _RUNTIME_FIELDS,
    },
}


def default_config(experiment: str) -> ExperimentConfig:
    config = ExperimentConfig(experiment=experiment)
    if experiment == "diffcrl":
        return config
    if experiment == "sac":
        return replace(
            config,
            continual=replace(config.continual, tasks=["reach-v3"], experts={}),
            evaluation=replace(config.evaluation, episodes=5, mode="sampled"),
            runtime=replace(config.runtime, device="auto", output="runs"),
        )
    if experiment == "continual_sac":
        return replace(
            config,
            sac=replace(config.sac, learning_rate=3e-4, batch_size=256),
            continual=replace(config.continual, experts={}, replay_mode="none"),
            evaluation=replace(config.evaluation, episodes=100, mode="sampled"),
            runtime=replace(config.runtime, device="auto"),
        )
    raise ValueError("experiment must be diffcrl, sac, or continual_sac.")


def _check_experiment_fields(values: dict, experiment: str) -> None:
    allowed = _EXPERIMENT_FIELDS[experiment]
    for section, section_values in values.items():
        if section == "experiment":
            continue
        if section not in allowed:
            raise ValueError(
                f"Config section {section} is not used by experiment {experiment}."
            )
        if not isinstance(section_values, dict):
            continue  # _merge reports the section's type error.
        irrelevant = set(section_values) - set(allowed[section])
        if irrelevant:
            field_name = sorted(irrelevant)[0]
            raise ValueError(
                f"Config field {section}.{field_name} is not used by "
                f"experiment {experiment}."
            )


def config_to_dict(config: ExperimentConfig) -> dict:
    """Return only fields consumed by the selected experiment."""
    validate_config(config)
    values = asdict(config)
    projected = {"experiment": config.experiment}
    for section, fields in _EXPERIMENT_FIELDS[config.experiment].items():
        # Preserve the exact historical serialization and resume identity of
        # diffusion and no-replay runs. Flow settings are scientific inputs only
        # when Flow Matching is the selected replay method.
        if section == "flow_matching" and config.continual.replay_mode != "flow-matching":
            continue
        projected[section] = {
            name: values[section][name]
            for name in fields
            if name != "run_dir" or values[section][name] is not None
            if not (
                section == "bc"
                and name == "task_conditioning"
                and config.bc.task_conditioning == "none"
                or
                section == "continual"
                and (
                    name == "historical_replay_training_only"
                    and not config.continual.historical_replay_training_only
                    or
                    name == "replay_allocation"
                    and config.continual.replay_allocation == "generator-default"
                    or name == "replay_budget"
                    and (
                        config.continual.replay_mode == "none"
                        or config.continual.replay_mode == "diffusion"
                        and config.continual.replay_allocation == "generator-default"
                    )
                )
            )
        }
    return projected


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("YAML configuration keys must be strings.")
        if key in result:
            raise ValueError(f"Duplicate YAML field: {key}.")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def _typed(value, annotation, path):
    origin, args = get_origin(annotation), get_args(annotation)
    union_type = getattr(types, "UnionType", None)
    if origin is Union or union_type is not None and origin is union_type:
        if value is None and type(None) in args:
            return None
        return _typed(value, next(t for t in args if t is not type(None)), path)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list.")
        return [_typed(v, args[0], f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping.")
        return {
            _typed(k, args[0], path): _typed(v, args[1], f"{path}.{k}")
            for k, v in value.items()
        }
    if annotation is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number.")
        return float(value)
    if type(value) is not annotation:
        raise ValueError(
            f"{path} must be {annotation.__name__}, got {type(value).__name__}."
        )
    return value


def _merge(default, values, path=""):
    if not isinstance(values, dict):
        raise ValueError(f"{path or 'config'} must be a mapping.")
    hints = get_type_hints(type(default))
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(
            f"Unknown config field: {path + '.' if path else ''}{sorted(unknown, key=str)[0]}."
        )
    updates = {}
    for key, value in values.items():
        location = f"{path}.{key}" if path else key
        prior = getattr(default, key)
        updates[key] = (
            _merge(prior, value, location)
            if is_dataclass(prior)
            else _typed(value, hints[key], location)
        )
    return replace(default, **updates)


_LEGACY_FIXED_FIELDS = {
    "sac": {"ent_coef": SAC_ENT_COEF, "net_arch": list(SAC_NET_ARCH)},
    "bc": {
        "hidden_sizes": list(BC_HIDDEN_SIZES),
        "normalization_mode": "layer-norm",
        "use_layer_norm": True,
    },
    "flow_matching": {
        "residual_blocks": FLOW_RESIDUAL_BLOCKS,
        "time_embedding": FLOW_TIME_EMBEDDING,
        "gradient_clip_norm": GENERATOR_GRADIENT_CLIP_NORM,
        "validation_samples": GENERATOR_VALIDATION_DRAWS,
        "checkpoint_selection": FLOW_CHECKPOINT_SELECTION,
        "probability_path": FLOW_PROBABILITY_PATH,
        "source_distribution": FLOW_SOURCE_DISTRIBUTION,
    },
    "continual": {"replay_priority_strategy": REPLAY_PRIORITY_STRATEGY},
}


def _without_legacy_fixed_fields(values: dict) -> dict:
    """Normalize an old resolved config without accepting obsolete alternatives."""
    normalized = dict(values)
    for section, expected_fields in _LEGACY_FIXED_FIELDS.items():
        section_values = normalized.get(section)
        if not isinstance(section_values, dict):
            continue
        section_values = dict(section_values)
        for name, expected in expected_fields.items():
            if name not in section_values:
                continue
            actual = section_values.pop(name)
            if actual != expected:
                raise ValueError(
                    f"Legacy config field {section}.{name} must be {expected!r}, "
                    f"got {actual!r}."
                )
        normalized[section] = section_values
    return normalized


def config_from_dict(
    values: dict,
    *,
    expected_experiment: Optional[str] = None,
    overrides: Optional[dict] = None,
    allow_legacy_fixed_fields: bool = False,
) -> ExperimentConfig:
    if not isinstance(values, dict) or "experiment" not in values:
        raise ValueError("Missing required field: experiment.")
    if allow_legacy_fixed_fields:
        values = _without_legacy_fixed_fields(values)
    default = default_config(values["experiment"])
    _check_experiment_fields(values, default.experiment)
    if "flow_matching" in values and (
        default.experiment != "diffcrl"
        or values.get("continual", {}).get("replay_mode", "diffusion")
        != "flow-matching"
    ):
        raise ValueError(
            "Config section flow_matching requires continual.replay_mode: flow-matching."
        )
    continual_values = values.get("continual", {})
    if (
        isinstance(continual_values, dict)
        and {"replay_allocation", "replay_budget"} & set(continual_values)
        and continual_values.get("replay_mode", "diffusion") == "none"
        and continual_values.get("replay_allocation", "generator-default")
        != "fixed-total"
    ):
        raise ValueError(
            "No Replay accepts explicit replay settings only as a nominal fixed-total budget."
        )
    config = _merge(default, values)
    if expected_experiment is not None and config.experiment != expected_experiment:
        raise ValueError(
            f"experiment must be {expected_experiment} for this entry point."
        )
    if overrides:
        config = replace(config, runtime=_merge(config.runtime, overrides, "runtime"))
    validate_config(config)
    return config


def load_config(
    path,
    *,
    expected_experiment=None,
    overrides=None,
    allow_legacy_fixed_fields=False,
) -> ExperimentConfig:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot load config {path}: {error}") from error
    return config_from_dict(
        values,
        expected_experiment=expected_experiment,
        overrides=overrides,
        allow_legacy_fixed_fields=allow_legacy_fixed_fields,
    )


def validate_config(config: ExperimentConfig) -> None:
    # Validate direct dataclass callers as strictly as YAML callers.
    _merge(default_config(config.experiment), asdict(config))
    for section, names in {
        "sac": (
            "steps",
            "learning_rate",
            "buffer_size",
            "batch_size",
            "train_freq",
            "gradient_steps",
            "checkpoint_freq",
        ),
        "bc": ("epochs", "batch_size", "learning_rate"),
        "diffusion": ("epochs", "batch_size", "learning_rate"),
        "flow_matching": (
            "epochs",
            "batch_size",
            "learning_rate",
            "integration_steps",
        ),
        "continual": ("steps_per_task",),
        "evaluation": ("episodes", "frequency"),
    }.items():
        for name in names:
            if getattr(getattr(config, section), name) <= 0:
                raise ValueError(f"{section}.{name} must be positive.")
    for section, name in [
        ("sac", "learning_starts"),
        ("continual", "boundary_warmup_steps"),
        ("evaluation", "seed_offset"),
    ]:
        if getattr(getattr(config, section), name) < 0:
            raise ValueError(f"{section}.{name} must be nonnegative.")
    for section, name in [
        ("runtime", "seed"),
        ("evaluation", "seed"),
        ("evaluation", "task_set_seed"),
    ]:
        if not 0 <= getattr(getattr(config, section), name) < 2**32:
            raise ValueError(f"{section}.{name} must be in [0, 2**32).")
    for name in ("gamma", "tau"):
        if not 0 < getattr(config.sac, name) <= 1:
            raise ValueError(f"sac.{name} must be in (0, 1].")
    if config.sac.early_stopping.min_steps < 0:
        raise ValueError("sac.early_stopping.min_steps must be nonnegative.")
    if config.sac.early_stopping.patience <= 0:
        raise ValueError("sac.early_stopping.patience must be positive.")
    if not 0 <= config.sac.early_stopping.success_threshold <= 1:
        raise ValueError(
            "sac.early_stopping.success_threshold must be in [0, 1]."
        )
    for path, allowed in {
        "environment.backend": ("metaworld-v3", "kuka-v3"),
        "environment.reward_function_version": ("v1", "v2"),
        "bc.loss": ("post-tanh-mse", "pre-tanh-mse"),
        "bc.task_conditioning": ("none", "one_hot"),
        "diffusion.action_space": ("raw", "pre-tanh"),
        "diffusion.generated_action_projection": ("none", "clip"),
        "continual.replay_mode": ("diffusion", "flow-matching", "real", "none"),
        "evaluation.mode": ("fixed-tasks", "sampled"),
        "sac.checkpoint_mode": ("low_disk", "full"),
    }.items():
        section, name = path.split(".")
        if getattr(getattr(config, section), name) not in allowed:
            raise ValueError(f"{path} must be one of {allowed}.")
    if config.experiment == "sac":
        final_seeds = config.evaluation.final_task_set_seeds
        if not final_seeds or len(set(final_seeds)) != len(final_seeds):
            raise ValueError("evaluation.final_task_set_seeds must be nonempty and unique.")
        for seed in (*final_seeds, config.evaluation.final_reset_seed):
            if not 0 <= seed < 2**32:
                raise ValueError("Final evaluation seeds must be in [0, 2**32).")
    if (
        config.diffusion.steps < 2
        or config.diffusion.width < 4
        or config.diffusion.width % 4
    ):
        raise ValueError(
            "diffusion.steps must be >=2; diffusion.width must be a positive multiple of 4."
        )
    if not 0 < config.diffusion.action_clamp_epsilon < 1:
        raise ValueError("diffusion.action_clamp_epsilon must be in (0, 1).")
    flow = config.flow_matching
    if flow.width < 4 or flow.width % 4:
        raise ValueError("flow_matching.width must be a positive multiple of 4.")
    if not math.isfinite(flow.time_embedding_scale) or flow.time_embedding_scale <= 0:
        raise ValueError("flow_matching.time_embedding_scale must be positive and finite.")
    if (
        not math.isfinite(flow.rtol)
        or not math.isfinite(flow.atol)
        or flow.rtol <= 0
        or flow.atol <= 0
    ):
        raise ValueError("flow_matching.rtol and flow_matching.atol must be positive and finite.")
    if flow.ode_solver not in ("euler", "midpoint", "heun2", "rk4", "dopri5"):
        raise ValueError("flow_matching.ode_solver is unsupported.")
    if config.continual.trajectories_per_task < 2:
        raise ValueError(
            "continual.trajectories_per_task must be >=2 for a disjoint holdout."
        )
    if isinstance(config.continual.replay_budget, bool) or config.continual.replay_budget < 0:
        raise ValueError("continual.replay_budget must be a nonnegative integer.")
    if config.continual.replay_allocation not in ("generator-default", "fixed-total"):
        raise ValueError(
            "continual.replay_allocation must be 'generator-default' or 'fixed-total'."
        )
    tasks = config.continual.tasks
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("continual.tasks must be a nonempty list of unique task names.")
    if config.environment.backend == "metaworld-v3" and any(
        not task.endswith("-v3") for task in tasks
    ):
        raise ValueError(
            "continual.tasks must be a nonempty list of unique MetaWorld v3 task names."
        )
    kuka_v3_tasks = {
        "kuka-reach-v3",
        "kuka-push-v3",
        "kuka-push-cube-v3",
        "kuka-handle-press-side-v3",
        "kuka-button-press-v3",
        "kuka-button-press-topdown-v3",
        "kuka-peg-unplug-side-v3",
        "kuka-hammer-v3",
        "kuka-drawer-close-v3",
        "kuka-faucet-open-v3",
        "kuka-faucet-close-v3",
        "kuka-window-open-v3",
        "kuka-window-close-v3",
        "kuka-door-open-v3",
        "kuka-pick-place-v3",
    }
    if config.environment.backend == "kuka-v3" and any(
        task not in kuka_v3_tasks for task in tasks
    ):
        raise ValueError(
            "The kuka-v3 backend supports only these tasks: "
            f"{sorted(kuka_v3_tasks)}."
        )
    if config.environment.backend == "kuka-v3" and config.experiment not in (
        "sac",
        "diffcrl",
    ):
        raise ValueError("The kuka-v3 backend is implemented only for sac and diffcrl.")
    if config.experiment == "sac" and len(tasks) != 1:
        raise ValueError("continual.tasks must contain exactly one task for sac.")
    if config.experiment == "continual_sac" and config.continual.replay_mode != "none":
        raise ValueError("continual.replay_mode must be none for continual_sac.")
    if config.continual.replay_mode == "none" and not (
        (
            config.continual.replay_allocation == "generator-default"
            and config.continual.replay_budget == 200
        )
        or config.continual.replay_allocation == "fixed-total"
    ):
        raise ValueError(
            "No Replay accepts either legacy replay defaults or a nominal fixed-total budget."
        )
    if (
        config.continual.historical_replay_training_only
        and config.continual.replay_mode == "none"
    ):
        raise ValueError("historical_replay_training_only requires replay.")
    if config.evaluation.mode == "fixed-tasks" and config.evaluation.episodes > 50:
        raise ValueError("evaluation.episodes cannot exceed 50 in fixed-tasks mode.")
    if config.experiment == "diffcrl":
        for task in tasks:
            if not config.continual.experts.get(task):
                raise ValueError(f"Missing required field: continual.experts.{task}.")
        a, b = config.runtime.seed, config.evaluation.seed
        if max(a, b) < min(
            a + config.continual.trajectories_per_task, b + config.evaluation.episodes
        ):
            raise ValueError(
                "runtime.seed collection range and evaluation.seed range must be disjoint."
            )
    runtime = config.runtime
    if not runtime.output or not runtime.output.strip():
        raise ValueError("Missing required field: runtime.output.")
    if runtime.run_name is not None and (
        not runtime.run_name.strip()
        or runtime.run_name in (".", "..")
        or "/" in runtime.run_name
        or "\\" in runtime.run_name
    ):
        raise ValueError(
            "runtime.run_name must be a nonempty directory name, not a path."
        )
    if re.fullmatch(r"(cpu|auto|cuda(?::[0-9]+)?)", runtime.device) is None:
        raise ValueError("runtime.device must be cpu, auto, cuda, or cuda:N.")


def resolve_device(config: ExperimentConfig) -> ExperimentConfig:
    """Resolve auto and fail on unavailable CUDA before creating a run directory."""
    import torch

    device = config.runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(
                f"runtime.device {device!r} is unavailable on this machine."
            )
    return replace(config, runtime=replace(config.runtime, device=device))


def parse_config(
    experiment: str,
    description: str,
    argv=None,
    *,
    config_required: bool = False,
    configure_parser=None,
    return_args: bool = False,
) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=description)
    config_argument = {"type": Path, "help": "Experiment YAML."}
    if config_required:
        config_argument["required"] = True
    else:
        config_argument["default"] = {
            "sac": Path("configs/sac/sac.yaml"),
            "diffcrl": Path("configs/diffcrl/diffcrl.yaml"),
            "continual_sac": Path("configs/continual_sac/continual_sac.yaml"),
        }[experiment]
        config_argument["help"] = "Experiment YAML (default: %(default)s)."
    parser.add_argument("--config", **config_argument)
    parser.add_argument("--seed", type=int, default=None, help="Override runtime.seed.")
    parser.add_argument(
        "--device",
        default=None,
        help="Override runtime.device: cpu, auto, cuda, cuda:N.",
    )
    parser.add_argument("--run-name", default=None, help="Override runtime.run_name.")
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args(argv)
    overrides = {
        name: getattr(args, name)
        for name in ("seed", "device", "run_name")
        if getattr(args, name) is not None
    }
    try:
        config = resolve_device(
            load_config(
                args.config, expected_experiment=experiment, overrides=overrides
            )
        )
        return (config, args) if return_args else config
    except ValueError as error:
        parser.error(str(error))


def output_directory(config: ExperimentConfig) -> Path:
    directory = Path(config.runtime.output)
    if config.runtime.run_name is not None:
        directory /= config.runtime.run_name
    return directory.resolve()


def save_resolved_config(config: ExperimentConfig, run_dir: Path) -> None:
    resolved = replace(
        config, runtime=replace(config.runtime, run_dir=str(run_dir.resolve()))
    )
    with (run_dir / "config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config_to_dict(resolved), stream, sort_keys=False)
