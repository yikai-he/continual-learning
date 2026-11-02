"""Configuration, temporal validation, and metrics for forward transfer."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.continual.bc_policy import GeneralPolicy
from src.support.reproducibility import file_hash


@dataclass(frozen=True)
class ForwardTransferSACConfig:
    steps: int = 100_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    log_std_init: float = -3.0
    checkpoint_frequency: int = 10_000
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256, 256])


@dataclass(frozen=True)
class ForwardTransferEvaluationConfig:
    frequency: int = 10_000
    episodes: int = 50
    seed: int = 10_000
    task_set_seed: int = 10_000
    success_threshold: float = 0.8


@dataclass(frozen=True)
class ForwardTransferRuntimeConfig:
    seed: int = 0
    device: str = "cpu"
    output: str = "runs/forward_transfer"


@dataclass(frozen=True)
class ForwardTransferConfig:
    experiment: str = "forward_transfer"
    condition: str = "random"
    target_task: str = "hammer-v3"
    task_history: list[str] = field(
        default_factory=lambda: ["button-press-v3", "faucet-close-v3"]
    )
    source_checkpoint: str = ""
    environment_backend: str = "metaworld-v3"
    reward_function_version: str = "v2"
    sac: ForwardTransferSACConfig = field(default_factory=ForwardTransferSACConfig)
    evaluation: ForwardTransferEvaluationConfig = field(
        default_factory=ForwardTransferEvaluationConfig
    )
    runtime: ForwardTransferRuntimeConfig = field(
        default_factory=ForwardTransferRuntimeConfig
    )


def _construct(data: dict, cls, section: str):
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {section} fields: {sorted(unknown)}")
    return cls(**data)


def load_forward_transfer_config(
    path: str | Path, *, seed: int | None = None
) -> ForwardTransferConfig:
    """Strictly load the standalone experiment configuration."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Forward-transfer config must be a YAML mapping.")
    allowed = set(ForwardTransferConfig.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown forward-transfer fields: {sorted(unknown)}")
    values = dict(data)
    values["sac"] = _construct(values.get("sac", {}), ForwardTransferSACConfig, "sac")
    values["evaluation"] = _construct(
        values.get("evaluation", {}),
        ForwardTransferEvaluationConfig,
        "evaluation",
    )
    runtime = _construct(
        values.get("runtime", {}), ForwardTransferRuntimeConfig, "runtime"
    )
    if seed is not None:
        runtime = ForwardTransferRuntimeConfig(
            seed=seed, device=runtime.device, output=runtime.output
        )
    values["runtime"] = runtime
    config = ForwardTransferConfig(**values)
    validate_forward_transfer_config(config)
    return config


def validate_forward_transfer_config(config: ForwardTransferConfig) -> None:
    if config.experiment != "forward_transfer":
        raise ValueError("experiment must be forward_transfer.")
    if config.condition not in ("random", "transferred"):
        raise ValueError("condition must be random or transferred.")
    if config.environment_backend != "metaworld-v3":
        raise ValueError("Initial forward-transfer experiments require metaworld-v3.")
    if not config.task_history or config.target_task in config.task_history:
        raise ValueError("task_history must be nonempty and precede the target task.")
    if not config.source_checkpoint:
        raise ValueError("source_checkpoint is required for paired provenance.")
    if config.sac.steps <= 0 or config.sac.steps % config.evaluation.frequency:
        raise ValueError("sac.steps must be a positive multiple of evaluation.frequency.")
    for name in (
        "buffer_size", "learning_starts", "batch_size", "train_freq",
        "gradient_steps", "checkpoint_frequency",
    ):
        if getattr(config.sac, name) <= 0:
            raise ValueError(f"sac.{name} must be positive.")
    if config.sac.learning_starts >= config.sac.steps:
        raise ValueError("sac.learning_starts must be smaller than sac.steps.")
    if config.sac.checkpoint_frequency != config.evaluation.frequency:
        raise ValueError(
            "sac.checkpoint_frequency must equal evaluation.frequency so every "
            "resumable boundary has a matching evaluation record."
        )
    if not np.isfinite(config.sac.learning_rate) or config.sac.learning_rate <= 0:
        raise ValueError("sac.learning_rate must be positive and finite.")
    if not 0 < config.sac.gamma <= 1 or not 0 < config.sac.tau <= 1:
        raise ValueError("sac.gamma and sac.tau must be in (0, 1].")
    if config.sac.ent_coef != "auto":
        raise ValueError("sac.ent_coef must remain auto.")
    if config.sac.net_arch != [256, 256, 256, 256]:
        raise ValueError("Forward transfer requires net_arch=[256, 256, 256, 256].")
    if config.evaluation.episodes <= 0 or config.evaluation.episodes > 50:
        raise ValueError("evaluation.episodes must be in [1, 50].")
    if not 0 <= config.evaluation.success_threshold <= 1:
        raise ValueError("evaluation.success_threshold must be in [0, 1].")
    if config.runtime.seed not in range(2**32):
        raise ValueError("runtime.seed must be in [0, 2**32).")


def resolved_run_directory(config: ForwardTransferConfig) -> Path:
    return (
        Path(config.runtime.output)
        / config.target_task
        / config.condition
        / f"seed_{config.runtime.seed}"
    ).resolve()


def load_temporally_valid_general_policy(
    config: ForwardTransferConfig,
) -> tuple[GeneralPolicy, dict[str, object]]:
    """Require a checkpoint trained through exactly the declared prior task."""
    path = Path(config.source_checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GeneralPolicy checkpoint does not exist: {path}")
    policy, _, metadata = GeneralPolicy.load_training(
        path,
        expected_use_layer_norm=True,
        expected_hidden_sizes=tuple(config.sac.net_arch),
    )
    expected_stage = len(config.task_history) - 1
    valid = (
        isinstance(metadata, dict)
        and metadata.get("stage") == expected_stage
        and metadata.get("task") == config.task_history[-1]
        and metadata.get("tasks", [])[: len(config.task_history)]
        == config.task_history
        and len(metadata.get("tasks", [])) > expected_stage + 1
        and metadata["tasks"][expected_stage + 1] == config.target_task
    )
    if not valid:
        raise ValueError(
            "GeneralPolicy checkpoint history does not establish that it was "
            "trained only through the declared task_history immediately before "
            "the target task."
        )
    identity = {
        "path": str(path),
        "sha256": file_hash(path),
        "stage": metadata["stage"],
        "task": metadata["task"],
        "task_history": list(config.task_history),
        "target_task": config.target_task,
    }
    return policy, identity


def validate_temporal_checkpoint(config: ForwardTransferConfig) -> dict[str, object]:
    """Return validated source identity without exposing the loaded policy."""
    _, identity = load_temporally_valid_general_policy(config)
    return identity


def evaluation_step_grid(total_steps: int, frequency: int) -> list[int]:
    if total_steps <= 0 or frequency <= 0 or total_steps % frequency:
        raise ValueError("Evaluation grid requires a divisible positive budget.")
    return list(range(0, total_steps + 1, frequency))


def learning_curve_metrics(
    steps: list[int],
    success_rates: list[float],
    mean_returns: list[float],
    *,
    success_threshold: float,
) -> dict[str, object]:
    """Compute normalized trapezoidal AUC and measured threshold crossing."""
    x = np.asarray(steps, dtype=float)
    y = np.asarray(success_rates, dtype=float)
    returns = np.asarray(mean_returns, dtype=float)
    if (
        x.ndim != 1
        or len(x) < 2
        or len(y) != len(x)
        or len(returns) != len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(returns).all()
        or np.any(np.diff(x) <= 0)
        or np.any((y < 0) | (y > 1))
    ):
        raise ValueError("Invalid learning-curve arrays.")
    crossings = x[y >= success_threshold]
    return {
        "normalized_success_auc": float(
            np.trapezoid(y, x) / (x[-1] - x[0])
        ),
        "final_success_rate": float(y[-1]),
        "final_mean_return": float(returns[-1]),
        "success_threshold": success_threshold,
        "threshold_step": int(crossings[0]) if len(crossings) else None,
    }


def compare_condition_summaries(random: dict, transferred: dict) -> dict[str, object]:
    if random["evaluation_steps"] != transferred["evaluation_steps"]:
        raise ValueError("Paired conditions use different evaluation grids.")
    if random["training_seed"] != transferred["training_seed"]:
        raise ValueError("Paired conditions use different training seeds.")
    return {
        "training_seed": random["training_seed"],
        "delta_normalized_success_auc": (
            transferred["metrics"]["normalized_success_auc"]
            - random["metrics"]["normalized_success_auc"]
        ),
        "delta_final_success_rate": (
            transferred["metrics"]["final_success_rate"]
            - random["metrics"]["final_success_rate"]
        ),
        "random_threshold_step": random["metrics"]["threshold_step"],
        "transferred_threshold_step": transferred["metrics"]["threshold_step"],
    }


def save_resolved_forward_transfer_config(
    config: ForwardTransferConfig, path: Path
) -> None:
    with Path(path).open("x", encoding="utf-8") as output:
        yaml.safe_dump(asdict(config), output, sort_keys=False)


def load_summary(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_sample_sd(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("At least two values are required for sample deviation.")
    return {
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }
