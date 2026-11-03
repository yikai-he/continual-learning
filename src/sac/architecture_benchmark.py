"""Configuration and read-only summaries for the four-layer SAC benchmark."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml


TASKS = [
    "reach-v3", "push-v3", "hammer-v3", "handle-press-side-v3",
    "button-press-topdown-v3", "faucet-open-v3", "drawer-close-v3",
    "window-close-v3", "door-open-v3", "peg-unplug-side-v3",
]
THREE_TASK_1M_TASKS = ["reach-v3", "hammer-v3", "handle-press-side-v3"]
FOUR_TASK_1M_TASKS = [
    "button-press-topdown-v3",
    "faucet-open-v3",
    "drawer-close-v3",
    "window-close-v3",
]
LOW_DISK_EXPERIMENTS = {
    "sac_4layer_3task_1m",
    "sac_4layer_4task_1m",
    "sac_kuka_5task_1m",
    "sac_kuka_remaining_4task_1m",
}
KUKA_FIVE_TASK_1M_TASKS = [
    "reach-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "button-press-topdown-v3",
    "faucet-open-v3",
]
KUKA_REMAINING_FOUR_TASK_1M_TASKS = [
    "drawer-close-v3",
    "window-close-v3",
    "door-open-v3",
    "peg-unplug-side-v3",
]


@dataclass(frozen=True)
class BenchmarkTraining:
    max_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    device: str = "cpu"
    torch_num_threads: int = 1


@dataclass(frozen=True)
class BenchmarkPolicy:
    type: str = "TransferableSACPolicy"
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256, 256])
    log_std_init: float = -3.0


@dataclass(frozen=True)
class BenchmarkEvaluation:
    frequency: int = 20_000
    episodes: int = 50
    validation_task_set_seed: int = 10_000
    validation_reset_seed: int = 10_000
    final_task_set_seeds: list[int] = field(default_factory=lambda: [20_000, 20_001])
    final_reset_seed: int = 30_000


@dataclass(frozen=True)
class BenchmarkEnvironment:
    env_backend: str = "metaworld-v3"
    hammer_reward_variant: str = "original"
    hammer_nail_progress_weight: float = 0.0
    peg_unplug_reward_variant: str = "original"
    peg_unplug_progress_weight: float = 0.0
    peg_unplug_progress_gamma: float = 0.99


@dataclass(frozen=True)
class BenchmarkEarlyStopping:
    enabled: bool = True
    threshold: float = 0.95
    consecutive_evaluations: int = 3
    min_timesteps: int = 100_000


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment: str = "sac_4layer_benchmark"
    algorithm: str = "SAC"
    tasks: list[str] = field(default_factory=lambda: list(TASKS))
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    reward_function_version: str = "v2"
    environment: BenchmarkEnvironment = field(default_factory=BenchmarkEnvironment)
    policy: BenchmarkPolicy = field(default_factory=BenchmarkPolicy)
    training: BenchmarkTraining = field(default_factory=BenchmarkTraining)
    evaluation: BenchmarkEvaluation = field(default_factory=BenchmarkEvaluation)
    early_stopping: BenchmarkEarlyStopping = field(default_factory=BenchmarkEarlyStopping)
    checkpoint_frequency: int = 50_000
    output: str = "runs/sac_4layer_benchmark"


def _section(value, cls, name):
    unknown = set(value) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    return cls(**value)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a YAML mapping.")
    unknown = set(data) - set(BenchmarkConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown benchmark fields: {sorted(unknown)}")
    data = dict(data)
    data["environment"] = _section(
        data.get("environment", {}), BenchmarkEnvironment, "environment"
    )
    data["policy"] = _section(data.get("policy", {}), BenchmarkPolicy, "policy")
    data["training"] = _section(data.get("training", {}), BenchmarkTraining, "training")
    data["evaluation"] = _section(data.get("evaluation", {}), BenchmarkEvaluation, "evaluation")
    data["early_stopping"] = _section(
        data.get("early_stopping", {}), BenchmarkEarlyStopping, "early_stopping"
    )
    config = BenchmarkConfig(**data)
    validate_benchmark_config(config)
    return config


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    profiles = {
        "sac_4layer_benchmark": (
            TASKS,
            True,
            "runs/sac_4layer_benchmark",
            "metaworld-v3",
        ),
        "sac_4layer_3task_1m": (
            THREE_TASK_1M_TASKS,
            False,
            "runs/sac_4layer_3task_1m",
            "metaworld-v3",
        ),
        "sac_4layer_4task_1m": (
            FOUR_TASK_1M_TASKS,
            False,
            "runs/sac_4layer_4task_1m",
            "metaworld-v3",
        ),
        "sac_kuka_5task_1m": (
            KUKA_FIVE_TASK_1M_TASKS,
            False,
            "runs/sac_kuka_5task_1m",
            "kuka-v3",
        ),
        "sac_kuka_remaining_4task_1m": (
            KUKA_REMAINING_FOUR_TASK_1M_TASKS,
            False,
            "runs/sac_kuka_remaining_4task_1m",
            "kuka-v3",
        ),
    }
    if config.experiment not in profiles:
        raise ValueError("Unknown SAC experiment profile.")
    tasks, early_stopping_enabled, output, backend = profiles[config.experiment]
    if config.tasks != tasks:
        raise ValueError("Benchmark experiment or ordered task list is invalid.")
    if config.output != output:
        raise ValueError(f"{config.experiment} output must be {output}.")
    if config.environment.env_backend != backend:
        raise ValueError(f"{config.experiment} requires env_backend={backend!r}.")
    if (
        config.environment.hammer_reward_variant != "original"
        or config.environment.hammer_nail_progress_weight != 0.0
    ):
        raise ValueError("This benchmark requires the default original Hammer reward.")
    if (
        config.environment.peg_unplug_reward_variant != "original"
        or config.environment.peg_unplug_progress_weight != 0.0
        or config.environment.peg_unplug_progress_gamma != 0.99
    ):
        raise ValueError(
            "This benchmark requires the default original Peg Unplug reward."
        )
    if config.algorithm != "SAC":
        raise ValueError("The benchmark algorithm must be SAC.")
    if (
        config.policy.type != "TransferableSACPolicy"
        or config.policy.net_arch != [256, 256, 256, 256]
        or not np.isfinite(config.policy.log_std_init)
    ):
        raise ValueError(
            "The benchmark requires TransferableSACPolicy [256, 256, 256, 256]."
        )
    if len(set(config.seeds)) != len(config.seeds) or any(s not in range(2**32) for s in config.seeds):
        raise ValueError("Benchmark seeds must be unique uint32 values.")
    t, e, stop = config.training, config.evaluation, config.early_stopping
    expected_training = BenchmarkTraining()
    if t != expected_training:
        raise ValueError(
            "SAC training settings must match the fixed 1M-step baseline."
        )
    if t.device != "cpu" or t.torch_num_threads != 1:
        raise ValueError("Benchmark requires CPU with one PyTorch thread.")
    if e != BenchmarkEvaluation():
        raise ValueError(
            "Evaluation settings must match the fixed validation and test banks."
        )
    if stop.threshold != 0.95 or stop.consecutive_evaluations != 3 or stop.min_timesteps != 100_000:
        raise ValueError("Early stopping requires 95% three times after 100,000 steps.")
    if stop.enabled is not early_stopping_enabled:
        raise ValueError(
            f"{config.experiment} requires early_stopping.enabled="
            f"{str(early_stopping_enabled).lower()}."
        )
    if config.checkpoint_frequency != 50_000:
        raise ValueError("Checkpoints must be updated every 50,000 steps.")


def validate_installed_tasks(tasks=TASKS, *, backend="metaworld-v3") -> None:
    """Validate registry support without constructing an environment."""
    if backend == "kuka-v3":
        from metaworld_kuka.registration import KUKA_ENVIRONMENTS

        tasks = [
            f"kuka-{task}" if not task.startswith("kuka-") else task
            for task in tasks
        ]
        missing = sorted(set(tasks) - set(KUKA_ENVIRONMENTS))
        if missing:
            raise ValueError(
                f"Installed KUKA MetaWorld does not support tasks: {missing}"
            )
        return
    if backend != "metaworld-v3":
        raise ValueError(f"Unsupported benchmark backend: {backend!r}.")
    import metaworld

    missing = sorted(set(tasks) - set(metaworld.MT1.ENV_NAMES))
    if missing:
        raise ValueError(f"Installed MetaWorld does not support tasks: {missing}")


def environment_task_id(config: BenchmarkConfig, task: str) -> str:
    """Resolve a profile label to the backend's actual registered task ID."""
    if task not in config.tasks:
        raise ValueError("Task is outside the declared benchmark matrix.")
    return f"kuka-{task}" if config.environment.env_backend == "kuka-v3" else task


def run_directory(config: BenchmarkConfig, task: str, seed: int) -> Path:
    if task not in config.tasks or seed not in config.seeds:
        raise ValueError("Task or seed is outside the declared benchmark matrix.")
    return (Path(config.output) / task / f"seed_{seed}").resolve()


def success_auc(steps, success_rates) -> float:
    x, y = np.asarray(steps, float), np.asarray(success_rates, float)
    if len(x) < 2 or len(x) != len(y) or np.any(np.diff(x) <= 0):
        raise ValueError("Invalid success learning curve.")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def stopping_reason(early_stopped: bool, actual_steps: int, maximum_steps: int) -> str:
    if early_stopped:
        return "success_criterion"
    if actual_steps >= maximum_steps:
        return "max_timesteps"
    raise ValueError("Training ended before its budget without early stopping.")


def aggregate_results(root: Path, tasks: list[str], seeds: list[int]):
    rows = []
    for task in tasks:
        for seed in seeds:
            path = root / task / f"seed_{seed}"
            if not (path / "RUN_COMPLETE").is_file():
                raise ValueError(f"Run is incomplete: {path}")
            value = json.loads((path / "result.json").read_text(encoding="utf-8"))
            rows.append({
                "task": task,
                "seed": seed,
                "actual_training_steps": value["actual_training_steps"],
                "stopping_reason": value["stopping_reason"],
                "best_validation_success": value["best_validation_success"],
                "final_test_success": value["final_test"]["success_rate"],
                "best_checkpoint": value["best_success_checkpoint"],
                "training_duration_seconds": value["training_duration_seconds"],
                "success_auc": value["success_auc"],
            })
    return rows


def write_aggregate(rows, json_path: Path, csv_path: Path) -> None:
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Aggregate output already exists.")
    if not rows:
        raise ValueError("Cannot write an empty benchmark aggregate.")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as output:
        json.dump({"kind": "sac_4layer_benchmark_aggregate_v1", "runs": rows}, output, indent=2)
        output.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
