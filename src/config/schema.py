"""Typed configuration schemas for the supported research workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


SAC_ENT_COEF = "auto"
SAC_NET_ARCH = (256, 256, 256, 256)
SAC_TARGET_OUTPUT_STD = 0.089
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
    backend: str = "metaworld-v3"
    reward_function_version: str = "v2"


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0
    device: str = "cpu"
    output: str | None = None
    run_name: str | None = None
    run_dir: str | None = None


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    min_steps: int = 100_000
    success_threshold: float = 0.95
    patience: int = 3
    plateau_patience: int | None = None
    min_delta: float = 0.01


@dataclass(frozen=True)
class SACConfig:
    steps: int = 10_000
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 1_000
    random_exploration_steps: int = 10_000
    target_output_std: float = SAC_TARGET_OUTPUT_STD
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 50
    gradient_steps: int = 50
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True)
class ActorInitConfig:
    """Optional actor-only initialization for a standalone SAC run."""

    mode: str = "scratch"
    checkpoint: str | None = None


@dataclass(frozen=True)
class SACResumeConfig:
    """Optional full-model continuation, distinct from actor initialization."""

    enabled: bool = False
    checkpoint: str | None = None
    replay_buffer: str | None = None
    refill_steps: int = 0


@dataclass(frozen=True)
class BCConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    loss: str = "deterministic_mse"
    task_conditioning: str = "none"
    initial_log_std: float = -3.0
    activation: str = "relu"
    hidden_dim: int = 256
    dual_checkpoint_selection: bool = False
    checkpoint_selection: str = "validation_nll"


@dataclass(frozen=True)
class DiffusionConfig:
    epochs: int = 300
    batch_size: int = 8
    learning_rate: float = 0.001
    steps: int = 100
    backbone: str = "temporal_cnn"
    width: int = 64
    action_space: str = "raw"
    action_clamp_epsilon: float = 1e-4
    generated_action_projection: str = "none"


@dataclass(frozen=True)
class FlowMatchingConfig:
    epochs: int = 300
    batch_size: int = 8
    learning_rate: float = 0.001
    backbone: str = "temporal_cnn"
    width: int = 64
    time_embedding_scale: float = 100.0
    integration_steps: int = 100
    ode_solver: str = "euler"
    rtol: float = 1e-5
    atol: float = 1e-5


@dataclass(frozen=True)
class DiffusionModelConfig:
    """Serialized generator dimensions, separate from experiment settings."""

    horizon: int = 200
    features: int = 43
    steps: int = 100
    backbone: str = "temporal_cnn"
    backbone_spec_version: int = 1
    width: int = 64
    num_tasks: int = 0
    diffusion_action_space: str = "raw"
    diffusion_action_clamp_epsilon: float = 1e-4


def _default_tasks() -> list[str]:
    return ["reach-v3", "button-press-v3", "hammer-v3"]


def _default_experts() -> dict[str, str]:
    return {
        "reach-v3": "runs/experts/metaworld-v3/reach-v3/policy.zip",
        "button-press-v3": "runs/experts/metaworld-v3/button-press-v3/policy.zip",
        "hammer-v3": "runs/experts/metaworld-v3/hammer-v3/policy.zip",
    }


@dataclass(frozen=True)
class SACTaskConfig:
    tasks: list[str] = field(default_factory=lambda: ["reach-v3"])


@dataclass(frozen=True)
class DiffCRLTaskConfig:
    tasks: list[str] = field(default_factory=_default_tasks)
    experts: dict[str, str] = field(default_factory=_default_experts)
    trajectories_per_task: int = 200
    current_trajectory_schedule: list[int] = field(default_factory=list)
    replay_mode: str = "diffusion"
    replay_allocation: str = "generator-default"
    replay_budget: int = 200
    replay_task_weights: list[float] = field(default_factory=list)
    historical_replay_training_only: bool = False
    collection_mode: str = "all"
    collection_max_attempt_multiplier: int = 2
    expert_action_mode: str = "deterministic"


@dataclass(frozen=True)
class ContinualEvaluationConfig:
    episodes: int = 50
    seed: int = 10_000
    mode: str = "fixed-tasks"
    task_set_seed: int = 10_000


@dataclass(frozen=True)
class SACEvaluationConfig:
    episodes: int = 5
    final_episodes: int | None = None
    frequency: int = 5_000
    seed_offset: int = 10_000
    mode: str = "sampled"
    task_set_seed: int = 10_000
    final_task_set_seeds: list[int] = field(default_factory=lambda: [20_000, 20_001])
    final_reset_seed: int = 30_000


@dataclass(frozen=True)
class SACExperimentConfig:
    experiment: str = "sac"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    actor_init: ActorInitConfig = field(default_factory=ActorInitConfig)
    resume: SACResumeConfig = field(default_factory=SACResumeConfig)
    continual: SACTaskConfig = field(default_factory=SACTaskConfig)
    evaluation: SACEvaluationConfig = field(default_factory=SACEvaluationConfig)
    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(device="auto", output="runs")
    )


@dataclass(frozen=True)
class DiffCRLExperimentConfig:
    experiment: str = "diffcrl"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    continual: DiffCRLTaskConfig = field(default_factory=DiffCRLTaskConfig)
    evaluation: ContinualEvaluationConfig = field(default_factory=ContinualEvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


ExperimentConfig = SACExperimentConfig | DiffCRLExperimentConfig


def default_config(experiment: str) -> ExperimentConfig:
    if experiment == "sac":
        return SACExperimentConfig()
    if experiment == "diffcrl":
        return DiffCRLExperimentConfig()
    raise ValueError("experiment must be diffcrl or sac.")
