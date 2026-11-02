"""Pretrained-expert DiffCRL with goal-conditioned replay and sequential BC."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from src.config import (
    ExperimentConfig,
    output_directory,
    resolve_device,
    save_resolved_config,
    validate_config,
)
from src.continual.stage_dataset import (
    bc_dataset,
    finalize_stage_groups,
    prepare_stage_group_split,
    validate_stage_group,
)
from src.envs import (
    ACTION_SHAPE,
    OBSERVATION_SHAPE,
    EnvironmentSettings,
    horizon_for_backend,
    make_env,
)
from src.support.expert_manifest import validate_expert_checkpoint
from src.support.io import write_json
from src.support.reproducibility import file_hash, state_hash
from src.support.run_manifest import complete_run, create_run_manifest
from stable_baselines3 import SAC
from tqdm.auto import tqdm

from .bc_policy import BC_CLAMP_EPSILON, GeneralPolicy, fit_bc
from .collector import collect_trajectories
from .diffcrl_replay import ReplayRequest, generate_previous_task_replay
from .diffusion_data import (
    PHYSICAL_ACTION_SLICE,
    normalize_generator_data,
    pack_trajectories,
    prepare_generator_data,
)
from .evaluation import EvaluationMatrix, evaluate_stage
from .expert_space import validate_expert_spaces
from .flow_matching_replay import (
    FlowMatchingReplayRequest,
    generate_previous_task_flow_matching_replay,
)
from .diffcrl_artifacts import (
    build_initial_run_summary,
    stage_checkpoint_paths,
    validate_finite_model_state,
)
from .policy_adapter import SB3SACPolicy
from .real_replay import RealReplayRequest, load_previous_task_real_replay
from .replay_allocation import replay_counts
from .reporting import (
    build_diffcrl_stage_report,
    print_final_report,
    print_stage_report,
)
from .resume import (
    completed_stage_markers,
    load_boundary_state,
    restore_rng_state,
    resolve_stage_attempt_paths,
    validate_resume_configuration,
    write_stage_boundary,
)
from .task_bank import (
    selected_configuration,
    training_bank,
    verify_disjoint_task_banks,
)
from .task_sequence import TaskSequence, TaskSpec
from .generator_normalization import FeatureNormalizer
from .trajectory_diffusion import (
    DiffusionConfig,
    TrajectoryDiffusion,
    fit_diffusion,
)
from .trajectory_flow_matching import (
    FlowMatchingConfig as FlowMatchingModelConfig,
)
from .trajectory_flow_matching import (
    TrajectoryFlowMatching,
    fit_flow_matching,
)

Metadata = dict[str, object]
LossReport = dict[str, object]


@dataclass(frozen=True)
class StageContext:
    """Immutable identity, paths, and pre-update hashes for one stage."""

    index: int
    task: TaskSpec
    trajectory_count: int
    directory: Path
    policy_before_hash: str | None
    diffusion_before_hash: str | None
    flow_matching_before_hash: str | None = None
    artifact_suffix: str = ""


@dataclass
class StageRawData:
    """Per-task tensors and provenance accumulated before stage splitting."""

    groups: dict[int, torch.Tensor]
    sources: dict[str, Metadata]


@dataclass
class CurrentTaskData:
    """Packed current-task data and provenance produced by expert collection."""

    group: torch.Tensor
    source: Metadata
    collected_configurations: list[Metadata]
    real_encode_clip_fraction: float


@dataclass
class StageData:
    """Prepared train/validation tensors and their stage provenance."""

    train: torch.Tensor
    validation: torch.Tensor
    train_labels: torch.Tensor
    validation_labels: torch.Tensor
    sources: dict[str, Metadata]
    splits: dict[str, Metadata]
    real_encode_clip_fraction: float


@dataclass
class DiffusionTrainingResult:
    """Normalization state and losses produced by the diffusion update."""

    normalizer: FeatureNormalizer
    goal_normalizer: FeatureNormalizer | None
    losses: LossReport | None
    training_seconds: float = 0.0


@dataclass
class PolicyTrainingResult:
    """Losses produced by the persistent GeneralPolicy update."""

    losses: LossReport


@dataclass(frozen=True)
class StageArtifacts:
    """Immutable checkpoint locations produced for a completed stage update."""

    diffusion_path: Path | None
    policy_path: Path
    flow_matching_path: Path | None = None


_build_stage_report = build_diffcrl_stage_report


class DiffCRLTrainer:
    """Coordinate sequential expert collection, diffusion replay, BC, and evaluation.

    Frozen SAC experts supply current-task data; diffusion represents previous-
    task memory; the persistent GeneralPolicy is the only continually updated
    decision policy and is executed during continual evaluation. Stage datasets
    and optimizers are fresh, while learned model weights persist.

    Generated data stays as physical tensors and is never presented as
    simulator-generated ``Trajectory`` objects.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        progress=True,
        resume_from=None,
        stage_complete_callback=None,
    ):
        """Validate configuration, initialize persistent state, and reserve output.

        Expert checkpoint hashes are captured up front and checked after every
        stage. New runs require a fresh output directory; explicit resumes use
        only hash-validated completed-stage markers.
        """
        validate_config(config)
        if config.experiment != "diffcrl":
            raise ValueError("DiffCRLTrainer requires experiment: diffcrl.")
        config = resolve_device(config)
        sequence = TaskSequence.from_names(config.continual.tasks)
        experts = config.continual.experts
        output = output_directory(config)
        self.sequence, self.config = sequence, config
        self.horizon = horizon_for_backend(config.environment.backend)
        self.environment_settings = EnvironmentSettings.from_config(config.environment)
        self.progress = progress
        self.experts = {
            t.task_name: Path(experts[t.task_name]).resolve() for t in sequence.tasks
        }
        self.expert_hashes = {
            name: file_hash(path) for name, path in self.experts.items()
        }
        self.expert_identities = {
            name: validate_expert_checkpoint(
                path,
                expected_task=name,
                expected_backend=config.environment.backend,
                expected_reward_function_version=config.environment.reward_function_version,
                expected_observation_shape=OBSERVATION_SHAPE,
                expected_action_shape=ACTION_SHAPE,
                expected_horizon=self.horizon,
                expected_hammer_reward_variant=(
                    config.environment.hammer_reward_variant
                    if name == "kuka-hammer-v3"
                    else None
                ),
                expected_hammer_nail_progress_weight=(
                    config.environment.hammer_nail_progress_weight
                    if name == "kuka-hammer-v3"
                    else None
                ),
                allow_legacy=config.environment.backend != "kuka-v3",
            )
            for name, path in self.experts.items()
        }
        self.task_bank_checks = (
            {
                task.task_name: verify_disjoint_task_banks(
                    task.task_name,
                    config.runtime.seed,
                    config.evaluation.task_set_seed,
                    backend=config.environment.backend,
                )
                for task in sequence.tasks
            }
            if config.environment.backend in ("metaworld-v3", "kuka-v3")
            and config.evaluation.mode == "fixed-tasks"
            else {}
        )
        self.output = (
            Path(resume_from).resolve()
            if resume_from is not None
            else Path(output).resolve()
        )
        if resume_from is None and self.output.exists():
            raise FileExistsError(self.output)
        if resume_from is not None and not self.output.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {self.output}")
        torch.set_num_threads(1)
        torch.manual_seed(config.runtime.seed)
        self.diffusion = (
            TrajectoryDiffusion(
                DiffusionConfig(
                    horizon=self.horizon,
                    steps=config.diffusion.steps,
                    width=config.diffusion.width,
                    num_tasks=sequence.num_tasks,
                    diffusion_action_space=config.diffusion.action_space,
                    diffusion_action_clamp_epsilon=config.diffusion.action_clamp_epsilon,
                )
            )
            if config.continual.replay_mode == "diffusion"
            else None
        )
        self.flow_matching = (
            TrajectoryFlowMatching(
                FlowMatchingModelConfig(
                    horizon=self.horizon,
                    width=config.flow_matching.width,
                    num_tasks=sequence.num_tasks,
                    integration_steps=config.flow_matching.integration_steps,
                    time_embedding_scale=config.flow_matching.time_embedding_scale,
                )
            )
            if config.continual.replay_mode == "flow-matching"
            else None
        )
        if self.diffusion is not None:
            self.diffusion.denoiser.to(config.runtime.device)
        if self.flow_matching is not None:
            self.flow_matching.velocity_model.to(config.runtime.device)
        if self.diffusion is not None and self.flow_matching is not None:
            raise RuntimeError("Replay generator selection must be mutually exclusive.")
        self.policy = None
        self.policy_optimizer = None
        self.matrix = EvaluationMatrix(sequence)
        self.previous_diffusion = None
        self.previous_flow_matching = None
        self.flow_dynamic_normalizer = None
        self.flow_goal_normalizer = None
        self.training_banks = {}
        self.train_configurations = {}
        self.next_stage = 0
        self.stage_complete_callback = stage_complete_callback
        if resume_from is None:
            self.output.mkdir(parents=True, exist_ok=False)
            save_resolved_config(config, self.output)
            create_run_manifest(
                self.output,
                config,
                experts={
                    name: identity.to_dict()
                    for name, identity in self.expert_identities.items()
                },
                task_banks=self.task_bank_checks,
            )
            write_json(
                self.output / "config.json",
                build_initial_run_summary(
                    config,
                    sequence,
                    self.horizon,
                    self.diffusion,
                    self.flow_matching,
                    self.experts,
                    self.expert_hashes,
                ),
            )
        else:
            self._restore_completed_run_state()

    def _restore_completed_run_state(self):
        """Restore the latest hash-validated, completed stage boundary."""
        validate_resume_configuration(self.output, self.config, self.expert_hashes)
        markers = completed_stage_markers(self.output, self.sequence.task_names)
        if not markers:
            raise ValueError(
                "Resume directory has no stage completion markers; start a new run."
            )
        if (
            len(markers) < self.sequence.num_tasks
            and not (self.output / "RUNNING").is_file()
        ):
            raise ValueError("Incomplete resume run is missing its RUNNING marker.")
        marker = markers[-1]
        artifacts = marker["artifacts"]
        self.policy, self.policy_optimizer, policy_metadata = (
            GeneralPolicy.load_training(
                artifacts["policy"]["resolved_path"],
                expected_use_layer_norm=self.config.bc.use_layer_norm,
            )
        )
        if (
            not isinstance(policy_metadata, dict)
            or policy_metadata.get("stage") != marker["stage"]
        ):
            raise ValueError("Policy checkpoint stage differs from completion marker.")
        self.policy.to(self.config.runtime.device)
        mode = self.config.continual.replay_mode
        has_diffusion = "diffusion" in artifacts
        has_flow = "flow_matching" in artifacts
        if mode == "diffusion":
            if has_flow:
                raise ValueError(
                    "Diffusion resume marker unexpectedly contains Flow Matching."
                )
            if "diffusion" not in artifacts:
                raise ValueError("Diffusion resume marker has no diffusion checkpoint.")
            expected_diffusion_config = DiffusionConfig(
                horizon=self.horizon,
                steps=self.config.diffusion.steps,
                width=self.config.diffusion.width,
                num_tasks=self.sequence.num_tasks,
                diffusion_action_space=self.config.diffusion.action_space,
                diffusion_action_clamp_epsilon=(
                    self.config.diffusion.action_clamp_epsilon
                ),
            )
            self.diffusion, _, _, diffusion_metadata = TrajectoryDiffusion.load(
                artifacts["diffusion"]["resolved_path"],
                expected_config=expected_diffusion_config,
                expected_action_space=self.config.diffusion.action_space,
                expected_clamp_epsilon=self.config.diffusion.action_clamp_epsilon,
                expected_projection=self.config.diffusion.generated_action_projection,
                expected_task_order=self.sequence.task_names,
            )
            self.diffusion.denoiser.to(self.config.runtime.device)
            self.previous_diffusion = Path(artifacts["diffusion"]["resolved_path"])
            self.previous_flow_matching = None
        elif mode == "flow-matching":
            if has_diffusion:
                raise ValueError(
                    "Flow Matching resume marker unexpectedly contains diffusion."
                )
            if not has_flow:
                raise ValueError("Flow Matching resume marker has no Flow checkpoint.")
            expected_flow_config = FlowMatchingModelConfig(
                horizon=self.horizon,
                width=self.config.flow_matching.width,
                num_tasks=self.sequence.num_tasks,
                integration_steps=self.config.flow_matching.integration_steps,
                time_embedding_scale=self.config.flow_matching.time_embedding_scale,
            )
            (
                self.flow_matching,
                self.flow_dynamic_normalizer,
                self.flow_goal_normalizer,
                diffusion_metadata,
            ) = TrajectoryFlowMatching.load(
                artifacts["flow_matching"]["resolved_path"],
                expected_config=expected_flow_config,
                expected_projection=self.config.diffusion.generated_action_projection,
                expected_action_space=self.config.diffusion.action_space,
                expected_clamp_epsilon=self.config.diffusion.action_clamp_epsilon,
                expected_task_order=self.sequence.task_names,
                expected_stage=marker["stage"],
                expected_task=marker["task"],
            )
            self.flow_matching.velocity_model.to(self.config.runtime.device)
            self.previous_flow_matching = Path(
                artifacts["flow_matching"]["resolved_path"]
            )
            self.previous_diffusion = None
        else:
            if has_diffusion or has_flow:
                raise ValueError(
                    "No-Replay resume marker unexpectedly contains a generator."
                )
            diffusion_metadata = {}
            self.previous_diffusion = None
            self.previous_flow_matching = None
        self.matrix = EvaluationMatrix.load(
            artifacts["evaluation_matrix"]["resolved_path"]
        )
        if self.matrix.sequence.task_names != self.sequence.task_names:
            raise ValueError(
                "Evaluation matrix task sequence differs from configuration."
            )
        state = load_boundary_state(marker)
        self.training_banks = state["training_banks"]
        self.train_configurations = state["train_configurations"]
        if diffusion_metadata and (
            diffusion_metadata.get("training_banks") != self.training_banks
            or diffusion_metadata.get("train_configurations")
            != self.train_configurations
        ):
            raise ValueError("Generator replay metadata differs from boundary state.")
        self.next_stage = marker["stage"] + 1
        restore_rng_state(state["rng"])

    def _progress_message(self, heading, **details):
        if self.progress:
            tqdm.write(
                "\n".join(
                    (
                        f"{heading}...",
                        *(f"  {name}: {value}" for name, value in details.items()),
                    )
                )
            )

    def _evaluate_stage(self, stage):
        """Execute GeneralPolicy with the configured stage-evaluation protocol.

        Fixed-task mode reuses the configured task bank and reset seeds across
        stages. Sampled mode instead allows MetaWorld to sample on each reset.
        """
        self._progress_message(
            "Evaluating learned tasks",
            **{"Episodes/task": self.config.evaluation.episodes},
        )
        return evaluate_stage(
            self.policy,
            self.matrix,
            stage,
            episodes=self.config.evaluation.episodes,
            seed=self.config.evaluation.seed,
            evaluation_mode=self.config.evaluation.mode,
            task_set_seed=self.config.evaluation.task_set_seed,
            reward_function_version=self.config.environment.reward_function_version,
            hammer_reward_variant=self.config.environment.hammer_reward_variant,
            hammer_nail_progress_weight=self.config.environment.hammer_nail_progress_weight,
            peg_unplug_reward_variant=self.config.environment.peg_unplug_reward_variant,
            peg_unplug_progress_weight=self.config.environment.peg_unplug_progress_weight,
            peg_unplug_progress_gamma=self.config.environment.peg_unplug_progress_gamma,
            backend=self.config.environment.backend,
            progress=self.progress,
        )

    def _generate_replay(self, context: StageContext) -> StageRawData:
        """Generate old-task replay before accessing the current task."""
        if not context.index or self.config.continual.replay_mode == "none":
            return StageRawData(groups={}, sources={})
        cfg = self.config
        common = {
            "stage": context.index,
            "task_names": tuple(self.sequence.task_names),
            "backend": cfg.environment.backend,
            "horizon": self.horizon,
            "runtime_seed": cfg.runtime.seed,
            "device": cfg.runtime.device,
            "reward_function_version": cfg.environment.reward_function_version,
            "hammer_reward_variant": cfg.environment.hammer_reward_variant,
            "hammer_nail_progress_weight": cfg.environment.hammer_nail_progress_weight,
            "action_space": cfg.diffusion.action_space,
            "action_clamp_epsilon": cfg.diffusion.action_clamp_epsilon,
            "action_projection": cfg.diffusion.generated_action_projection,
            "bc_clamp_epsilon": BC_CLAMP_EPSILON,
            "progress": self.progress,
        }
        allocation = replay_counts(
            range(context.index),
            generator=(
                "diffusion"
                if cfg.continual.replay_mode == "real"
                else cfg.continual.replay_mode
            ),
            allocation=cfg.continual.replay_allocation,
            trajectories_per_task=context.trajectory_count,
            total_budget=cfg.continual.replay_budget,
            strategy=cfg.continual.replay_priority_strategy,
        )
        if cfg.continual.replay_mode == "real":
            groups, sources = load_previous_task_real_replay(
                RealReplayRequest(
                    stage=context.index,
                    task_names=tuple(self.sequence.task_names),
                    trajectory_counts=allocation,
                    run_directory=self.output,
                    runtime_seed=cfg.runtime.seed,
                )
            )
            return StageRawData(groups=groups, sources=sources)
        if self.diffusion is not None:
            replay = generate_previous_task_replay(
                ReplayRequest(
                    trajectory_count=context.trajectory_count,
                    trajectory_counts=allocation,
                    checkpoint=self.previous_diffusion,
                    expected_state_hash=context.diffusion_before_hash,
                    **common,
                ),
                progress_message=self._progress_message,
            )
        elif self.flow_matching is not None:
            replay = generate_previous_task_flow_matching_replay(
                FlowMatchingReplayRequest(
                    trajectory_counts=allocation,
                    checkpoint=self.previous_flow_matching,
                    expected_state_hash=context.flow_matching_before_hash,
                    integration_steps=cfg.flow_matching.integration_steps,
                    **common,
                ),
                progress_message=self._progress_message,
            )
        else:
            return StageRawData(groups={}, sources={})
        return StageRawData(groups=replay.groups, sources=replay.sources)

    def _collect_current_task(self, context: StageContext) -> CurrentTaskData:
        """Collect deterministic current-task episodes from its frozen SAC expert.

        The method records task configurations and provenance, initializes the
        GeneralPolicy at stage zero, and returns packed ``(N, horizon, 43)`` data.
        All episodes are retained regardless of success, but formal packing
        requires the selected backend's full horizon and rejects shorter episodes.
        """
        cfg = self.config
        task = context.task
        count = context.trajectory_count
        directory = context.directory
        self._progress_message(
            "Collecting expert trajectories", Task=task.task_name, Trajectories=count
        )
        expert_model = SAC.load(self.experts[task.task_name], device=cfg.runtime.device)
        settings = getattr(
            self,
            "environment_settings",
            EnvironmentSettings.from_config(cfg.environment),
        )
        env = make_env(
            settings.backend,
            task.task_name,
            cfg.runtime.seed,
            **settings.kwargs(),
        )
        try:
            validate_expert_spaces(
                expert_model, env, self.expert_identities[task.task_name]
            )
            expert = SB3SACPolicy(expert_model)
            bank = training_bank(
                task.task_name,
                cfg.runtime.seed,
                backend=cfg.environment.backend,
            )
            collected_configurations = []
            trajectories = collect_trajectories(
                env,
                expert,
                task.task_id,
                episodes=count,
                initial_seed=cfg.runtime.seed,
                deterministic=True,
                on_reset=lambda e, obs, seed, episode: collected_configurations.append(
                    selected_configuration(e, bank, obs, episode)
                ),
                progress=self.progress,
                description=f"Expert collection: {task.task_name}",
            )
            if self.policy is None:
                torch.manual_seed(cfg.runtime.seed)
                self.policy = GeneralPolicy(
                    env.action_space.low,
                    env.action_space.high,
                    hidden_sizes=tuple(cfg.bc.hidden_sizes),
                    use_layer_norm=cfg.bc.use_layer_norm,
                ).to(cfg.runtime.device)
        finally:
            env.close()
        del expert
        group = pack_trajectories(
            trajectories,
            horizon=self.horizon,
            task_names=self.sequence.task_names,
        )
        real_actions = group[..., PHYSICAL_ACTION_SLICE]
        if torch.any(real_actions < -1) or torch.any(real_actions > 1):
            raise ValueError("Real expert actions exceed environment bounds.")
        real_encode_clip_fraction = float(
            (
                (real_actions < -1 + cfg.diffusion.action_clamp_epsilon)
                | (real_actions > 1 - cfg.diffusion.action_clamp_epsilon)
            )
            .float()
            .mean()
        )
        np.savez_compressed(
            directory / "real_trajectories.npz",
            **{
                f"episode_{i}_{key}": value
                for i, t in enumerate(trajectories)
                for key, value in asdict(t).items()
            },
        )
        source = {
            "kind": "real_expert",
            "checkpoint": str(self.experts[task.task_name]),
            "checkpoint_sha256": self.expert_hashes[task.task_name],
            "seeds": list(range(cfg.runtime.seed, cfg.runtime.seed + count)),
            "success_count": sum(bool(t.success) for t in trajectories),
            "success_rate": float(np.mean([t.success for t in trajectories])),
            "episode_returns": [t.episode_return for t in trajectories],
            "mean_return": float(np.mean([t.episode_return for t in trajectories])),
            "std_return": float(np.std([t.episode_return for t in trajectories])),
            "real_diffusion_encode_clip_fraction": real_encode_clip_fraction,
        }
        source["collection_configurations"] = collected_configurations
        self.training_banks[task.task_name] = bank
        return CurrentTaskData(
            group=group,
            source=source,
            collected_configurations=collected_configurations,
            real_encode_clip_fraction=real_encode_clip_fraction,
        )

    def _prepare_stage_data(
        self,
        context: StageContext,
        raw_data: StageRawData,
        current_task: CurrentTaskData,
    ) -> StageData:
        """Persist stage tensors and create deterministic whole-episode splits.

        Real and generated task groups remain separate while splitting so no
        episode contributes timesteps to both training and validation. Returned
        labels condition diffusion by task index.
        """
        cfg = self.config
        stage = context.index
        count = context.trajectory_count
        directory = context.directory
        groups, sources = raw_data.groups, raw_data.sources
        collected_configurations = current_task.collected_configurations
        preparations = {}
        for index, values in groups.items():
            name = self.sequence.task(index).task_name
            group_count = validate_stage_group(
                values,
                sources[name],
                stage=stage,
                task=name,
                horizon=self.horizon,
            )
            path = directory / f"data_task_{index}.npy"
            np.save(path, values.numpy(), allow_pickle=False)
            sources[name].update(data_path=str(path), data_sha256=file_hash(path))
            prepared = prepare_stage_group_split(
                values,
                count=group_count,
                split_seed=cfg.runtime.seed + stage * 1000 + index,
                action_low=self.policy.action_low.cpu(),
                action_high=self.policy.action_high.cpu(),
                clamp_epsilon=BC_CLAMP_EPSILON,
                training_only=getattr(
                    getattr(cfg, "continual", None),
                    "historical_replay_training_only",
                    False,
                )
                and index < stage,
            )
            preparations[index] = prepared
            sources[name]["bc_target_clip_fraction"] = prepared.bc_target_clip_fraction
            if index == stage:
                self.train_configurations[name] = {
                    "indices": [
                        collected_configurations[i]["configuration_index"]
                        for i in prepared.train_ids
                    ],
                    "source_trajectory_ids": prepared.train_ids,
                }
        prepared_data, splits = finalize_stage_groups(
            groups,
            preparations,
            self.sequence.task_names,
            horizon=self.horizon,
        )
        return StageData(
            train=prepared_data.train,
            validation=prepared_data.validation,
            train_labels=prepared_data.train_labels,
            validation_labels=prepared_data.validation_labels,
            sources=sources,
            splits=splits,
            real_encode_clip_fraction=current_task.real_encode_clip_fraction,
        )

    def _fit_stage_diffusion(
        self, context: StageContext, data: StageData
    ) -> DiffusionTrainingResult:
        """Prepare stage normalization and fit the selected replay generator.

        Physical 43D data is encoded as 22D dynamics plus a separately
        normalized 3D goal. Normalizers use only the stage training split.
        Without replay, a 43D normalizer is still recorded for stage reporting.
        The GeneralPolicy must remain unchanged until its BC update begins.
        """
        cfg = self.config
        stage = context.index
        policy_before = context.policy_before_hash
        train, validation = data.train, data.validation
        labels, vlabels = data.train_labels, data.validation_labels
        goal_normalizer = None
        generator_enabled = self.diffusion is not None or self.flow_matching is not None
        # Old replay was decoded with OLD statistics; now all groups are physical.
        if generator_enabled:
            prepared_generator = prepare_generator_data(
                train,
                validation,
                action_space=cfg.diffusion.action_space,
                action_clamp_epsilon=cfg.diffusion.action_clamp_epsilon,
            )
            normalizer = prepared_generator.normalizer
            goal_normalizer = prepared_generator.goal_normalizer
        else:
            normalizer = FeatureNormalizer.fit(train)
        if policy_before is not None and state_hash(self.policy) != policy_before:
            raise RuntimeError("GeneralPolicy changed before its stage update.")
        diff_losses = None
        training_seconds = 0.0
        if self.diffusion is not None:
            self._progress_message("Training diffusion", Epochs=cfg.diffusion.epochs)
            training_started = time.perf_counter()
            normalized_generator = normalize_generator_data(prepared_generator)
            diff_losses = fit_diffusion(
                self.diffusion,
                normalized_generator.train,
                normalized_generator.validation,
                normalized_generator.train_goals,
                normalized_generator.validation_goals,
                epochs=cfg.diffusion.epochs,
                seed=cfg.runtime.seed + stage,
                batch_size=cfg.diffusion.batch_size,
                learning_rate=cfg.diffusion.learning_rate,
                train_task_ids=labels,
                validation_task_ids=vlabels,
                progress=self.progress,
            )
            training_seconds = time.perf_counter() - training_started
        elif self.flow_matching is not None:
            self._progress_message(
                "Training Flow Matching", Epochs=cfg.flow_matching.epochs
            )
            training_started = time.perf_counter()
            normalized_generator = normalize_generator_data(prepared_generator)
            diff_losses = fit_flow_matching(
                self.flow_matching,
                normalized_generator.train,
                normalized_generator.validation,
                normalized_generator.train_goals,
                normalized_generator.validation_goals,
                epochs=cfg.flow_matching.epochs,
                seed=cfg.runtime.seed + stage,
                batch_size=cfg.flow_matching.batch_size,
                learning_rate=cfg.flow_matching.learning_rate,
                train_task_ids=labels,
                validation_task_ids=vlabels,
                progress=self.progress,
            )
            training_seconds = time.perf_counter() - training_started
        return DiffusionTrainingResult(
            normalizer=normalizer,
            goal_normalizer=goal_normalizer,
            losses=diff_losses,
            training_seconds=training_seconds,
        )

    def _fit_stage_policy(
        self, context: StageContext, data: StageData
    ) -> PolicyTrainingResult:
        """Update the persistent GeneralPolicy on current data plus replay.

        A fresh Adam optimizer is created for the stage, while policy weights
        carry forward. Generated out-of-bounds targets remain unchanged for
        diagnostics. The tanh-bounded policy cannot reproduce those targets
        exactly, but every action it executes remains inside environment bounds.
        """
        cfg = self.config
        stage = context.index
        self._progress_message("Training GeneralPolicy BC", Epochs=cfg.bc.epochs)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.bc.learning_rate
        )
        bc_losses = fit_bc(
            self.policy,
            bc_dataset(data.train),
            bc_dataset(data.validation),
            epochs=cfg.bc.epochs,
            seed=cfg.runtime.seed + stage,
            batch_size=cfg.bc.batch_size,
            learning_rate=cfg.bc.learning_rate,
            allow_out_of_bounds_targets=True,
            optimizer=self.policy_optimizer,
            bc_loss=cfg.bc.loss,
            progress=self.progress,
        )
        return PolicyTrainingResult(losses=bc_losses)

    def _save_stage_checkpoints(
        self,
        context: StageContext,
        diffusion_training: DiffusionTrainingResult,
        policy_training: PolicyTrainingResult,
    ) -> StageArtifacts:
        """Validate finite model state and save immutable stage checkpoints.

        Diffusion checkpoints include normalization and conditioning metadata;
        policy checkpoints include the paired optimizer state selected at the
        best validation epoch.
        """
        cfg = self.config
        stage = context.index
        task = context.task
        normalizer = diffusion_training.normalizer
        goal_normalizer = diffusion_training.goal_normalizer
        bc_losses = policy_training.losses
        validate_finite_model_state(self.policy, self.diffusion, self.flow_matching)
        diffusion_path, policy_path, flow_matching_path = stage_checkpoint_paths(
            self.output,
            stage=stage,
            task=task.task_name,
            artifact_suffix=context.artifact_suffix,
            has_diffusion=self.diffusion is not None,
            has_flow_matching=self.flow_matching is not None,
        )
        if self.diffusion is not None:
            self.diffusion.save(
                diffusion_path,
                normalizer,
                goal_normalizer,
                {
                    "task_order": self.sequence.task_names,
                    "task_conditioning": "integer task embedding plus projected normalized 3D goal",
                    "training_banks": self.training_banks,
                    "train_configurations": self.train_configurations,
                    "generated_action_projection": cfg.diffusion.generated_action_projection,
                    "diffusion_action_space": cfg.diffusion.action_space,
                    "diffusion_action_clamp_epsilon": cfg.diffusion.action_clamp_epsilon,
                    "diffusion_fit": {
                        "epochs": cfg.diffusion.epochs,
                        "batch_size": cfg.diffusion.batch_size,
                        "learning_rate": cfg.diffusion.learning_rate,
                    },
                },
            )
        if self.flow_matching is not None:
            self.flow_matching.save(
                flow_matching_path,
                normalizer,
                goal_normalizer,
                {
                    "stage": stage,
                    "task": task.task_name,
                    "seed": cfg.runtime.seed,
                    "task_order": self.sequence.task_names,
                    "task_conditioning": "integer task embedding plus projected normalized 3D goal",
                    "training_banks": self.training_banks,
                    "train_configurations": self.train_configurations,
                    "generated_action_projection": cfg.diffusion.generated_action_projection,
                    "action_space": cfg.diffusion.action_space,
                    "action_clamp_epsilon": cfg.diffusion.action_clamp_epsilon,
                    "flow_matching_fit": asdict(cfg.flow_matching),
                },
            )
        self.policy.save(
            policy_path,
            optimizer=self.policy_optimizer,
            metadata={
                "stage": stage,
                "task": task.task_name,
                "tasks": self.sequence.task_names,
                "seed": cfg.runtime.seed,
                "best_epoch": bc_losses["best_epoch"],
                "bc_loss": cfg.bc.loss,
                "bc_clamp_epsilon": BC_CLAMP_EPSILON,
                "bc_normalization_mode": cfg.bc.normalization_mode,
                "bc_use_layer_norm": cfg.bc.use_layer_norm,
                "diffusion_action_space": cfg.diffusion.action_space,
                "diffusion_action_clamp_epsilon": cfg.diffusion.action_clamp_epsilon,
                "generated_action_projection": cfg.diffusion.generated_action_projection,
            },
        )
        return StageArtifacts(
            diffusion_path=diffusion_path,
            policy_path=policy_path,
            flow_matching_path=flow_matching_path,
        )

    def _finish_stage(
        self,
        context: StageContext,
        data: StageData,
        diffusion_training: DiffusionTrainingResult,
        policy_training: PolicyTrainingResult,
        artifacts: StageArtifacts,
    ) -> Metadata:
        """Finalize stage artifacts, replay state, and reporting.

        The method writes provenance and matrix snapshots, rechecks frozen
        expert hashes, selects the new replay checkpoint, advances the stage
        counter, writes the completion marker last, and prints the stage report.
        """
        cfg = self.config
        stage = context.index
        task = context.task
        count = context.trajectory_count
        directory = context.directory
        diffusion_path = artifacts.diffusion_path
        flow_matching_path = artifacts.flow_matching_path
        policy_path = artifacts.policy_path
        report = _build_stage_report(
            cfg,
            context,
            data,
            diffusion_training,
            policy_training,
            artifacts,
            policy_after_state_sha256=state_hash(self.policy),
            diffusion_sha256=file_hash(diffusion_path)
            if diffusion_path is not None
            else None,
            flow_matching_sha256=file_hash(flow_matching_path)
            if flow_matching_path is not None
            else None,
            policy_sha256=file_hash(policy_path),
        )
        self.matrix.stage_provenance[str(stage)] = {
            "report": str(directory / "report.json"),
            "bc_normalization_mode": cfg.bc.normalization_mode,
            "bc_use_layer_norm": cfg.bc.use_layer_norm,
            "bc_loss": cfg.bc.loss,
            "bc_clamp_epsilon": BC_CLAMP_EPSILON,
            "generated_action_projection": cfg.diffusion.generated_action_projection,
        }
        write_json(directory / "report.json", report)
        write_json(directory / "evaluation_matrix.json", self.matrix.to_dict())
        for name, path in self.experts.items():
            if file_hash(path) != self.expert_hashes[name]:
                raise RuntimeError("Source expert changed during experiment.")
        self.previous_diffusion = diffusion_path
        self.previous_flow_matching = flow_matching_path
        self.next_stage += 1
        write_stage_boundary(
            self.output,
            stage=stage,
            task=task.task_name,
            policy_path=policy_path,
            diffusion_path=diffusion_path,
            flow_matching_path=flow_matching_path,
            report_path=directory / "report.json",
            matrix_path=directory / "evaluation_matrix.json",
            training_banks=self.training_banks,
            train_configurations=self.train_configurations,
            artifact_suffix=context.artifact_suffix,
        )
        callback = getattr(self, "stage_complete_callback", None)
        if callback is not None:
            callback(self.output, stage, self.sequence.task_names)
        print_stage_report(
            self.matrix,
            stage,
            task_training_steps=None,
            global_timesteps=None,
            training_detail=f"Training trajectories/task: {count}",
        )
        return report

    def train_task(self, stage):
        """Execute one stage in strict sequence and return its provenance report.

        Replay is generated first, followed by expert collection, deterministic
        splitting, diffusion fitting, BC, checkpointing, and evaluation.
        """
        if stage != self.next_stage:
            raise ValueError("Stages must execute once, in sequence order.")
        task = self.sequence.task(stage)
        if self.progress:
            tqdm.write(
                f"\nStage {stage + 1}/{self.sequence.num_tasks}: {task.task_name}"
            )
        count = self.config.continual.trajectories_per_task
        directory, artifact_suffix = resolve_stage_attempt_paths(
            self.output, stage, task.task_name
        )
        directory.mkdir()
        policy_before = None if self.policy is None else state_hash(self.policy)
        diffusion_before = (
            state_hash(self.diffusion) if self.diffusion is not None else None
        )
        flow_matching_before = (
            state_hash(self.flow_matching) if self.flow_matching is not None else None
        )

        context = StageContext(
            index=stage,
            task=task,
            trajectory_count=count,
            directory=directory,
            policy_before_hash=policy_before,
            diffusion_before_hash=diffusion_before,
            flow_matching_before_hash=flow_matching_before,
            artifact_suffix=artifact_suffix,
        )
        replay = self._generate_replay(context)
        current_task = self._collect_current_task(context)
        raw_data = StageRawData(
            groups={**replay.groups, stage: current_task.group},
            sources={**replay.sources, task.task_name: current_task.source},
        )
        data = self._prepare_stage_data(context, raw_data, current_task)
        diffusion_training = self._fit_stage_diffusion(context, data)
        policy_training = self._fit_stage_policy(context, data)
        artifacts = self._save_stage_checkpoints(
            context, diffusion_training, policy_training
        )
        self._evaluate_stage(stage)
        return self._finish_stage(
            context,
            data,
            diffusion_training,
            policy_training,
            artifacts,
        )

    def run(self):
        """Train all configured tasks sequentially and return the final matrix."""
        for stage in range(self.next_stage, self.sequence.num_tasks):
            self.train_task(stage)
        final_matrix = self.output / "evaluation_matrix.json"
        if final_matrix.exists():
            existing = json.loads(final_matrix.read_text(encoding="utf-8"))
            if existing != self.matrix.to_dict():
                raise ValueError(
                    "Existing final evaluation matrix differs from resumed state."
                )
        else:
            write_json(final_matrix, self.matrix.to_dict())
        print_final_report(self.matrix)
        if not (self.output / "RUN_COMPLETE").exists():
            complete_run(self.output)
        return self.matrix
