"""Pretrained-expert DiffCRL with goal-conditioned replay and sequential BC."""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from tqdm.auto import tqdm

from src.config import (
    REPLAY_PRIORITY_STRATEGY,
    ExperimentConfig,
    output_directory,
    resolve_device,
    save_resolved_config,
    validate_config,
)
from src.continual.stage_dataset import (
    bc_dataset,
    prepare_stage_groups,
)
from continual_manipulation.environments import (
    ACTION_SHAPE,
    OBSERVATION_SHAPE,
    EnvironmentSettings,
    horizon_for_backend,
    make_env,
)
from src.support.expert_manifest import validate_expert_checkpoint
from src.support.io import atomic_replace_json, write_json
from src.support.reproducibility import file_hash, state_hash
from src.support.run_manifest import complete_run, create_run_manifest

from .bc_policy import (
    BC_CLAMP_EPSILON,
    GeneralPolicy,
    fit_bc,
)
from .collector import collect_trajectories
from .diffcrl_artifacts import (
    build_initial_run_summary,
    finish_stage,
    save_stage_checkpoints,
    synchronize_timing_device as _synchronize_timing_device,
)
from .diffcrl_replay import ReplayRequest, generate_previous_task_replay
from .diffusion_data import (
    GENERATOR_FEATURES,
    PHYSICAL_ACTION_SLICE,
    normalize_generator_data,
    pack_trajectories,
    prepare_generator_data,
)
from continual_manipulation.evaluation import EvaluationMatrix, evaluate_stage
from .expert_recollection import (
    ExpertRecollectionRequest,
    collect_historical_expert_replay,
)
from .expert_space import validate_expert_spaces
from .flow_matching_replay import (
    FlowMatchingReplayRequest,
    generate_previous_task_flow_matching_replay,
)
from .generator_normalization import FeatureNormalizer
from .policy_adapter import SB3SACPolicy
from .replay_allocation import replay_counts
from .reporting import print_final_report
from .task_bank import (
    selected_configuration,
    training_bank,
)
from continual_manipulation.task_banks import verify_disjoint_task_banks
from continual_manipulation.sequences import TaskSequence, TaskSpec
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


def _timed_call(device, function, *args, **kwargs):
    """Return ``(result, elapsed_seconds)`` without changing numerical behavior."""
    _synchronize_timing_device(device)
    started = time.perf_counter()
    result = function(*args, **kwargs)
    _synchronize_timing_device(device)
    return result, time.perf_counter() - started


@contextmanager
def _preserve_random_state():
    """Isolate optional diagnostic evaluation from subsequent training RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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


@dataclass
class StageRawData:
    """Per-task tensors and provenance accumulated before stage splitting."""

    groups: dict[int, torch.Tensor]
    sources: dict[str, Metadata]
    replay_allocation: dict[int, int] | None = None


@dataclass
class CurrentTaskData:
    """Packed current-task data and provenance produced by expert collection."""

    group: torch.Tensor
    source: Metadata
    collected_configurations: list[Metadata]


@dataclass
class StageData:
    """Prepared train/validation tensors and their stage provenance."""

    train: torch.Tensor
    validation: torch.Tensor
    train_labels: torch.Tensor
    validation_labels: torch.Tensor
    sources: dict[str, Metadata]
    splits: dict[str, Metadata]
    train_reference_goals: torch.Tensor | None = None
    validation_reference_goals: torch.Tensor | None = None
    replay_allocation: dict[int, int] | None = None


@dataclass
class DiffusionTrainingResult:
    """Normalization state and losses produced by the diffusion update."""

    normalizer: FeatureNormalizer
    goal_normalizer: FeatureNormalizer | None
    losses: LossReport | None
    training_seconds: float = 0.0


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
        stage_complete_callback=None,
    ):
        """Validate configuration, initialize persistent state, and reserve output.

        Expert checkpoint hashes are captured up front and checked after every
        stage. Every run requires a fresh output directory.
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
        self.expert_identities = {
            name: validate_expert_checkpoint(
                path,
                expected_task=name,
                expected_backend=config.environment.backend,
                expected_reward_function_version=config.environment.reward_function_version,
                expected_observation_shape=OBSERVATION_SHAPE,
                expected_action_shape=ACTION_SHAPE,
                expected_horizon=self.horizon,
                allow_legacy=config.environment.backend != "kuka-v3",
            )
            for name, path in self.experts.items()
        }
        self.expert_hashes = {
            name: identity.checkpoint_sha256
            for name, identity in self.expert_identities.items()
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
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError(self.output)
        torch.set_num_threads(1)
        torch.manual_seed(config.runtime.seed)
        self.diffusion = (
            TrajectoryDiffusion(
                DiffusionConfig(
                    horizon=self.horizon,
                    features=GENERATOR_FEATURES,
                    steps=config.diffusion.steps,
                    backbone=config.diffusion.backbone,
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
                    features=GENERATOR_FEATURES,
                    backbone=config.flow_matching.backbone,
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
        self.best_mse_matrix = (
            EvaluationMatrix(sequence)
            if config.bc.dual_checkpoint_selection
            else None
        )
        self.best_mse_policy_state = None
        self.previous_diffusion = None
        self.previous_flow_matching = None
        self.training_banks = {}
        self.train_configurations = {}
        self.next_stage = 0
        self.stage_complete_callback = stage_complete_callback
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
            backend=self.config.environment.backend,
            progress=self.progress,
        )

    def _evaluate_best_mse_stage(self, stage):
        """Evaluate an isolated best-MSE policy without advancing training state."""
        if self.best_mse_matrix is None or self.best_mse_policy_state is None:
            raise RuntimeError("Best-MSE evaluation was requested without a selected state.")
        alternative = deepcopy(self.policy)
        alternative.load_state_dict(self.best_mse_policy_state)
        alternative.eval()
        with _preserve_random_state():
            return evaluate_stage(
                alternative,
                self.best_mse_matrix,
                stage,
                episodes=self.config.evaluation.episodes,
                seed=self.config.evaluation.seed,
                evaluation_mode=self.config.evaluation.mode,
                task_set_seed=self.config.evaluation.task_set_seed,
                reward_function_version=self.config.environment.reward_function_version,
                backend=self.config.environment.backend,
                progress=self.progress,
            )

    def _generate_replay(self, context: StageContext) -> StageRawData:
        """Generate old-task replay before accessing the current task."""
        if not context.index or self.config.continual.replay_mode == "none":
            return StageRawData(groups={}, sources={}, replay_allocation={})
        cfg = self.config
        common = {
            "stage": context.index,
            "task_names": tuple(self.sequence.task_names),
            "backend": cfg.environment.backend,
            "horizon": self.horizon,
            "runtime_seed": cfg.runtime.seed,
            "device": cfg.runtime.device,
            "reward_function_version": cfg.environment.reward_function_version,
            "action_space": cfg.diffusion.action_space,
            "action_clamp_epsilon": cfg.diffusion.action_clamp_epsilon,
            "action_projection": cfg.diffusion.generated_action_projection,
            "progress": self.progress,
        }
        allocation = replay_counts(
            range(context.index),
            generator=(
                "diffusion"
                if cfg.continual.replay_mode == "expert-recollection"
                else cfg.continual.replay_mode
            ),
            allocation=cfg.continual.replay_allocation,
            trajectories_per_task=context.trajectory_count,
            total_budget=cfg.continual.replay_budget,
            strategy=REPLAY_PRIORITY_STRATEGY,
            task_weights=(
                cfg.continual.replay_task_weights[: context.index]
                if cfg.continual.replay_allocation == "fixed-weighted"
                else None
            ),
        )
        if cfg.continual.replay_mode == "expert-recollection":
            groups, sources = collect_historical_expert_replay(
                ExpertRecollectionRequest(
                    stage=context.index,
                    task_names=tuple(self.sequence.task_names),
                    trajectory_counts=allocation,
                    experts=self.experts,
                    expert_hashes=self.expert_hashes,
                    expert_identities=self.expert_identities,
                    environment=self.environment_settings,
                    runtime_seed=cfg.runtime.seed,
                    device=cfg.runtime.device,
                    horizon=self.horizon,
                    collection_mode=cfg.continual.collection_mode,
                    max_attempt_multiplier=(
                        cfg.continual.collection_max_attempt_multiplier
                    ),
                    progress=self.progress,
                )
            )
            return StageRawData(
                groups=groups,
                sources=sources,
                replay_allocation=allocation,
            )
        if self.diffusion is not None:
            replay = generate_previous_task_replay(
                ReplayRequest(
                    trajectory_count=context.trajectory_count,
                    trajectory_counts=allocation,
                    checkpoint=self.previous_diffusion,
                    expected_state_hash=context.diffusion_before_hash,
                    expected_config=self.diffusion.config,
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
                    expected_config=self.flow_matching.config,
                    integration_steps=cfg.flow_matching.integration_steps,
                    solver=cfg.flow_matching.ode_solver,
                    rtol=cfg.flow_matching.rtol,
                    atol=cfg.flow_matching.atol,
                    **common,
                ),
                progress_message=self._progress_message,
            )
        else:
            return StageRawData(groups={}, sources={}, replay_allocation=allocation)
        return StageRawData(
            groups=replay.groups,
            sources=replay.sources,
            replay_allocation=allocation,
        )

    def _collect_current_task(self, context: StageContext) -> CurrentTaskData:
        """Collect current-task episodes from its frozen SAC expert.

        The method records task configurations and provenance, initializes the
        GeneralPolicy at stage zero, and returns packed ``(N, horizon, 43)`` data.
        Collection is either the original unfiltered mode or the configured
        success-preferred selection with deterministic fallback. Formal packing
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
            attempted_configurations = []
            collection_mode = cfg.continual.collection_mode
            attempt_multiplier = cfg.continual.collection_max_attempt_multiplier
            expert_action_mode = cfg.continual.expert_action_mode
            trajectories, collection = collect_trajectories(
                env,
                expert,
                task.task_id,
                episodes=count,
                initial_seed=cfg.runtime.seed,
                deterministic=expert_action_mode == "deterministic",
                on_reset=lambda e, obs, seed, episode: attempted_configurations.append(
                    selected_configuration(
                        e, bank, obs, episode, reset_seed=seed
                    )
                ),
                progress=self.progress,
                description=f"Expert collection: {task.task_name}",
                collection_mode=collection_mode,
                max_attempts=count * attempt_multiplier,
                return_metadata=True,
            )
            collected_configurations = [
                attempted_configurations[index]
                for index in collection["selected_attempt_indices"]
            ]
            if self.policy is None:
                torch.manual_seed(cfg.runtime.seed)
                self.policy = GeneralPolicy(
                    env.action_space.low,
                    env.action_space.high,
                    task_conditioning=cfg.bc.task_conditioning,
                    task_order=(
                        self.sequence.task_names
                        if cfg.bc.task_conditioning == "one_hot"
                        else None
                    ),
                    initial_log_std=cfg.bc.initial_log_std,
                    activation=cfg.bc.activation,
                    hidden_dim=cfg.bc.hidden_dim,
                    objective=cfg.bc.loss,
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
            "requested_trajectory_count": count,
            "trajectory_count": len(trajectories),
            "checkpoint": str(self.experts[task.task_name]),
            "checkpoint_sha256": self.expert_hashes[task.task_name],
            "seeds": collection["selected_seeds"],
            "success_count": sum(bool(t.success) for t in trajectories),
            "success_rate": float(np.mean([t.success for t in trajectories])),
            "episode_returns": [t.episode_return for t in trajectories],
            "mean_return": float(np.mean([t.episode_return for t in trajectories])),
            "std_return": float(np.std([t.episode_return for t in trajectories])),
            "collection": {
                key: value
                for key, value in collection.items()
                if key not in ("selected_attempt_indices", "selected_seeds")
            },
        }
        source["collection_configurations"] = collected_configurations
        self.training_banks[task.task_name] = bank
        return CurrentTaskData(
            group=group,
            source=source,
            collected_configurations=collected_configurations,
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
        directory = context.directory
        groups, sources = raw_data.groups, raw_data.sources
        collected_configurations = current_task.collected_configurations

        prepared_stage = prepare_stage_groups(
            groups,
            sources,
            self.sequence.task_names,
            stage=stage,
            horizon=self.horizon,
            runtime_seed=cfg.runtime.seed,
            historical_replay_training_only=(
                cfg.continual.historical_replay_training_only
            ),
        )
        for index, values in groups.items():
            name = self.sequence.task(index).task_name
            path = directory / f"data_task_{index}.npy"
            np.save(path, values.numpy(), allow_pickle=False)
            sources[name].update(data_path=str(path), data_sha256=file_hash(path))

        preparations = prepared_stage.preparations
        for index, prepared in preparations.items():
            name = self.sequence.task(index).task_name
            if sources[name].get("kind") == "expert_recollection":
                attempts = sources[name]["source_attempt_ids"]
                hashes = sources[name]["trajectory_sha256"]
                train_attempts = [attempts[i] for i in prepared.train_ids]
                validation_attempts = [attempts[i] for i in prepared.validation_ids]
                attempt_overlap = sorted(
                    set(train_attempts) & set(validation_attempts)
                )
                if attempt_overlap:
                    raise RuntimeError(
                        "Expert recollection attempt IDs cross the stage split."
                    )
                train_hashes = {hashes[i] for i in prepared.train_ids}
                validation_hashes = {hashes[i] for i in prepared.validation_ids}
                sources[name]["split_diagnostics"] = {
                    "train_trajectory_count": len(prepared.train_ids),
                    "validation_trajectory_count": len(prepared.validation_ids),
                    "train_source_attempt_ids": train_attempts,
                    "validation_source_attempt_ids": validation_attempts,
                    "source_attempt_id_overlap": attempt_overlap,
                    "exact_trajectory_sha256_overlap": sorted(
                        train_hashes & validation_hashes
                    ),
                }
            if index == stage:
                self.train_configurations[name] = {
                    "indices": [
                        collected_configurations[i]["configuration_index"]
                        for i in prepared.train_ids
                    ],
                    "source_trajectory_ids": prepared.train_ids,
                }
        prepared_data = prepared_stage.dataset
        train_reference_goals = []
        validation_reference_goals = []
        for index, values in groups.items():
            name = self.sequence.task(index).task_name
            reference = torch.as_tensor(
                sources[name].get("reference_goals", values[:, 0, 36:39]),
                dtype=values.dtype,
            )
            if (
                reference.shape != (len(values), 3)
                or not torch.isfinite(reference).all()
            ):
                raise ValueError(f"Invalid reference goals for stage task {name}.")
            train_reference_goals.append(reference[preparations[index].train_ids])
            validation_reference_goals.append(
                reference[preparations[index].validation_ids]
            )
        return StageData(
            train=prepared_data.train,
            validation=prepared_data.validation,
            train_labels=prepared_data.train_labels,
            validation_labels=prepared_data.validation_labels,
            train_reference_goals=torch.cat(train_reference_goals),
            validation_reference_goals=torch.cat(validation_reference_goals),
            sources=sources,
            splits=prepared_stage.splits,
            replay_allocation=raw_data.replay_allocation,
        )

    def _fit_stage_diffusion(
        self, context: StageContext, data: StageData
    ) -> DiffusionTrainingResult:
        """Prepare stage normalization and fit the selected replay generator.

        Physical 43D data is normalized directly alongside a separately
        normalized 3D goal condition. Normalizers use only the training split.
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
                train_reference_goals=data.train_reference_goals,
                validation_reference_goals=data.validation_reference_goals,
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
            _synchronize_timing_device(cfg.runtime.device)
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
            _synchronize_timing_device(cfg.runtime.device)
            training_seconds = time.perf_counter() - training_started
        elif self.flow_matching is not None:
            self._progress_message(
                "Training Flow Matching", Epochs=cfg.flow_matching.epochs
            )
            _synchronize_timing_device(cfg.runtime.device)
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
            _synchronize_timing_device(cfg.runtime.device)
            training_seconds = time.perf_counter() - training_started
        return DiffusionTrainingResult(
            normalizer=normalizer,
            goal_normalizer=goal_normalizer,
            losses=diff_losses,
            training_seconds=training_seconds,
        )

    def _fit_stage_policy(self, context: StageContext, data: StageData) -> LossReport:
        """Update the persistent GeneralPolicy on current data plus replay.

        A fresh Adam optimizer is created for the stage, while policy weights
        carry forward. Squashed-Gaussian NLL consumes the unchanged physical
        datasets; inverse-tanh clipping handles exact-boundary and diagnostic
        out-of-bounds replay targets. Executed actions remain bounded.
        """
        cfg = self.config
        stage = context.index
        self._progress_message("Training GeneralPolicy BC", Epochs=cfg.bc.epochs)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.bc.learning_rate
        )
        bc_losses = fit_bc(
            self.policy,
            bc_dataset(
                data.train,
                data.train_labels if cfg.bc.task_conditioning == "one_hot" else None,
            ),
            bc_dataset(
                data.validation,
                data.validation_labels
                if cfg.bc.task_conditioning == "one_hot"
                else None,
            ),
            epochs=cfg.bc.epochs,
            seed=cfg.runtime.seed + stage,
            batch_size=cfg.bc.batch_size,
            learning_rate=cfg.bc.learning_rate,
            allow_out_of_bounds_targets=True,
            optimizer=self.policy_optimizer,
            bc_loss=cfg.bc.loss,
            progress=self.progress,
            track_best_mse=cfg.bc.dual_checkpoint_selection,
            checkpoint_selection=cfg.bc.checkpoint_selection,
        )
        self.best_mse_policy_state = bc_losses.pop(
            "_best_mse_state_dict", None
        )
        return bc_losses

    def _begin_stage(self, stage: int) -> tuple[StageContext, float]:
        """Validate stage order and reserve its canonical directory."""
        if stage != self.next_stage:
            raise ValueError("Stages must execute once, in sequence order.")
        _synchronize_timing_device(self.config.runtime.device)
        stage_started = time.perf_counter()
        task = self.sequence.task(stage)
        if self.progress:
            tqdm.write(
                f"\nStage {stage + 1}/{self.sequence.num_tasks}: {task.task_name}"
            )
        directory = self.output / f"stage_{stage}_{task.task_name}"
        directory.mkdir(exist_ok=False)
        schedule = self.config.continual.current_trajectory_schedule
        trajectory_count = (
            schedule[stage]
            if schedule
            else self.config.continual.trajectories_per_task
        )
        return (
            StageContext(
                index=stage,
                task=task,
                trajectory_count=trajectory_count,
                directory=directory,
                policy_before_hash=(
                    None if self.policy is None else state_hash(self.policy)
                ),
                diffusion_before_hash=(
                    state_hash(self.diffusion) if self.diffusion is not None else None
                ),
                flow_matching_before_hash=(
                    state_hash(self.flow_matching)
                    if self.flow_matching is not None
                    else None
                ),
            ),
            stage_started,
        )

    def _stage_timing(
        self,
        data: StageData,
        generator_training: DiffusionTrainingResult,
        *,
        stage_started: float,
        replay_seconds: float,
        collection_seconds: float,
        bc_seconds: float,
        evaluation_seconds: float,
    ) -> Metadata:
        """Build the established stage timing payload without changing its schema."""
        _synchronize_timing_device(self.config.runtime.device)
        return {
            "expert_collection_seconds": collection_seconds,
            "replay_generation_seconds": replay_seconds,
            "generator_training_seconds": generator_training.training_seconds,
            "bc_training_seconds": bc_seconds,
            "evaluation_seconds": evaluation_seconds,
            "_stage_started_perf_counter": stage_started,
            "replay_neural_function_evaluations": sum(
                int(source.get("neural_function_evaluations", 0))
                for source in data.sources.values()
                if source.get("kind") == "generated"
            ),
            "generator_sampling_steps": (
                self.config.diffusion.steps
                if self.config.continual.replay_mode == "diffusion"
                else self.config.flow_matching.integration_steps
                if self.config.continual.replay_mode == "flow-matching"
                else 0
            ),
            "device_synchronized": torch.device(self.config.runtime.device).type
            == "cuda",
        }

    def _commit_stage(
        self,
        context: StageContext,
        data: StageData,
        generator_training: DiffusionTrainingResult,
        policy_training: LossReport,
        timing: Metadata,
    ) -> Metadata:
        """Persist trained state, reports, and the stage completion marker."""
        artifacts = save_stage_checkpoints(
            self, context, generator_training, policy_training
        )
        return finish_stage(
            self,
            context,
            data,
            generator_training,
            policy_training,
            artifacts,
            timing=timing,
        )

    def train_task(self, stage):
        """Run the seven scientific operations of one continual-learning stage."""
        context, stage_started = self._begin_stage(stage)

        replay, replay_seconds = _timed_call(
            self.config.runtime.device, self._generate_replay, context
        )
        current_task, collection_seconds = _timed_call(
            self.config.runtime.device, self._collect_current_task, context
        )
        raw_data = StageRawData(
            groups={**replay.groups, stage: current_task.group},
            sources={
                **replay.sources,
                context.task.task_name: current_task.source,
            },
            replay_allocation=replay.replay_allocation,
        )
        data = self._prepare_stage_data(context, raw_data, current_task)
        generator_training = self._fit_stage_diffusion(context, data)

        _synchronize_timing_device(self.config.runtime.device)
        bc_started = time.perf_counter()
        policy_training = self._fit_stage_policy(context, data)
        _synchronize_timing_device(self.config.runtime.device)
        bc_seconds = time.perf_counter() - bc_started

        _, evaluation_seconds = _timed_call(
            self.config.runtime.device, self._evaluate_stage, stage
        )
        if self.config.bc.dual_checkpoint_selection:
            self._evaluate_best_mse_stage(stage)

        timing = self._stage_timing(
            data,
            generator_training,
            stage_started=stage_started,
            replay_seconds=replay_seconds,
            collection_seconds=collection_seconds,
            bc_seconds=bc_seconds,
            evaluation_seconds=evaluation_seconds,
        )
        return self._commit_stage(
            context,
            data,
            generator_training,
            policy_training,
            timing,
        )

    def finalize_run(self, *, process_started=None):
        """Publish final run artifacts after every configured stage is complete."""
        timing_device = self.config.runtime.device
        if process_started is None:
            _synchronize_timing_device(timing_device)
            process_started = time.perf_counter()
        final_matrix = self.output / "evaluation_matrix.json"
        write_json(final_matrix, self.matrix.to_dict())
        if self.best_mse_matrix is not None:
            atomic_replace_json(
                self.output / "evaluation_matrix_best_mse.json",
                self.best_mse_matrix.to_dict(),
                allow_nan=False,
            )
        timing_path = self.output / "timing_summary.json"
        if not (self.output / "RUN_COMPLETE").exists():
            stage_timings = []
            for stage in range(self.sequence.num_tasks):
                provenance = self.matrix.stage_provenance[str(stage)]
                stage_report = json.loads(
                    Path(provenance["report"]).read_text(encoding="utf-8")
                )
                stage_timings.append(stage_report["timing"])
            _synchronize_timing_device(timing_device)
            fields = (
                "expert_collection_seconds",
                "replay_generation_seconds",
                "generator_training_seconds",
                "bc_training_seconds",
                "evaluation_seconds",
                "total_stage_seconds",
                "replay_neural_function_evaluations",
            )
            write_json(
                timing_path,
                {
                    "kind": "diffcrl_timing_summary_v1",
                    "status": "complete",
                    "current_process_wall_clock_seconds": time.perf_counter()
                    - process_started,
                    "accumulated_completed_stage_seconds": sum(
                        item["total_stage_seconds"] for item in stage_timings
                    ),
                    "totals": {
                        field: sum(item[field] for item in stage_timings)
                        for field in fields
                    },
                    "stages": stage_timings,
                },
            )
        print_final_report(self.matrix)
        complete_run(self.output)
        return self.matrix

    def run(self):
        """Train all configured tasks sequentially and return the final matrix."""
        _synchronize_timing_device(self.config.runtime.device)
        process_started = time.perf_counter()
        for stage in range(self.sequence.num_tasks):
            self.train_task(stage)
        return self.finalize_run(process_started=process_started)
