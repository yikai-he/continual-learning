"""Persistent deterministic or Squashed-Gaussian BC policy used by DiffCRL."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from continual_manipulation.environments import ACTION_SHAPE, OBSERVATION_SHAPE
from src.support.io import atomic_torch_save
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

NLL_ATANH_EPSILON = 1e-4
BC_CLAMP_EPSILON = NLL_ATANH_EPSILON
SQUASHED_GAUSSIAN_NLL = "squashed_gaussian_nll"
DETERMINISTIC_MSE = "deterministic_mse"
LEGACY_SQUASHED_GAUSSIAN_NLL = "squashed-gaussian-nll"
BC_OBJECTIVES = (SQUASHED_GAUSSIAN_NLL, DETERMINISTIC_MSE)
NLL_CHECKPOINT_SELECTIONS = ("validation_nll", "validation_physical_mse")
BC_LOSSES = (*BC_OBJECTIVES, LEGACY_SQUASHED_GAUSSIAN_NLL)
POLICY_CHECKPOINT_KINDS = {
    SQUASHED_GAUSSIAN_NLL: "general_policy_squashed_gaussian_v1",
    DETERMINISTIC_MSE: "general_policy_deterministic_mse_v1",
}
GENERAL_POLICY_HIDDEN_SIZES = (256, 256, 256, 256)
GENERAL_POLICY_HIDDEN_DIMS = (256, 512)
GENERAL_POLICY_INITIAL_LOG_STD = -3.0
GENERAL_POLICY_ACTIVATIONS = ("relu", "leaky_relu")
LEGACY_LEAKY_RELU_NEGATIVE_SLOPE = 0.01


class GeneralPolicy(nn.Module):
    """Persistent BC policy with an explicitly selected training objective.

    The network consumes raw 39D observations. Its tanh output is affinely
    mapped to the environment bounds, so executed actions remain valid even
    when generated replay contains out-of-bounds training targets.
    """

    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        task_conditioning: str = "none",
        task_order: tuple[str, ...] | list[str] | None = None,
        initial_log_std: float = GENERAL_POLICY_INITIAL_LOG_STD,
        activation: str = "relu",
        hidden_dim: int = 256,
        objective: str = SQUASHED_GAUSSIAN_NLL,
    ) -> None:
        super().__init__()
        low, high = np.asarray(action_low), np.asarray(action_high)
        if (
            low.shape != ACTION_SHAPE
            or high.shape != ACTION_SHAPE
            or not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or not (low < high).all()
        ):
            raise ValueError("Expected finite, ordered MetaWorld action bounds.")
        if hidden_dim not in GENERAL_POLICY_HIDDEN_DIMS:
            raise ValueError(
                f"hidden_dim must be one of {GENERAL_POLICY_HIDDEN_DIMS}."
            )
        self.hidden_dim = hidden_dim
        self.hidden_sizes = (hidden_dim,) * 4
        self.use_layer_norm = True
        self.objective = canonical_bc_objective(objective)
        if activation not in GENERAL_POLICY_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {GENERAL_POLICY_ACTIVATIONS}."
            )
        self.activation = activation
        if task_conditioning not in ("none", "one_hot"):
            raise ValueError("task_conditioning must be 'none' or 'one_hot'.")
        order = tuple(task_order or ())
        if task_conditioning == "one_hot" and (not order or len(set(order)) != len(order)):
            raise ValueError("One-hot task conditioning requires a nonempty unique task order.")
        if task_conditioning == "none" and order:
            raise ValueError("Task order is only valid with one-hot task conditioning.")
        self.task_conditioning = task_conditioning
        self.task_order = order
        self.num_tasks = len(order)
        if not np.isfinite(initial_log_std) or not LOG_STD_MIN <= initial_log_std <= LOG_STD_MAX:
            raise ValueError(
                f"initial_log_std must be finite and in [{LOG_STD_MIN}, {LOG_STD_MAX}]."
            )
        self.initial_log_std = float(initial_log_std)
        self.input_dim = OBSERVATION_SHAPE[0] + self.num_tasks
        self.register_buffer("action_low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(high, dtype=torch.float32))
        first_activation: nn.Module = (
            nn.ReLU() if activation == "relu" else nn.Tanh()
        )
        later_activation = (
            (lambda: nn.ReLU())
            if activation == "relu"
            else (lambda: nn.LeakyReLU(LEGACY_LEAKY_RELU_NEGATIVE_SLOPE))
        )
        layers: list[nn.Module] = [
            nn.Linear(self.input_dim, self.hidden_sizes[0]),
            nn.LayerNorm(self.hidden_sizes[0]),
            first_activation,
        ]
        for input_size, output_size in zip(
            self.hidden_sizes, self.hidden_sizes[1:]
        ):
            layers.extend((nn.Linear(input_size, output_size), later_activation()))
        self.shared_network = nn.Sequential(*layers)
        self.mean_head = nn.Linear(self.hidden_sizes[-1], ACTION_SHAPE[0])
        if self.objective == SQUASHED_GAUSSIAN_NLL:
            self.log_std_head = nn.Linear(
                self.hidden_sizes[-1], ACTION_SHAPE[0]
            )
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, self.initial_log_std)
        else:
            self.log_std_head = None

    def policy_input(
        self, observations: torch.Tensor, task_ids: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        if self.task_conditioning == "none":
            if task_ids is not None:
                raise ValueError("Task IDs are invalid for a task-agnostic policy.")
            return observations
        if task_ids is None:
            raise ValueError("Task IDs are required for a task-conditioned policy.")
        ids = torch.as_tensor(task_ids, device=observations.device, dtype=torch.long)
        if observations.ndim == 1:
            if ids.ndim != 0:
                raise ValueError("Expected one scalar task ID for one observation.")
        elif ids.shape != observations.shape[:-1]:
            raise ValueError("Task ID shape must match observation batch dimensions.")
        if torch.any(ids < 0) or torch.any(ids >= self.num_tasks):
            raise ValueError("Task ID is outside the configured task order.")
        return torch.cat(
            (observations, F.one_hot(ids, num_classes=self.num_tasks).float()), dim=-1
        )

    def distribution_parameters(
        self, observations: torch.Tensor, task_ids: torch.Tensor | int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian mean and bounded log standard deviation."""
        if self.log_std_head is None:
            raise ValueError("Deterministic MSE policy has no Gaussian distribution.")
        latent = self.shared_network(self.policy_input(observations, task_ids))
        mean = self.mean_head(latent)
        log_std = self.log_std_head(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def scale_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """Affinely map normalized actions to physical environment bounds."""
        return self.action_low + (normalized_action + 1) * 0.5 * (
            self.action_high - self.action_low
        )

    def forward(
        self, observations: torch.Tensor, task_ids: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        """Return the deterministic physical action ``scale(tanh(mean))``."""
        latent = self.shared_network(self.policy_input(observations, task_ids))
        mean = self.mean_head(latent)
        return self.scale_action(torch.tanh(mean))

    def normalized_log_prob(
        self,
        observations: torch.Tensor,
        physical_actions: torch.Tensor,
        task_ids: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Log probability of physical targets in normalized coordinates."""
        if self.objective != SQUASHED_GAUSSIAN_NLL:
            raise ValueError("Normalized log probability requires the NLL objective.")
        normalized = 2 * (physical_actions - self.action_low) / (
            self.action_high - self.action_low
        ) - 1
        normalized = normalized.clamp(
            -1 + NLL_ATANH_EPSILON, 1 - NLL_ATANH_EPSILON
        )
        pre_tanh = torch.atanh(normalized)
        mean, log_std = self.distribution_parameters(observations, task_ids)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        log_det = 2 * (
            np.log(2.0) - pre_tanh - F.softplus(-2 * pre_tanh)
        )
        return (distribution.log_prob(pre_tanh) - log_det).sum(dim=-1)

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        *,
        task_id: int | None = None,
    ) -> np.ndarray:
        """Return the squashed mean or a reparameterized Gaussian sample."""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != OBSERVATION_SHAPE or not np.isfinite(observation).all():
            raise ValueError("Expected one finite MetaWorld observation.")
        tensor = torch.from_numpy(observation).to(self.action_low.device)
        if self.objective == DETERMINISTIC_MSE:
            return self(tensor, task_id).cpu().numpy().copy()
        mean, log_std = self.distribution_parameters(tensor, task_id)
        pre_tanh = (
            mean
            if deterministic
            else mean + log_std.exp() * torch.randn_like(mean)
        )
        return self.scale_action(torch.tanh(pre_tanh)).cpu().numpy().copy()

    def save(self, path: str | Path, *, optimizer=None, metadata=None) -> None:
        """Save policy state and optional Adam state without overwriting a file."""
        extra = {"metadata": metadata}
        if optimizer is not None:
            if type(optimizer) is not torch.optim.Adam:
                raise ValueError(
                    "Continuation checkpoints currently support Adam only."
                )
            _validate_optimizer(self, optimizer)
            names = {id(p): name for name, p in self.named_parameters()}
            extra.update(
                optimizer_state=optimizer.state_dict(),
                optimizer_type="Adam",
                optimizer_parameter_names=[
                    [names[id(p)] for p in group["params"]]
                    for group in optimizer.param_groups
                ],
            )
        architecture = {
            "observation_dim": self.input_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "action_dim": ACTION_SHAPE[0],
            "activation": self.activation,
            "first_hidden_activation": (
                "relu" if self.activation == "relu" else "tanh"
            ),
            "leaky_relu_negative_slope": (
                LEGACY_LEAKY_RELU_NEGATIVE_SLOPE
                if self.activation == "leaky_relu"
                else None
            ),
            "policy_distribution": (
                "squashed_gaussian"
                if self.objective == SQUASHED_GAUSSIAN_NLL
                else "deterministic"
            ),
            "mean_activation": "identity",
            "layer_norm_after_hidden": [0] if self.use_layer_norm else [],
        }
        if self.objective == SQUASHED_GAUSSIAN_NLL:
            architecture.update(
                log_std_bounds=[LOG_STD_MIN, LOG_STD_MAX],
                nll_atanh_epsilon=NLL_ATANH_EPSILON,
            )
        payload = {
            "kind": POLICY_CHECKPOINT_KINDS[self.objective],
            "objective": self.objective,
            "architecture": architecture,
            "hidden_sizes": self.hidden_sizes,
            "hidden_dim": self.hidden_dim,
            "use_layer_norm": self.use_layer_norm,
            "initial_log_std": self.initial_log_std,
            "activation": self.activation,
            "state_dict": self.state_dict(),
            **extra,
        }
        if self.task_conditioning == "one_hot":
            payload.update(
                task_conditioning=self.task_conditioning,
                task_order=list(self.task_order),
                num_tasks=self.num_tasks,
            )
        atomic_torch_save(path, payload)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_use_layer_norm: bool | None = None,
        expected_hidden_sizes: tuple[int, ...] | None = None,
        expected_task_conditioning: str | None = None,
        expected_task_order: tuple[str, ...] | list[str] | None = None,
        expected_activation: str | None = None,
        expected_objective: str | None = None,
    ) -> GeneralPolicy:
        """Restore a compatible inference checkpoint on CPU in evaluation mode."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        kind = data.get("kind")
        if kind == POLICY_CHECKPOINT_KINDS[SQUASHED_GAUSSIAN_NLL]:
            objective = SQUASHED_GAUSSIAN_NLL
        elif kind == POLICY_CHECKPOINT_KINDS[DETERMINISTIC_MSE]:
            objective = DETERMINISTIC_MSE
        else:
            objective = None
        if objective is not None:
            use_layer_norm = data.get("use_layer_norm")
            if not isinstance(use_layer_norm, bool):
                raise ValueError(
                    "GeneralPolicy checkpoint has no valid use_layer_norm flag."
                )
        elif kind == "general_policy_v2":
            raise ValueError(
                "Deterministic general_policy_v2 checkpoints are historical and "
                "cannot be loaded as a stochastic GeneralPolicy."
            )
        else:
            raise ValueError(
                "Checkpoint is incompatible with the GeneralPolicy architecture."
            )
        stored_objective = data.get("objective", objective)
        if stored_objective != objective:
            raise ValueError("GeneralPolicy checkpoint objective metadata is inconsistent.")
        if expected_objective is not None and objective != canonical_bc_objective(
            expected_objective
        ):
            raise ValueError("GeneralPolicy checkpoint objective does not match the requested objective.")
        if (
            expected_use_layer_norm is not None
            and use_layer_norm != expected_use_layer_norm
        ):
            raise ValueError(
                "GeneralPolicy checkpoint LayerNorm architecture does not match "
                f"the requested configuration: checkpoint={use_layer_norm}, "
                f"requested={expected_use_layer_norm}."
            )
        if use_layer_norm is not True:
            raise ValueError(
                "GeneralPolicy checkpoint uses an unsupported architecture; "
                "LayerNorm after the first hidden layer is required."
            )
        state = data["state_dict"]
        task_conditioning = data.get("task_conditioning", "none")
        task_order = tuple(data.get("task_order", ()))
        if expected_task_conditioning is not None and task_conditioning != expected_task_conditioning:
            raise ValueError("GeneralPolicy checkpoint task-conditioning mode does not match the requested configuration.")
        if expected_task_order is not None and task_order != tuple(expected_task_order):
            raise ValueError("GeneralPolicy checkpoint task ordering does not match the requested configuration.")
        hidden_sizes = _validated_hidden_sizes(data)
        activation = _validated_activation(data)
        if expected_activation is not None and activation != expected_activation:
            raise ValueError(
                "GeneralPolicy checkpoint activation does not match the requested "
                f"configuration: checkpoint={activation}, requested={expected_activation}."
            )
        if (
            len(hidden_sizes) != 4
            or len(set(hidden_sizes)) != 1
            or hidden_sizes[0] not in GENERAL_POLICY_HIDDEN_DIMS
        ):
            raise ValueError(
                "GeneralPolicy checkpoint uses an unsupported architecture; only "
                "four equal-width hidden layers of width 256 or 512 are accepted."
            )
        if expected_hidden_sizes is not None and hidden_sizes != tuple(expected_hidden_sizes):
            raise ValueError(
                "GeneralPolicy checkpoint architecture does not match the requested "
                f"architecture: checkpoint={hidden_sizes}, requested={tuple(expected_hidden_sizes)}."
            )
        policy = cls(
            state["action_low"].numpy(),
            state["action_high"].numpy(),
            task_conditioning=task_conditioning,
            task_order=task_order,
            initial_log_std=data.get(
                "initial_log_std", GENERAL_POLICY_INITIAL_LOG_STD
            ),
            activation=activation,
            hidden_dim=hidden_sizes[0],
            objective=objective,
        )
        try:
            policy.load_state_dict(state)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint state_dict is incompatible with its stored "
                f"GeneralPolicy architecture {policy.hidden_sizes}."
            ) from exc
        policy.eval()
        return policy

    @classmethod
    def load_training(
        cls,
        path: str | Path,
        *,
        expected_use_layer_norm: bool | None = None,
        expected_hidden_sizes: tuple[int, ...] | None = None,
        expected_task_conditioning: str | None = None,
        expected_task_order: tuple[str, ...] | list[str] | None = None,
        expected_activation: str | None = None,
        expected_objective: str | None = None,
    ):
        """Return (policy, Adam or None, metadata).

        No optimizer is silently initialized for a checkpoint lacking its state.
        Checkpoints from the former observation-z-score architecture are rejected.
        """
        policy = cls.load(
            path,
            expected_use_layer_norm=expected_use_layer_norm,
            expected_hidden_sizes=expected_hidden_sizes,
            expected_task_conditioning=expected_task_conditioning,
            expected_task_order=expected_task_order,
            expected_activation=expected_activation,
            expected_objective=expected_objective,
        )
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "optimizer_state" not in data:
            return policy, None, data.get("metadata")
        if data.get("optimizer_type") != "Adam":
            raise ValueError("Unsupported saved optimizer type.")
        parameters = dict(policy.named_parameters())
        groups = [
            {"params": [parameters[name] for name in names]}
            for names in data["optimizer_parameter_names"]
        ]
        optimizer = torch.optim.Adam(groups)
        optimizer.load_state_dict(data["optimizer_state"])
        _validate_optimizer(policy, optimizer)
        return policy, optimizer, data.get("metadata")


def _validate_optimizer(policy, optimizer):
    """Require the optimizer to own every policy parameter exactly once."""
    expected = {id(p) for p in policy.parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Optimizer must own each policy parameter exactly once.")


def _validated_hidden_sizes(data: dict) -> tuple[int, ...]:
    """Validate the explicit current checkpoint architecture."""
    value = data.get("hidden_sizes")
    if not isinstance(value, (tuple, list)):
        raise ValueError("GeneralPolicy checkpoint has invalid hidden_sizes metadata.")
    hidden_sizes = tuple(value)
    stored_hidden_dim = data.get("hidden_dim", 256)
    if not hidden_sizes or stored_hidden_dim != hidden_sizes[0]:
        raise ValueError(
            "GeneralPolicy checkpoint hidden-width metadata is inconsistent."
        )
    architecture = data.get("architecture")
    objective = _checkpoint_objective(data)
    expected = {
        "observation_dim": OBSERVATION_SHAPE[0] + len(data.get("task_order", ())),
        "hidden_sizes": list(hidden_sizes),
        "action_dim": ACTION_SHAPE[0],
        "activation": _validated_activation(data),
        "first_hidden_activation": (
            "relu" if _validated_activation(data) == "relu" else "tanh"
        ),
        "leaky_relu_negative_slope": (
            LEGACY_LEAKY_RELU_NEGATIVE_SLOPE
            if _validated_activation(data) == "leaky_relu"
            else None
        ),
        "policy_distribution": (
            "squashed_gaussian"
            if objective == SQUASHED_GAUSSIAN_NLL
            else "deterministic"
        ),
        "mean_activation": "identity",
        "layer_norm_after_hidden": [0] if data.get("use_layer_norm", True) else [],
    }
    if objective == SQUASHED_GAUSSIAN_NLL:
        expected.update(
            log_std_bounds=[LOG_STD_MIN, LOG_STD_MAX],
            nll_atanh_epsilon=NLL_ATANH_EPSILON,
        )
    normalized = dict(architecture) if isinstance(architecture, dict) else architecture
    if isinstance(normalized, dict):
        activation = normalized.get("activation", "relu")
        normalized.setdefault("activation", activation)
        normalized.setdefault("first_hidden_activation", activation)
        normalized.setdefault("leaky_relu_negative_slope", None)
    if normalized != expected:
        raise ValueError(
            "Checkpoint state_dict is incompatible with its stored "
            "GeneralPolicy architecture metadata."
        )
    return hidden_sizes


def _validated_activation(data: dict) -> str:
    """Return explicit activation identity, treating old metadata as ReLU."""
    architecture = data.get("architecture")
    architecture_activation = (
        architecture.get("activation", "relu")
        if isinstance(architecture, dict)
        else "relu"
    )
    activation = data.get("activation", architecture_activation)
    if activation not in GENERAL_POLICY_ACTIVATIONS:
        raise ValueError("GeneralPolicy checkpoint has invalid activation metadata.")
    if architecture_activation != activation:
        raise ValueError(
            "GeneralPolicy checkpoint activation metadata is inconsistent."
        )
    return activation


def canonical_bc_objective(value: str) -> str:
    """Resolve the historical NLL spelling without changing its semantics."""
    if value == LEGACY_SQUASHED_GAUSSIAN_NLL:
        return SQUASHED_GAUSSIAN_NLL
    if value not in BC_OBJECTIVES:
        raise ValueError(f"BC objective must be one of {BC_OBJECTIVES}.")
    return value


def _checkpoint_objective(data: dict) -> str:
    kind = data.get("kind")
    for objective, checkpoint_kind in POLICY_CHECKPOINT_KINDS.items():
        if kind == checkpoint_kind:
            stored = data.get("objective", objective)
            if stored != objective:
                raise ValueError("GeneralPolicy checkpoint objective metadata is inconsistent.")
            return objective
    raise ValueError("GeneralPolicy checkpoint has an unsupported objective.")


def fit_bc(
    policy: GeneralPolicy,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    allow_out_of_bounds_targets: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    bc_loss: str = LEGACY_SQUASHED_GAUSSIAN_NLL,
    progress: bool = False,
    track_best_mse: bool = False,
    checkpoint_selection: str = "validation_nll",
) -> dict:
    """Fit the selected BC objective and restore its lowest-validation epoch.

    Supplied optimizer is retained, including its hyperparameters. Model and
    optimizer are restored together to the best validation epoch. Without one,
    a fresh Adam is created as in the standalone API.

    Whole episodes are split before this function; the supplied datasets contain
    flattened timestep state-action samples. ``seed`` controls minibatch shuffling.
    NLL is summed over action coordinates and averaged over samples. Physical
    targets are normalized and clipped only for the numerically safe inverse
    tanh; the source replay tensors remain unchanged. The explicit diagnostic
    opt-in accepts generated targets outside physical bounds, mapping them to
    the same inverse-tanh boundary used for exact-boundary targets.
    """
    if (
        epochs < 1
        or batch_size < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    objective_name = canonical_bc_objective(bc_loss)
    if policy.objective != objective_name:
        raise ValueError("BC objective does not match the GeneralPolicy objective.")
    if track_best_mse and objective_name != SQUASHED_GAUSSIAN_NLL:
        raise ValueError("Dual checkpoint selection requires the NLL objective.")
    if checkpoint_selection not in NLL_CHECKPOINT_SELECTIONS:
        raise ValueError(
            f"Checkpoint selection must be one of {NLL_CHECKPOINT_SELECTIONS}."
        )
    if (
        checkpoint_selection == "validation_physical_mse"
        and objective_name != SQUASHED_GAUSSIAN_NLL
    ):
        raise ValueError(
            "Physical-MSE checkpoint selection requires the NLL objective."
        )
    expected_tensors = 3 if policy.task_conditioning == "one_hot" else 2
    for dataset in (train_data, validation_data):
        if len(dataset.tensors) != expected_tensors or len(dataset) == 0:
            raise ValueError("Expected nonempty observation/action datasets.")
        x, y = dataset.tensors[:2]
        if (
            x.shape != (len(dataset), *OBSERVATION_SHAPE)
            or y.shape != (len(dataset), *ACTION_SHAPE)
            or not torch.isfinite(x).all()
            or not torch.isfinite(y).all()
        ):
            raise ValueError("Invalid BC sample shapes or values.")
        if not allow_out_of_bounds_targets and (
            torch.any(y < policy.action_low.cpu())
            or torch.any(y > policy.action_high.cpu())
        ):
            raise ValueError("Dataset actions exceed policy bounds.")
        if expected_tensors == 3:
            task_ids = dataset.tensors[2]
            if task_ids.shape != (len(dataset),) or task_ids.dtype != torch.long:
                raise ValueError("Expected one integer task ID per BC sample.")
            policy.policy_input(x, task_ids)
    device = policy.action_low.device
    train_data = TensorDataset(*(t.to(device) for t in train_data.tensors))
    validation_data = TensorDataset(*(t.to(device) for t in validation_data.tensors))
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    _validate_optimizer(policy, optimizer)
    criterion = nn.MSELoss()

    def losses(dataset: TensorDataset) -> dict:
        """Report objective-space loss separately from physical action MSE."""
        with torch.no_grad():
            task_ids = dataset.tensors[2] if expected_tensors == 3 else None
            action = policy(dataset.tensors[0], task_ids)
            target = dataset.tensors[1]
            action_mse = criterion(action, target).item()
            optimization_loss = (
                action_mse
                if objective_name == DETERMINISTIC_MSE
                else -policy.normalized_log_prob(
                    dataset.tensors[0], target, task_ids
                ).mean().item()
            )
            return {"loss": optimization_loss, "action_mse": action_mse}

    def metrics():
        """Evaluate train and validation datasets without minibatch averaging bias."""
        train, validation = losses(train_data), losses(validation_data)
        return {
            "train_loss": train["loss"],
            "validation_loss": validation["loss"],
            "train_action_mse": train["action_mse"],
            "validation_action_mse": validation["action_mse"],
        }

    initial = metrics()
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    best_optimizer_state = None
    best_mse, best_mse_record, best_mse_state = float("inf"), None, None
    best_mse_optimizer_state = None
    select_by_mse = checkpoint_selection == "validation_physical_mse"
    with tqdm(
        range(1, epochs + 1),
        desc="BC training",
        unit="epoch",
        disable=not progress,
    ) as epoch_progress:
        for epoch in epoch_progress:
            policy.train()
            for batch in loader:
                observations, actions = batch[:2]
                task_ids = batch[2] if expected_tensors == 3 else None
                optimizer.zero_grad()
                objective = (
                    criterion(policy(observations, task_ids), actions)
                    if objective_name == DETERMINISTIC_MSE
                    else -policy.normalized_log_prob(
                        observations, actions, task_ids
                    ).mean()
                )
                if not torch.isfinite(objective):
                    raise ValueError("BC loss became non-finite.")
                objective.backward()
                optimizer.step()
            policy.eval()
            record = {"epoch": epoch, **metrics()}
            history.append(record)
            if record["validation_loss"] < best_loss:
                best_loss, best_epoch = record["validation_loss"], epoch
                best_state = deepcopy(policy.state_dict())
                best_optimizer_state = deepcopy(optimizer.state_dict())
            if (track_best_mse or select_by_mse) and record[
                "validation_action_mse"
            ] < best_mse:
                best_mse = record["validation_action_mse"]
                best_mse_record = dict(record)
                best_mse_state = deepcopy(policy.state_dict())
                best_mse_optimizer_state = deepcopy(optimizer.state_dict())
            epoch_progress.set_postfix(
                train=f"{record['train_loss']:.5f}",
                validation=f"{record['validation_loss']:.5f}",
                best=(
                    f"{best_mse:.5f}@{best_mse_record['epoch']}"
                    if select_by_mse
                    else f"{best_loss:.5f}@{best_epoch}"
                ),
            )
    selected_epoch = best_mse_record["epoch"] if select_by_mse else best_epoch
    policy.load_state_dict(best_mse_state if select_by_mse else best_state)
    optimizer.load_state_dict(
        best_mse_optimizer_state if select_by_mse else best_optimizer_state
    )
    result = {
        "initial": initial,
        "best_epoch": selected_epoch,
        **metrics(),
        "history": history,
    }
    if track_best_mse:
        result["best_mse_checkpoint"] = {
            "epoch": best_mse_record["epoch"],
            "validation_loss": best_mse_record["validation_loss"],
            "validation_action_mse": best_mse_record["validation_action_mse"],
        }
        # Consumed by the stage coordinator before JSON reporting. Keeping the
        # tensor state here avoids mutating the authoritative restored policy.
        result["_best_mse_state_dict"] = best_mse_state
    if progress:
        tqdm.write(
            f"BC selected epoch {selected_epoch}/{epochs}: "
            f"objective={objective_name}, "
            f"validation={result['validation_loss']:.6f}, "
            f"train={result['train_loss']:.6f}"
        )
    return result
