"""Persistent deterministic behavior-cloning policy used by DiffCRL.

The policy uses bounded tanh actions and supports the configured post-tanh or
pre-tanh MSE objective. Replay composition is owned by the DiffCRL trainer.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from src.envs import ACTION_SHAPE, OBSERVATION_SHAPE
from src.support.io import atomic_torch_save
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

BC_CLAMP_EPSILON = 1e-4
BC_LOSSES = ("post-tanh-mse", "pre-tanh-mse")
GENERAL_POLICY_HIDDEN_SIZES = (256, 256, 256, 256)


class GeneralPolicy(nn.Module):
    """Persistent deterministic BC policy shared across DiffCRL stages.

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
        self.hidden_sizes = GENERAL_POLICY_HIDDEN_SIZES
        self.use_layer_norm = True
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
        self.input_dim = OBSERVATION_SHAPE[0] + self.num_tasks
        self.register_buffer("action_low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(high, dtype=torch.float32))
        layers: list[nn.Module] = [
            nn.Linear(self.input_dim, GENERAL_POLICY_HIDDEN_SIZES[0]),
            nn.LayerNorm(GENERAL_POLICY_HIDDEN_SIZES[0]),
            nn.ReLU(),
        ]
        for input_size, output_size in zip(
            GENERAL_POLICY_HIDDEN_SIZES, GENERAL_POLICY_HIDDEN_SIZES[1:]
        ):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        layers.extend(
            (nn.Linear(GENERAL_POLICY_HIDDEN_SIZES[-1], ACTION_SHAPE[0]), nn.Tanh())
        )
        self.network = nn.Sequential(*layers)

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

    def forward(
        self, observations: torch.Tensor, task_ids: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        """Map raw 39D observations to bounded physical 4D actions."""
        actions = self.network(self.policy_input(observations, task_ids))
        return self.action_low + (actions + 1) * 0.5 * (
            self.action_high - self.action_low
        )

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        *,
        task_id: int | None = None,
    ) -> np.ndarray:
        """Common policy interface; this policy is deterministic for either flag."""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != OBSERVATION_SHAPE or not np.isfinite(observation).all():
            raise ValueError("Expected one finite MetaWorld observation.")
        return (
            self(torch.from_numpy(observation).to(self.action_low.device), task_id)
            .cpu()
            .numpy()
            .copy()
        )

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
        payload = {
                "kind": "general_policy_v2",
                "architecture": {
                    "observation_dim": self.input_dim,
                    "hidden_sizes": list(self.hidden_sizes),
                    "action_dim": ACTION_SHAPE[0],
                    "activation": "relu",
                    "output_activation": "tanh",
                    "layer_norm_after_hidden": [0] if self.use_layer_norm else [],
                },
                "hidden_sizes": self.hidden_sizes,
                "use_layer_norm": self.use_layer_norm,
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
    ) -> GeneralPolicy:
        """Restore a compatible inference checkpoint on CPU in evaluation mode."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        kind = data.get("kind")
        if kind == "general_policy_v2":
            use_layer_norm = data.get("use_layer_norm")
            if not isinstance(use_layer_norm, bool):
                raise ValueError(
                    "GeneralPolicy checkpoint has no valid use_layer_norm flag."
                )
        else:
            raise ValueError(
                "Checkpoint is incompatible with the GeneralPolicy architecture."
            )
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
        if hidden_sizes != GENERAL_POLICY_HIDDEN_SIZES:
            raise ValueError(
                "GeneralPolicy checkpoint uses an unsupported architecture; only "
                "four hidden layers of width 256 are accepted."
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
    architecture = data.get("architecture")
    expected = {
        "observation_dim": OBSERVATION_SHAPE[0] + len(data.get("task_order", ())),
        "hidden_sizes": list(hidden_sizes),
        "action_dim": ACTION_SHAPE[0],
        "activation": "relu",
        "output_activation": "tanh",
        "layer_norm_after_hidden": [0] if data.get("use_layer_norm", True) else [],
    }
    if architecture != expected:
        raise ValueError(
            "Checkpoint state_dict is incompatible with its stored "
            "GeneralPolicy architecture metadata."
        )
    return hidden_sizes


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
    bc_loss: str = "post-tanh-mse",
    progress: bool = False,
) -> dict:
    """Fit BC and restore the epoch with lowest held-out timestep action MSE.

    Supplied optimizer is retained, including its hyperparameters. Model and
    optimizer are restored together to the best validation epoch. Without one,
    a fresh Adam is created as in the standalone API.

    Whole episodes are split before this function; the supplied datasets contain
    flattened timestep state-action samples. ``seed`` controls minibatch shuffling.
    Losses are means across all samples and all four action coordinates.
    The explicit diagnostic opt-in allow_out_of_bounds_targets leaves targets
    untouched. Policy outputs remain bounded, so such targets cannot be fitted
    exactly. Default callers still reject targets outside the action space.
    """
    if (
        epochs < 1
        or batch_size < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    if bc_loss not in BC_LOSSES:
        raise ValueError(f"BC loss must be one of {BC_LOSSES}.")
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

    def predictions(observations, task_ids=None):
        z = policy.network[:-1](policy.policy_input(observations, task_ids))
        action = policy.action_low + (torch.tanh(z) + 1) * 0.5 * (
            policy.action_high - policy.action_low
        )
        return z, action

    def losses(dataset: TensorDataset) -> dict:
        """Report objective-space loss separately from physical action MSE."""
        with torch.no_grad():
            task_ids = dataset.tensors[2] if expected_tensors == 3 else None
            z, action = predictions(dataset.tensors[0], task_ids)
            target = dataset.tensors[1]
            action_mse = criterion(action, target).item()
            if bc_loss == "pre-tanh-mse":
                scaled = (
                    2
                    * (target - policy.action_low)
                    / (policy.action_high - policy.action_low)
                    - 1
                )
                target = torch.atanh(
                    scaled.clamp(-1 + BC_CLAMP_EPSILON, 1 - BC_CLAMP_EPSILON)
                )
                optimization_loss = criterion(z, target).item()
            else:
                optimization_loss = action_mse
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
                z, predicted_action = predictions(observations, task_ids)
                if bc_loss == "pre-tanh-mse":
                    scaled = (
                        2
                        * (actions - policy.action_low)
                        / (policy.action_high - policy.action_low)
                        - 1
                    )
                    target = torch.atanh(
                        scaled.clamp(-1 + BC_CLAMP_EPSILON, 1 - BC_CLAMP_EPSILON)
                    )
                    objective = criterion(z, target)
                else:
                    objective = criterion(predicted_action, actions)
                if not torch.isfinite(objective):
                    raise ValueError("BC loss became non-finite.")
                objective.backward()
                optimizer.step()
            policy.eval()
            record = {"epoch": epoch, **metrics()}
            history.append(record)
            if record["validation_action_mse"] < best_loss:
                best_loss, best_epoch = record["validation_action_mse"], epoch
                best_state = deepcopy(policy.state_dict())
                best_optimizer_state = deepcopy(optimizer.state_dict())
            epoch_progress.set_postfix(
                train=f"{record['train_action_mse']:.5f}",
                validation=f"{record['validation_action_mse']:.5f}",
                best=f"{best_loss:.5f}@{best_epoch}",
            )
    policy.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer_state)
    result = {
        "initial": initial,
        "best_epoch": best_epoch,
        **metrics(),
        "history": history,
    }
    if progress:
        tqdm.write(
            f"BC selected epoch {best_epoch}/{epochs}: "
            f"validation_action_mse={result['validation_action_mse']:.6f}, "
            f"train_action_mse={result['train_action_mse']:.6f}"
        )
    return result
