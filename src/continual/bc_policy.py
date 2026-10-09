"""Persistent deterministic behavior-cloning policy used by DiffCRL.

The policy uses bounded tanh actions and supports the configured post-tanh or
pre-tanh MSE objective. Replay composition is owned by the DiffCRL trainer.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.envs import ACTION_SHAPE, OBSERVATION_SHAPE
from src.support.io import atomic_torch_save

BC_CLAMP_EPSILON = 1e-4
BC_LOSSES = ("post-tanh-mse", "pre-tanh-mse")


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
        hidden_sizes: tuple[int, int] = (256, 256),
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
        if len(hidden_sizes) != 2 or any(h <= 0 for h in hidden_sizes):
            raise ValueError("Provide two positive hidden sizes.")
        self.hidden_sizes = tuple(hidden_sizes)
        self.register_buffer("action_low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(high, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_SHAPE[0], hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], ACTION_SHAPE[0]),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map raw 39D observations to bounded physical 4D actions."""
        actions = self.network(observations)
        return self.action_low + (actions + 1) * 0.5 * (
            self.action_high - self.action_low
        )

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Common policy interface; this policy is deterministic for either flag."""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != OBSERVATION_SHAPE or not np.isfinite(observation).all():
            raise ValueError("Expected one finite MetaWorld observation.")
        return (
            self(torch.from_numpy(observation).to(self.action_low.device))
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
        atomic_torch_save(
            path,
            {
                "kind": "general_policy_layer_norm_v1",
                "hidden_sizes": self.hidden_sizes,
                "state_dict": self.state_dict(),
                **extra,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> GeneralPolicy:
        """Restore a compatible inference checkpoint on CPU in evaluation mode."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("kind") != "general_policy_layer_norm_v1":
            raise ValueError(
                "Checkpoint is incompatible with the LayerNorm GeneralPolicy."
            )
        state = data["state_dict"]
        policy = cls(
            state["action_low"].numpy(),
            state["action_high"].numpy(),
            hidden_sizes=tuple(data["hidden_sizes"]),
        )
        policy.load_state_dict(state)
        policy.eval()
        return policy

    @classmethod
    def load_training(cls, path: str | Path):
        """Return (policy, Adam or None, metadata).

        No optimizer is silently initialized for a checkpoint lacking its state.
        Checkpoints from the former observation-z-score architecture are rejected.
        """
        policy = cls.load(path)
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
    for dataset in (train_data, validation_data):
        if len(dataset.tensors) != 2 or len(dataset) == 0:
            raise ValueError("Expected nonempty observation/action datasets.")
        x, y = dataset.tensors
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

    def predictions(observations):
        z = policy.network[:-1](observations)
        action = policy.action_low + (torch.tanh(z) + 1) * 0.5 * (
            policy.action_high - policy.action_low
        )
        return z, action

    def losses(dataset: TensorDataset) -> dict:
        """Report objective-space loss separately from physical action MSE."""
        with torch.no_grad():
            z, action = predictions(dataset.tensors[0])
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
    for epoch in tqdm(
        range(1, epochs + 1),
        desc="BC training",
        unit="epoch",
        disable=not progress,
    ):
        policy.train()
        for observations, actions in loader:
            optimizer.zero_grad()
            z, predicted_action = predictions(observations)
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
    policy.load_state_dict(best_state)
    optimizer.load_state_dict(best_optimizer_state)
    return {
        "initial": initial,
        "best_epoch": best_epoch,
        **metrics(),
        "history": history,
    }
