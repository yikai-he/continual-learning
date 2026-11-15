#!/usr/bin/env python3
"""Isolated Button Press deterministic-MSE versus stochastic-NLL diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.continual.bc_policy import (  # noqa: E402
    GENERAL_POLICY_HIDDEN_SIZES,
    GeneralPolicy,
)
from continual_manipulation.evaluation import evaluate_task  # noqa: E402
from src.continual.stage_dataset import bc_dataset  # noqa: E402
from continual_manipulation.sequences import TaskSequence  # noqa: E402
from src.support.io import atomic_torch_save, write_json  # noqa: E402

TASK = "button-press-v3"
DEFAULT_PILOT = ROOT / "runs/metaworld_4stage_ddpm_vs_fm_30plus30/ddpm/seed_0"
DEFAULT_OUTPUT = ROOT / "runs/diagnostics/button_press_bc_mse_vs_nll_seed0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HistoricalDeterministicPolicy(nn.Module):
    """Exact isolated reconstruction of the historical general_policy_v2 network."""

    def __init__(self, action_low: np.ndarray, action_high: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        layers: list[nn.Module] = [
            nn.Linear(39, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        ]
        for input_size, output_size in zip(
            GENERAL_POLICY_HIDDEN_SIZES, GENERAL_POLICY_HIDDEN_SIZES[1:]
        ):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        layers.extend((nn.Linear(256, 4), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = self.network(observations)
        return self.action_low + (normalized + 1) * 0.5 * (
            self.action_high - self.action_low
        )

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        del deterministic
        value = torch.as_tensor(observation, dtype=torch.float32)
        return self(value).cpu().numpy().copy()


def tensor_dataset(group: torch.Tensor, ids: list[int]) -> TensorDataset:
    return bc_dataset(group[ids])


def parameter_norm(parameters) -> float:
    squared = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return squared**0.5


def full_mse(policy, dataset: TensorDataset) -> float:
    with torch.no_grad():
        return float(nn.functional.mse_loss(policy(dataset.tensors[0]), dataset.tensors[1]))


def stochastic_metrics(policy: GeneralPolicy, dataset: TensorDataset) -> dict[str, float]:
    observations, actions = dataset.tensors
    with torch.no_grad():
        prediction = policy(observations)
        nll = -policy.normalized_log_prob(observations, actions).mean()
        _, log_std = policy.distribution_parameters(observations)
        flat = log_std.flatten()
        quantiles = torch.quantile(
            flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        )
    return {
        "nll": float(nll),
        "physical_action_mse": float(nn.functional.mse_loss(prediction, actions)),
        "mean_log_std": float(flat.mean()),
        "median_log_std": float(quantiles[2]),
        "min_log_std": float(flat.min()),
        "max_log_std": float(flat.max()),
        "log_std_q10": float(quantiles[0]),
        "log_std_q25": float(quantiles[1]),
        "log_std_q75": float(quantiles[3]),
        "log_std_q90": float(quantiles[4]),
        "lower_clamp_fraction": float((flat == -20).float().mean()),
        "upper_clamp_fraction": float((flat == 2).float().mean()),
        "mean_action_std": float(flat.exp().mean()),
    }


def train_deterministic(policy, train, validation, config):
    loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["training_seed"]),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"])
    history = []
    best_epoch, best_mse, best_state = 0, float("inf"), None
    for epoch in range(1, config["epochs"] + 1):
        policy.train()
        gradients = []
        for observations, actions in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(policy(observations), actions)
            loss.backward()
            gradients.append(parameter_norm(policy.parameters()))
            optimizer.step()
        policy.eval()
        record = {
            "epoch": epoch,
            "train_mse": full_mse(policy, train),
            "validation_mse": full_mse(policy, validation),
            "gradient_norm_mean_before_clipping": float(np.mean(gradients)),
        }
        history.append(record)
        if record["validation_mse"] < best_mse:
            best_epoch, best_mse = epoch, record["validation_mse"]
            best_state = deepcopy(policy.state_dict())
    return history, best_epoch, best_state


def train_stochastic(policy, train, validation, config):
    loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["training_seed"]),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"])
    history = []
    best_nll = (float("inf"), 0, None)
    best_mse = (float("inf"), 0, None)
    for epoch in range(1, config["epochs"] + 1):
        policy.train()
        head_mean_norms, head_std_norms, trunk_norms = [], [], []
        for observations, actions in loader:
            optimizer.zero_grad()
            loss = -policy.normalized_log_prob(observations, actions).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite NLL at epoch {epoch}.")
            loss.backward()
            head_mean_norms.append(parameter_norm(policy.mean_head.parameters()))
            head_std_norms.append(parameter_norm(policy.log_std_head.parameters()))
            trunk_norms.append(parameter_norm(policy.shared_network.parameters()))
            optimizer.step()
        policy.eval()
        train_metrics = stochastic_metrics(policy, train)
        validation_metrics = stochastic_metrics(policy, validation)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "gradients": {
                "measurement": "mean minibatch L2 norm after backward, before optimizer step; no clipping",
                "mean_head": float(np.mean(head_mean_norms)),
                "log_std_head": float(np.mean(head_std_norms)),
                "shared_trunk": float(np.mean(trunk_norms)),
            },
        }
        history.append(record)
        if validation_metrics["nll"] < best_nll[0]:
            best_nll = (validation_metrics["nll"], epoch, deepcopy(policy.state_dict()))
        if validation_metrics["physical_action_mse"] < best_mse[0]:
            best_mse = (
                validation_metrics["physical_action_mse"],
                epoch,
                deepcopy(policy.state_dict()),
            )
    return history, best_nll, best_mse, deepcopy(policy.state_dict())


def stochastic_from_state(low, high, state):
    policy = GeneralPolicy(low, high)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def metric_at(history, epoch):
    return history[epoch - 1]["validation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    pilot, output = args.pilot.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")

    resolved = json.loads((pilot / "config.json").read_text())
    stage = pilot / "stage_0_button-press-v3"
    report = json.loads((stage / "report.json").read_text())
    dataset_path = stage / "data_task_0.npy"
    packed_array = np.load(dataset_path)
    if packed_array.shape != (30, 200, 43):
        raise ValueError(f"Unexpected persisted dataset shape: {packed_array.shape}")
    packed = torch.from_numpy(packed_array).float()
    split = report["splits"][TASK]
    train_ids, validation_ids = split["train_ids"], split["validation_ids"]
    train, validation = tensor_dataset(packed, train_ids), tensor_dataset(packed, validation_ids)

    pilot_policy = GeneralPolicy.load(pilot / Path(report["policy_checkpoint"]).name)
    low, high = pilot_policy.action_low.numpy(), pilot_policy.action_high.numpy()
    config = {
        "kind": "button_press_bc_mse_vs_nll_v1",
        "task": TASK,
        "backend": resolved["environment_backend"],
        "training_seed": resolved["seed"],
        "epochs": resolved["bc"]["epochs"],
        "batch_size": resolved["bc"]["batch_size"],
        "learning_rate": resolved["bc"]["learning_rate"],
        "evaluation_episodes": resolved["eval_episodes"],
        "evaluation_seed": resolved["eval_seed"],
        "evaluation_mode": resolved["evaluation_mode"],
        "task_set_seed": resolved["task_set_seed"],
        "reward_function_version": "v2",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": file_sha256(dataset_path),
        "dataset_shape": list(packed_array.shape),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "action_low": low.tolist(),
        "action_high": high.tolist(),
        "stochastic_objective": resolved["bc"]["objective"],
        "deterministic_objective": "post-tanh-mse",
        "optimizer": "Adam; fresh for each independent job; no gradient clipping",
    }
    if config["dataset_sha256"] != "a8a80902b0f40d942317b1e70c4eb308dcb67ed210236cc4973d7660d9c3be82":
        raise ValueError("Stage-1 dataset identity does not match the audited pilot.")
    if (train_ids, validation_ids) != (
        [28, 16, 23, 6, 18, 25, 3, 29, 8, 0, 19, 12, 20, 13, 7, 5, 17, 14, 22, 9, 27, 24, 1, 15],
        [2, 11, 26, 21, 10, 4],
    ):
        raise ValueError("Stage-1 split identity does not match the audited pilot.")

    torch.manual_seed(config["training_seed"])
    deterministic = HistoricalDeterministicPolicy(low, high)
    torch.manual_seed(config["training_seed"])
    stochastic = GeneralPolicy(low, high)
    deterministic_linears = [m for m in deterministic.network if isinstance(m, nn.Linear)]
    stochastic_linears = [m for m in stochastic.shared_network if isinstance(m, nn.Linear)] + [stochastic.mean_head]
    common_initialization_equal = all(
        torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
        for a, b in zip(deterministic_linears, stochastic_linears)
    ) and torch.equal(deterministic.network[1].weight, stochastic.shared_network[1].weight)
    config["common_mean_layer_initialization_exact"] = common_initialization_equal
    config["initialization_note"] = (
        "Both policies were independently initialized after torch.manual_seed(0); all shared "
        "trunk and deterministic/mean-head tensors are exactly equal. The stochastic log-std "
        "head is additional, zero-weight initialized with bias -3."
    )
    if not common_initialization_equal:
        raise RuntimeError("Expected common mean-path initialization was not reproduced.")

    deterministic_history, a_epoch, a_state = train_deterministic(
        deterministic, train, validation, config
    )
    stochastic_history, best_nll, best_mse, final_state = train_stochastic(
        stochastic, train, validation, config
    )

    deterministic.load_state_dict(a_state)
    deterministic.eval()
    b1 = stochastic_from_state(low, high, best_nll[2])
    b2 = stochastic_from_state(low, high, best_mse[2])
    sequence = TaskSequence.from_names([TASK])
    evaluations = {}
    for name, policy in (("deterministic_mse_best", deterministic), ("stochastic_nll_best", b1), ("stochastic_mse_best", b2)):
        evaluations[name] = evaluate_task(
            policy,
            sequence.task(0),
            episodes=config["evaluation_episodes"],
            seed=config["evaluation_seed"],
            evaluation_mode=config["evaluation_mode"],
            task_set_seed=config["task_set_seed"],
            reward_function_version=config["reward_function_version"],
            backend=config["backend"],
            progress=args.progress,
        )

    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", config)
    write_json(output / "deterministic_history.json", deterministic_history)
    write_json(output / "stochastic_history.json", stochastic_history)
    atomic_torch_save(output / "policy_A_deterministic_mse_best.pt", {
        "kind": "diagnostic_historical_general_policy_v2_reconstruction",
        "epoch": a_epoch,
        "state_dict": a_state,
        "action_low": torch.from_numpy(low.copy()),
        "action_high": torch.from_numpy(high.copy()),
    })
    b1.save(output / "policy_B1_stochastic_nll_best.pt", metadata={"epoch": best_nll[1], "selection": "validation_nll"})
    b2.save(output / "policy_B2_stochastic_mse_best.pt", metadata={"epoch": best_mse[1], "selection": "validation_physical_action_mse"})
    atomic_torch_save(output / "stochastic_final_epoch_observational.pt", {
        "kind": "diagnostic_observational_final_state",
        "epoch": config["epochs"],
        "state_dict": final_state,
    })
    write_json(output / "evaluation_results.json", evaluations)

    a_validation_mse = deterministic_history[a_epoch - 1]["validation_mse"]
    comparison = {
        "A_deterministic_mse_best": {
            "selected_epoch": a_epoch,
            "validation_physical_mse": a_validation_mse,
            "validation_nll": None,
            **{k: evaluations["deterministic_mse_best"][k] for k in ("success_rate", "success_count", "mean_return", "std_return", "mean_episode_length")},
        },
        "B1_stochastic_nll_best": {
            "selected_epoch": best_nll[1],
            **metric_at(stochastic_history, best_nll[1]),
            **{k: evaluations["stochastic_nll_best"][k] for k in ("success_rate", "success_count", "mean_return", "std_return", "mean_episode_length")},
        },
        "B2_stochastic_mse_best": {
            "selected_epoch": best_mse[1],
            **metric_at(stochastic_history, best_mse[1]),
            **{k: evaluations["stochastic_mse_best"][k] for k in ("success_rate", "success_count", "mean_return", "std_return", "mean_episode_length")},
        },
        "final_epoch": {"epoch": config["epochs"], **stochastic_history[-1]["validation"]},
    }
    write_json(output / "comparison_report.json", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
