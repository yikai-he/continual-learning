"""Private mechanics shared by DDPM and Flow Matching generator training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from src.config import GENERATOR_GRADIENT_CLIP_NORM, GENERATOR_VALIDATION_DRAWS


def run_cpu_inputs_on_module_device(module: nn.Module, *inputs):
    """Run one module on its device and restore the established CPU output contract."""
    device = next(module.parameters()).device
    moved = tuple(
        value.to(device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )
    return module(*moved).cpu()


@torch.no_grad()
def _assess_generator_loss(model, data, goals, task_ids, noise_seed: int) -> float:
    """Average four deterministic loss draws using a fresh local generator."""
    local = torch.Generator().manual_seed(noise_seed)
    return (
        sum(
            model.loss(data, generator=local, task_ids=task_ids, goals=goals).item()
            for _ in range(GENERATOR_VALIDATION_DRAWS)
        )
        / GENERATOR_VALIDATION_DRAWS
    )


def generator_metrics(
    model,
    train,
    validation,
    train_goals,
    validation_goals,
    train_task_ids,
    validation_task_ids,
    seed: int,
) -> dict[str, float]:
    """Assess train then validation loss with the established seed offsets."""
    return {
        "train_loss": _assess_generator_loss(
            model, train, train_goals, train_task_ids, seed + 1000
        ),
        "validation_loss": _assess_generator_loss(
            model,
            validation,
            validation_goals,
            validation_task_ids,
            seed + 2000,
        ),
    }


def train_generator_epoch(
    model,
    train,
    train_goals,
    train_task_ids,
    *,
    optimizer,
    rng: torch.Generator,
    batch_size: int,
    nonfinite_error: str,
) -> None:
    """Run one identically shuffled, clipped generator-training epoch."""
    model.train()
    for batch in torch.randperm(len(train), generator=rng).split(batch_size):
        optimizer.zero_grad()
        loss = model.loss(
            train[batch],
            generator=rng,
            task_ids=train_task_ids[batch],
            goals=train_goals[batch],
        )
        if not torch.isfinite(loss):
            raise ValueError(nonfinite_error)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GENERATOR_GRADIENT_CLIP_NORM)
        optimizer.step()
    model.eval()


def fit_generator(
    model,
    train,
    validation,
    train_goals,
    validation_goals,
    train_task_ids,
    validation_task_ids,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    progress: bool,
    progress_label: str,
    nonfinite_error: str,
    report_epoch_metrics: bool = False,
) -> dict[str, Any]:
    """Run the common seeded fit lifecycle after model-specific validation."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid fit settings.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = torch.Generator().manual_seed(seed)

    def metrics() -> dict[str, float]:
        return generator_metrics(
            model,
            train,
            validation,
            train_goals,
            validation_goals,
            train_task_ids,
            validation_task_ids,
            seed,
        )

    initial = metrics()
    history = []
    best = float("inf")
    best_state = None
    best_epoch = 0
    with tqdm(
        range(1, epochs + 1),
        desc=progress_label,
        unit="epoch",
        disable=not progress,
    ) as epoch_progress:
        for epoch in epoch_progress:
            train_generator_epoch(
                model,
                train,
                train_goals,
                train_task_ids,
                optimizer=optimizer,
                rng=rng,
                batch_size=batch_size,
                nonfinite_error=nonfinite_error,
            )
            record = {"epoch": epoch, **metrics()}
            history.append(record)
            if record["validation_loss"] < best:
                best = record["validation_loss"]
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
            if report_epoch_metrics:
                epoch_progress.set_postfix(
                    train=f"{record['train_loss']:.5f}",
                    validation=f"{record['validation_loss']:.5f}",
                    best=f"{best:.5f}@{best_epoch}",
                )

    model.load_state_dict(best_state)
    result = {
        "initial": initial,
        "best_epoch": best_epoch,
        "history": history,
        **metrics(),
    }
    if progress and report_epoch_metrics:
        selection_label = progress_label.removesuffix(" training")
        tqdm.write(
            f"{selection_label} selected epoch {best_epoch}/{epochs}: "
            f"validation={result['validation_loss']:.6f}, "
            f"train={result['train_loss']:.6f}"
        )
    return result


def generator_checkpoint_payload(
    *,
    kind: str,
    config: Any,
    state_dict: dict,
    dynamic_normalizer: Any,
    goal_normalizer: Any,
    metadata: dict,
) -> dict:
    """Build the existing ordered checkpoint envelope for either generator."""
    return {
        "kind": kind,
        "config": asdict(config),
        "state_dict": state_dict,
        "dynamic_mean": dynamic_normalizer.mean,
        "dynamic_scale": dynamic_normalizer.scale,
        "goal_mean": goal_normalizer.mean,
        "goal_scale": goal_normalizer.scale,
        "metadata": metadata,
    }
