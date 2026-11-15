#!/usr/bin/env python3
"""Benchmark one formal Flow Matching backbone in an isolated process.

The benchmark uses deterministic synthetic normalized trajectories. It measures
the established CPU-input training path and public sampling API, not environment
collection or end-to-end continual learning.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
)  # noqa: E402


FORMAL_CONFIGS = {
    "temporal_cnn": Path(
        "configs/diffcrl/replay_budget_ablation/"
        "kuka_4task_fm_mse_100real_100replay_seed0.yaml"
    ),
    "temporal_unet1d": Path(
        "configs/diffcrl/replay_budget_ablation/"
        "kuka_4task_fm_unet1d_w32_mse_100real_100replay_seed0.yaml"
    ),
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(call, device: torch.device) -> float:
    synchronize(device)
    started = time.perf_counter()
    call()
    synchronize(device)
    return time.perf_counter() - started


def memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=tuple(FORMAL_CONFIGS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.sample_batch_size < 1:
        parser.error("warmup must be nonnegative and repeat/batch counts positive")

    source = FORMAL_CONFIGS[args.backbone]
    experiment = load_config(source, expected_experiment="diffcrl")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    flow = experiment.flow_matching
    model_config = FlowMatchingConfig(
        horizon=200,
        features=43,
        backbone=flow.backbone,
        width=flow.width,
        num_tasks=len(experiment.continual.tasks),
        integration_steps=flow.integration_steps,
        time_embedding_scale=flow.time_embedding_scale,
    )
    if model_config.backbone != args.backbone:
        raise ValueError("Selected formal config has a different backbone.")

    torch.manual_seed(experiment.runtime.seed)
    model = TrajectoryFlowMatching(model_config)
    model.velocity_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=flow.learning_rate)

    # A local CPU generator makes the synthetic workload identical across the
    # two independent benchmark processes.
    inputs = torch.Generator().manual_seed(20261113)
    train_batch = torch.randn(
        (flow.batch_size, model_config.horizon, model_config.features),
        generator=inputs,
    )
    train_tasks = torch.arange(flow.batch_size, dtype=torch.long) % model_config.num_tasks
    train_goals = torch.randn((flow.batch_size, 3), generator=inputs)
    sample_tasks = (
        torch.arange(args.sample_batch_size, dtype=torch.long) % model_config.num_tasks
    )
    sample_goals = torch.randn((args.sample_batch_size, 3), generator=inputs)
    loss_rng = torch.Generator().manual_seed(20261114)

    def training_step() -> None:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(
            train_batch,
            generator=loss_rng,
            task_ids=train_tasks,
            goals=train_goals,
        )
        loss.backward()
        optimizer.step()

    def sampling() -> None:
        result = model.sample(
            args.sample_batch_size,
            seed=20261115,
            task_ids=sample_tasks,
            goals=sample_goals,
            integration_steps=flow.integration_steps,
            solver=flow.ode_solver,
            rtol=flow.rtol,
            atol=flow.atol,
        )
        if result.device.type != "cpu":
            raise RuntimeError("Public sampling result must remain on CPU.")

    for _ in range(args.warmup):
        training_step()
        sampling()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_seconds = [timed(training_step, device) for _ in range(args.repeats)]
    training_memory = memory(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampling_seconds = [timed(sampling, device) for _ in range(args.repeats)]
    sampling_memory = memory(device)
    sample_median = statistics.median(sampling_seconds)

    print(
        json.dumps(
            {
                "backbone": model_config.backbone,
                "source_config": str(source),
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "dtype": "float32",
                "parameters": sum(value.numel() for value in model.parameters()),
                "horizon": model_config.horizon,
                "features": model_config.features,
                "num_tasks": model_config.num_tasks,
                "width": model_config.width,
                "solver": flow.ode_solver,
                "integration_steps": flow.integration_steps,
                "actual_nfe": model.last_sample_nfe,
                "training_batch_size": flow.batch_size,
                "sampling_batch_size": args.sample_batch_size,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "training_median_seconds": statistics.median(training_seconds),
                "sampling_median_seconds": sample_median,
                "sampling_trajectories_per_second": (
                    args.sample_batch_size / sample_median
                ),
                "training_memory": training_memory,
                "sampling_memory": sampling_memory,
                "synthetic_input_seed": 20261113,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
