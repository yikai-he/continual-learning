---
status: current
last_verified: 2026-11-07
applies_to: continual-learning
---

# Fixed-budget Flow Matching replay

This is the authoritative architecture contract for Flow Matching replay in
DiffCRL. DiffCRL is the overall continual-learning pipeline; Flow Matching and
diffusion are distinct generative models. “Generative replay” describes their
shared role, while this document specifically describes Flow Matching.

## Execution path

For stage `k`, configuration selects the ordered task sequence and current SAC
expert. Before accessing the current task, `replay_allocation.py` divides the
fixed total historical budget among task IDs `0..k-1`. The previous completed
Flow checkpoint then generates that allocation from scratch, preserving task
labels, reconstructed goals, deterministic seeds, trajectory validation, and
checkpoint hashes. The current expert trajectories are collected separately.

The historical and current groups are split as whole episodes and combined.
The same combined training and validation tensors update the Flow model and the
persistent GeneralPolicy through behavior cloning. Model weights carry across
stages; the existing per-fit optimizer behavior is unchanged. Evaluation,
reports, immutable checkpoints, and the stage completion marker follow.

## Configuration and allocation

```yaml
continual:
  trajectories_per_task: 200
  replay_mode: flow-matching
  replay_budget: 200
flow_matching:
  ode_solver: euler
  integration_steps: 100
```

`trajectories_per_task` is the current expert collection count.
`replay_budget` is the total number of historical synthetic trajectories at
each nonzero stage. Current expert trajectories are excluded. Uniform
allocation uses integer division and assigns the remainder in curriculum
order: 200 across three tasks is `67, 67, 66`. If the budget is smaller than
the number of historical tasks, later task IDs receive zero. Allocations are
recorded in stage reports. Unsupported strategies fail validation; there is no
silent fallback.

The sampler supports the solver names accepted by `FlowMatchingConfig`;
canonical fixed-budget configurations use fixed-step Euler integration. Solver,
step count, tolerances, and neural-function-evaluation metadata are recorded in
replay provenance.

`allocate_replay` is independent of environments, models, generation, and
training. A future prioritization method can extend its strategy input once a
scientifically specified score contract exists.

## Reproducibility and resume

Replay comes only from the previous completed Flow checkpoint. Sampling seeds
remain a deterministic function of run seed, stage, and task ID. Reports record
the checkpoint identity, per-task allocation, sampled task configurations, and
generation diagnostics. Resume validates completed marker hashes and restores
Flow and policy weights, task-bank metadata, evaluation state, and RNG state.
It regenerates replay for the next stage using the same allocation and source
checkpoint. Persistent synthetic snapshots are not part of the architecture.

## Scientific limitations

A fixed total budget controls historical data quantity as the curriculum grows,
but each individual task receives fewer trajectories at later stages. It does
not equalize generation cost perfectly when trajectory difficulty differs.
Comparisons with older per-task-budget runs confound allocation with total data
and training samples; they must not be interpreted as replay-method-only
comparisons. Unit and smoke tests establish software contracts, not scientific
effectiveness.
