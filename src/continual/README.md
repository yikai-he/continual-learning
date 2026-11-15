---
status: current
last_verified: 2026-11-14
applies_to: continual-learning
---

# Continual package architecture

Modules are grouped below by architectural responsibility.

## Foundational contracts

Algorithm-independent task, environment, rollout, evaluation, and metric
contracts now live in the installable package under
`packages/continual-manipulation/`. Algorithm and replay implementations remain
in this package.

- `policy_adapter.py` — Implements the SB3 SAC adapter to the benchmark policy interface.

## Data and task services

Trajectory acquisition, direct 43D preparation, and task provenance.

- `collector.py` — Collects validated episodes through the common policy interface.
- `diffusion_data.py` — Packs physical 43D trajectories, transforms configured
  action coordinates, and reports generated-trajectory diagnostics.
- `task_bank.py` — Implements replay-specific bank state and goal reconstruction
  using the benchmark's canonical task-bank identities and hashes.

## Learning and replay

Trainable policies, generative trajectory memory, and replay support.

- `bc_policy.py` — Implements and trains the persistent behavior-cloning policy.
- `trajectory_diffusion.py` — Implements trajectory diffusion training, sampling, and persistence.
- `diffcrl_replay.py` — Generates previous-task replay from the prior diffusion checkpoint.
- `trajectory_flow_matching.py` — Implements Flow Matching training and sampling.
- `generator_training.py`, `generator_normalization.py` — Share fitting, validation, checkpoint, and normalization behavior.
- `replay_allocation.py` — Allocates a fixed historical budget across prior tasks.
- `flow_matching_replay.py` — Generates fresh allocated replay from the prior Flow checkpoint.
- `expert_recollection.py` — Collects fresh deterministic rollouts from frozen
  historical SAC experts using a dedicated reset-seed namespace.
- `stage_dataset.py`, `replay_common.py` — Assemble stage data and validate common replay provenance.

Both generative models consume and produce `(N,T,43)` tensors directly. Task
IDs and normalized 3D goals remain external conditions, while goal coordinates
also remain inside the generated trajectory. Replay preparation never rebuilds
previous states or replaces generated goals.

The active GeneralPolicy supports the controlled `squashed_gaussian_nll` and
`deterministic_mse` objectives. Both use the same four-layer MLP, first-layer
LayerNorm, activation selection, mean head, tanh action bound, and physical
action scaling. The NLL variant adds a state-dependent log-standard-deviation
head; the deterministic variant optimizes physical-action MSE and has no such
head. ReLU is the default. `bc.activation: leaky_relu` selects the Legacy-inspired placement:
Tanh after the first Linear+LayerNorm and LeakyReLU(0.01) after the remaining
three Linear layers. Output heads remain linear; checkpoint loading validates
the activation and objective identities. Both stage-evaluation paths are
deterministic. Historical `general_policy_v2` deterministic checkpoints remain
specified in `docs/architecture/DETERMINISTIC_GENERAL_POLICY.md` and are not
reinterpreted as the new deterministic-MSE variant.
`bc.hidden_dim` supports 256 (default) and 512 while retaining four layers.
The 512 option changes only GeneralPolicy capacity. Legacy-inspired experiment configurations may pair
512 with BC learning rate `1e-4` and 300 epochs without changing defaults.

## Evaluation

Algorithm-specific reports and console output.

- `reporting.py` — Builds DiffCRL reports and prints benchmark-owned evaluation formatting.

## Application orchestration

Top-level workflows that compose the lower layers into complete experiments.

- `diffcrl.py` — Exposes the seven scientific operations of each continual stage.
- `diffcrl_artifacts.py` — Persists stage checkpoints, reports, and completion markers.
