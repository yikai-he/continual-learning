# Continual package architecture

Modules are grouped below by architectural responsibility.

## Foundational contracts

Shared interfaces and identities used throughout the package.

- `schema.py` — Defines and validates the canonical collected trajectory.
- `policy_adapter.py` — Defines the common policy interface and the SB3 SAC adapter.
- `task_sequence.py` — Represents an ordered task curriculum and environment switching.

## Data and task services

Trajectory acquisition, representation conversion, and MetaWorld task provenance.

- `collector.py` — Collects validated episodes through the common policy interface.
- `diffusion_data.py` — Packs trajectories and converts between physical and diffusion representations.
- `task_bank.py` — Records MT1 task identities and reconstructs task goals for replay.

## Learning and replay

Trainable policies, generative trajectory memory, and replay support.

- `bc_policy.py` — Implements and trains the persistent behavior-cloning policy.
- `trajectory_diffusion.py` — Implements trajectory diffusion training, sampling, and persistence.
- `diffcrl_replay.py` — Generates previous-task replay from the prior diffusion checkpoint.
- `diagnostics.py` — Reports action-bound and projection diagnostics for generated replay.

## Evaluation

Stage evaluation, continual-learning metrics, and human-readable output.

- `evaluation.py` — Evaluates tasks and maintains the stage-by-task result matrix.
- `metrics.py` — Computes continual-learning metrics from evaluation results.
- `reporting.py` — Formats and prints evaluation and continual-learning reports.

## Application orchestration

Top-level workflows that compose the lower layers into complete experiments.

- `diffcrl.py` — Coordinates expert collection, diffusion replay, BC, and evaluation across stages.
- `sequential_sac.py` — Runs the persistent-SAC continual baseline without cross-task replay.