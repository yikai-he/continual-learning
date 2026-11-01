# Canonical Runs Inventory

Updated 2026-10-30 after large-artifact cleanup. Only three run families are
maintained:

- `runs/experts/`: one validated successful SAC policy for each backend/task
  with a successful candidate. `expert_registry.json` is authoritative.
- `runs/formal/`: canonical five-task, 200-trajectory DiffCRL and matched
  no-replay experiments for seeds 0, 1, and 2, plus an earlier seed-0 pair.
- `runs/formal_40traj/`: canonical 40-trajectory DiffCRL/no-replay comparison.

Formal directories retain configurations, evaluation matrices, stage-wise
GeneralPolicy and diffusion checkpoints, completion markers, manifests,
datasets, and provenance required by supported resume and analysis workflows.
Historical records are indexed in `docs/archive/experiments/README.md`; their
deleted checkpoints and datasets are not recoverable from this repository.

No verified successful policy was found for KUKA pick-place, KUKA push, KUKA
push-cube, MetaWorld pick-place, MetaWorld push, MetaWorld push-wall, or
MetaWorld shelf-place. Failed checkpoints were not promoted as experts.
