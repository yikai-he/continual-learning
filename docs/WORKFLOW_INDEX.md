---
status: current
last_verified: 2026-11-07
applies_to: continual-learning
---

# Workflow and Configuration Index

This index distinguishes supported entry points from research-specific and
historical material. It does not change the status or contents of any
experiment artifact.

Use [Development status](DEVELOPMENT_STATUS.md) for capability maturity and
[Architecture](ARCHITECTURE.md) for implementation contracts.

## Active workflows

| Workflow | Entry point | Primary configurations |
|---|---|---|
| Single-task SAC experts | `train_sac.py` | `configs/sac/sac.yaml`, formal task presets, and matching smoke presets |
| DiffCRL (`diffusion`, `flow-matching`, `real`, or `none`) | `train_diffcrl.py` | `configs/diffcrl/` |
| Sequential SAC baseline | `train_continual_sac.py` | `configs/continual_sac/` |
| Matched replay comparison | `run_replay_comparison.py` | `configs/diffcrl/policy_replay_comparison_*.yaml` |
| Expert trajectory collection | `scripts/collection/collect_expert_trajectories.py` | A dedicated `experiment: trajectory_collection` YAML passed by the caller |
| Policy evaluation | `scripts/evaluation/evaluate_sac.py`, `scripts/evaluation/evaluate_continual.py` | Resolved run configuration and checkpoint |
| Completed-stage backup/restore | `manage_diffcrl_sync.py` | Existing DiffCRL run metadata |

`configs/local/` contains machine-specific smoke, sweep, and trajectory
collection settings and is not a canonical experiment source.

## Historical and reproducibility material

- `configs/archive/` contains thesis configurations retained for provenance.
- `configs/sac/archive/` contains short probes, reward ablations, and historical
  controls. These should not be used as new defaults.
- `docs/archive/experiments/` contains lightweight reports and records from
  completed or removed experiments.
- `docs/RUNS_INVENTORY.md` identifies retained active artifacts; cleanup history
  is archived under `docs/archive/maintenance/`.

Historical configurations, reports, task seeds, and provenance records should
remain immutable. New experiments should start from an active configuration or
an explicitly documented experimental preset.
