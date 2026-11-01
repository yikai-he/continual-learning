# Workflow and Configuration Index

This index distinguishes supported entry points from research-specific and
historical material. It does not change the status or contents of any
experiment artifact.

## Active workflows

| Workflow | Entry point | Primary configurations |
|---|---|---|
| Single-task SAC experts | `train_sac.py` | `configs/sac/sac.yaml`, formal task presets, and matching smoke presets |
| DiffCRL (`diffusion`, `flow-matching`, `real`, or `none`) | `train_diffcrl.py` | `configs/diffcrl/` |
| Sequential SAC baseline | `train_continual_sac.py` | `configs/continual_sac/` |
| Matched replay comparison | `run_replay_comparison.py` | `configs/diffcrl/policy_replay_comparison_*.yaml` |
| Expert trajectory collection | `scripts/collection/collect_expert_trajectories.py` | Experiment YAML passed by the caller |
| Policy evaluation | `scripts/evaluation/evaluate_sac.py`, `evaluate_continual.py` | Resolved run configuration and checkpoint |
| Completed-stage backup/restore | `manage_diffcrl_sync.py` | Existing DiffCRL run metadata |

Generic visualization and trajectory inspection live in
`scripts/visualization/` and `scripts/diagnostics/inspect_trajectory.py`.

## Experimental workflows

The following are maintained research utilities rather than general training
entry points:

- `scripts/flow_matching/` and `scripts/experiments/run_flow_matching_sanity.py`
  contain Flow Matching pilots, harder-task validation, and aggregators.
- Task-specific KUKA evaluation and diagnostics live under
  `scripts/evaluation/` and `scripts/diagnostics/`.
- KUKA rendering and scripted-control utilities are the `scripts/render_kuka_*`
  and `scripts/kuka_push_scripted_controller.py` files.
- Pilot, comparison, architecture-ablation, and fixed-budget YAML files remain
  in `configs/diffcrl/` because tests and research records reference them.
- `configs/flow_matching_sanity/` and `configs/flow_matching_harder_task/` are
  study-specific rather than default DiffCRL configurations.
- `configs/local/` contains machine-specific smoke, sweep, and trajectory
  collection settings and is not a canonical experiment source.

`scripts/flow_matching/launch_flow_matching_ablation.sh` is a retained
historical launcher. Its original runner and configuration are absent, and the
script intentionally exits with an explanatory error instead of running.

## Historical and reproducibility material

- `configs/archive/` contains thesis configurations retained for provenance.
- `configs/sac/archive/` contains short probes, reward ablations, and historical
  controls. These should not be used as new defaults.
- `docs/archive/experiments/` contains lightweight reports and records from
  completed or removed experiments.
- `docs/RUNS_INVENTORY.md` identifies retained active artifacts; cleanup history
  is recorded in `docs/ARTIFACT_CLEANUP_REPORT.md`.

Historical configurations, reports, task seeds, and provenance records should
remain immutable. New experiments should start from an active configuration or
an explicitly documented experimental preset.

## Compatibility modules

`src/continual/diagnostics.py` and `src/support/dataset.py` are compatibility
re-exports. Production code uses `src.continual.diffusion_data` and
`src.continual.stage_dataset`, respectively. The shims remain to support older
notebooks and external scripts; tests verify that the old imports resolve to
the canonical implementations.
