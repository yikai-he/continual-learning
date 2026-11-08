---
status: archived
original_date: 2026-10-30
archived_on: 2026-10-30
last_verified: unknown
superseded_by: docs/archive/README.md
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# Historical Experiment Reports

This directory preserves lightweight scientific records extracted before the
large artifact cleanup on 2026-10-30. It contains **838 extracted files totaling
47,145,947 bytes**, plus project notes and earlier cleanup reports.

## Organization

- `records/runs/reward_ablations/`: reward and retention ablations.
- `records/runs/exploratory/`: KUKA push-cube curricula, initialization,
  reward, and stability investigations.
- `records/runs/pilots/`, `flow_matching_*`, and `validation/`: early DiffCRL,
  Flow Matching replay, replay-quality, and normalization experiments.
- `records/runs/hammer_reward_ablation_250k/`: Hammer reward comparison.
- `records/runs/kuka_diffcrl_5task.failed-goal-bounds-2026-10-09/`: negative
  evidence from the KUKA goal-bounds failure.
- `records/runs/diagnostics/`, `records/artifacts/`, and
  `records/experiments/`: trajectory, geometry, retention, and environment
  diagnostics.
- `records/runs/reward_v2/`, `reward_v3/`, `dev/`, and `kuka_experts/`:
  historical SAC configurations, evaluations, and provenance.
- `records/continual-learning-archive/`: lightweight records recovered from
  the deleted external archive.
- `project_notes/`: narrative experiment reports formerly in the main docs.
- `previous_cleanup/`: superseded audit reports retained for provenance.

Paths beneath `records/` preserve their original location. Large policies,
replay buffers, trajectory arrays, TensorBoard events, videos, and intermediate
checkpoints were deliberately deleted. These reports preserve results and
research decisions but generally **do not permit exact reproduction**.

Disposable smoke, temporary debug, duplicate, and visualization-only runs were
deleted without archival copies. Canonical results remain under `runs/`.

