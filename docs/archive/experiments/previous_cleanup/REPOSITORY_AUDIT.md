---
status: archived
original_date: 2026-10-30
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# Repository Audit

Audit date: 2026-10-30.

## Scope and safety state

The audit covered the complete repository tree, Git state, tracked, untracked
and ignored files, source/config references, entry points, tests,
documentation, run families, large artifacts, disk space, and active
processes. No training, evaluation, or pytest process was running when cleanup
started. The filesystem had approximately 14 GiB free.

The worktree already contained extensive user changes: modified source and
tests, renamed SAC configurations, and untracked Flow Matching and KUKA push
work. Those changes were preserved. No algorithm, hyperparameter, trajectory
representation, normalization, replay behavior, loss, or evaluation semantic
was changed by this cleanup.

## Active structure and entry points

- Root entry points: `train_sac.py`, `train_diffcrl.py`,
  `train_continual_sac.py`, and `manage_diffcrl_sync.py`.
- `src/continual/`: collection, validation, behavioral cloning,
  diffusion/Flow Matching replay, DiffCRL, evaluation, resume, and run state.
- `src/env_adapters/` and `src/envs.py`: MetaWorld v3 and custom KUKA-v3.
- `src/support/`: configs, manifests, datasets, seeds, devices, and metrics.
- `configs/`: maintained canonical, baseline, ablation, smoke, and KUKA
  configurations.
- `scripts/`: collection, evaluation, diagnostics, visualization, experiment,
  and Flow Matching utilities.
- `tests/`: unit, contract, resume, configuration, and optional KUKA coverage.

The proposed target layout is already substantially present. Moving root entry
points or consolidating modules would create interface churn without a cleanup
benefit, so no structural refactor was performed.

## Cleanup summary

| Disposition | Files | Size / note |
|---|---:|---|
| Retained in active repository | 3,106 | Excludes `.git`; includes this audit and the validation report. |
| Archived from active repository | 255 | 12,213,509,075 bytes moved to the sibling archive. |
| Deleted generated clutter | 357 | Python/pytest caches and temporary files only. |
| Retained for review | 648 | 48,638,590,722 bytes; this is a subset of retained files, not an additional total. |

## Classification

### KEEP

- Canonical five-task DiffCRL and matched no-replay results under
  `runs/formal/` and `runs/formal_40traj/`.
- Current Sawyer experts under `runs/reward_v2/`, and canonical KUKA exports
  under `runs/experts/kuka-v3/`.
- Continual SAC, SAC, DiffCRL/no-replay, Flow Matching, validation, preflight,
  and diagnostic outputs whose distinct purpose is in `RUNS_INVENTORY.md`.
- Maintained source, entry points, configs, tests, scripts, and documentation.
  Static non-import is not proof that a command-line script is obsolete.

### ARCHIVE

The already-audited `.runs_quarantine_20261011/` and its verified compressed
backup were moved to the sibling `continual-learning-archive/`. Its manifest
records the exact mapping and integrity evidence.

### DELETE

357 generated files were removed from Python `__pycache__` directories,
`.pytest_cache`, and verified editor/tool temporary patterns. They are fully
regenerable and contained no experiment data.

### REVIEW

648 files (48,638,590,722 bytes) remain in this non-overlapping review set:

- `runs/dev/`: development runs have unresolved provenance and historically
  included `RUNNING`/resume artifacts.
- `runs/reward_ablations/`: current configs, tests, and documents refer to
  these outputs.
- `runs/exploratory/`: current KUKA push-cube configs and a diagnostic default
  path refer to it, so moving it now would break workflows.
- `artifacts/` and `rendered_videos/`: outputs may be regenerable, but
  uniqueness and publication value have not been established.
- `configs/archive/` and `configs/sac/archive/`: historical configs are small
  and reproducibility-relevant and should move only with a stable link policy.

## Experiment audit

The primary thesis evidence is the multi-seed five-task DiffCRL versus
no-replay comparison in `runs/formal/diffcrl_5task_200traj/` and
`runs/formal/diffcrl_5task_no_replay_200traj/`, with the 40-trajectory matched
comparison retained separately. `configs/archive/thesis_5task/` preserves the
exact formal presets. Current configs in `configs/diffcrl/` continue to refer
to five Sawyer expert checkpoints in `runs/reward_v2/`.

KUKA canonical experts are exported under `runs/experts/kuka-v3/` with
manifests and checksums. Reward ablations, push-cube curricula, Flow Matching
pilots, failed goal-bounds runs, and peg-unplug degradation evidence are not
duplicates of the primary result and remain identifiable as baselines,
negative results, or active investigations.

## Dependency and reproducibility risks

- Many configs use repository-relative checkpoint and output paths. Run from
  the repository root to avoid resolving them incorrectly.
- KUKA execution depends on the separate custom MetaWorld fork and Conda
  environment; it is not self-contained here.
- Large ignored artifacts are essential to canonical configs but are not in
  Git history. Back up manifests/checksums and the external archive with Git.
- Exploratory and reward-ablation paths are coupled to untracked configs and
  tests. Reclassify them only after those experiments are finalized.
- The substantial pre-existing dirty worktree should receive a clean baseline
  commit before later aggressive reorganization.

