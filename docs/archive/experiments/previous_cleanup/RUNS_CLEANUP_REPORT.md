---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# Runs Cleanup Report

## Summary

The cleanup audited all 2,022 original files under `runs/`, hashed every file
with SHA-256, inspected experiment configs, manifests, summaries, evaluation
matrices, checkpoint exports, source references, active-process state, and
write activity.

| Metric | Before | After |
|---|---:|---:|
| Config-bearing run roots | 106 | 94 |
| Files | 2,022 | 1,768 |
| File bytes under `runs/` | 27,167,583,742 | 15,705,397,216 |
| Displayed size (`du -sh`) | 26 GiB | 15 GiB |

The live tree was reduced by **11,462,186,526 bytes (10.676 GiB)**. The data
is recoverable from `.runs_quarantine_20261011/`; a second verified compressed
archive occupies 751,322,549 bytes. Consequently this is logical space removed
from the active `runs/` tree, not yet an equivalent filesystem-space saving.

## Original structure

The original tree mixed six kinds of artifacts at the top level:

- formal DiffCRL and no-replay thesis experiments;
- exported SAC experts and their original development source runs;
- SAC reward-version runs and task probes;
- ablations and controlled stability experiments;
- smoke, preflight, validation, and diagnostic runs;
- failed or abandoned runs retaining `RUNNING` markers.

The existing `formal/`, `experts/`, `diagnostics/`, `validation/`, `controlled/`,
and `dev/` families were already meaningful. A wholesale rename would have
broken archived configs and provenance, so cleanup retained that organization
and removed redundant run roots instead of adding another nesting layer.

## Retained essential experiments

- `runs/formal/`: all 200-trajectory DiffCRL and no-replay seeds 0, 1, and 2,
  plus the earlier formal seed-0 comparison.
- `runs/formal_40traj/`: the canonical DiffCRL/no-replay seed-0 comparison.
- `runs/experts/kuka-v3/`: ten validated KUKA SAC experts, including model,
  SHA-256 manifest, source metadata, config, evaluation history, and summary.
- `runs/reward_v2/{button-press,faucet-close,hammer,handle-press-side,window-close}-v3/`:
  SAC experts referenced by the current Sawyer DiffCRL configs.
- `runs/reward_v2/kuka-hammer-v3/seed_0_1m`: validated hammer source run.
- `runs/hammer_reward_ablation_250k/`: original, progress-weight-2, and
  progress-weight-4 ablation results.
- `runs/pilot_layernorm_4task/`: scientifically useful DiffCRL/no-replay pilot.

## Retained development and diagnostic evidence

- All fixed-task and sampled KUKA DiffCRL/no-replay smoke comparisons.
- Hammer-inclusive KUKA smoke comparisons.
- The failed goal-bounds five-task run and the compact pre-task-bank failure,
  retained as evidence for important environment/task-bank bugs.
- `formal_preflight/`, `validation/`, `diagnostics/`, and the SAC stability
  control.
- DiffCRL, continual-SAC, no-replay, and SAC smoke suites whose names identify
  separate refactor/contract validations.
- KUKA push reward-v2/v3 probes, pick-place diagnostics, and the current
  peg-unplug degradation run.
- Every run with an unresolved `RUNNING` marker or plausible resume pair.

## Quarantined artifacts

Twelve run roots totaling 11,436,187,452 bytes by directory accounting were
moved outside `runs/`. They comprise nine completed development source runs
whose selected experts are byte-identical to retained exports, one duplicate
40-trajectory DiffCRL run, one abandoned config-only no-replay directory, and
one redundant failed smoke run. Exact paths, sizes, hashes, and reasons are in
[RUNS_DELETION_MANIFEST.md](RUNS_DELETION_MANIFEST.md).

No unique formal result, random seed, evaluation matrix, expert, meaningful
negative result, or potentially resumable run was quarantined.

## Duplicate analysis

SHA-256 hashing found substantial deliberate duplication:

- exported `expert_model.zip` files duplicate selected source-run checkpoints;
- best, best-success, final, and boundary checkpoints are often identical when
  early stopping occurs at the best step;
- DiffCRL and no-replay runs share initial-task datasets and policies by design;
- smoke runs repeat deterministic fixtures across refactor validations;
- evaluation matrices are duplicated at run root and final-stage level.

These scientifically intentional within-run duplicates were retained. The only
complete duplicate formal run (`formal_40traj/diffcrl_5task/haha`) was
quarantined after confirming identical learned policies, diffusion models,
datasets, and trajectories.

## References and reorganization

`scripts/render_kuka_10_tasks.py` now reads the canonical exported models from
`runs/experts/kuka-v3/<task>/policy.zip` rather than completed development
source directories. Current KUKA DiffCRL configs already used these canonical
exports. Formal and archived config paths were not rewritten, preserving the
recorded provenance of completed experiments.

The remaining top-level organization is:

- `formal/`, `formal_40traj/`: thesis/formal results;
- `experts/`, `kuka_experts/`: canonical and historical expert exports;
- `reward_v2/`, `reward_v3/`: SAC source experiments and reward probes;
- `hammer_reward_ablation_250k/`, `controlled/`: ablations;
- `diagnostics/`, `validation/`, `formal_preflight/`: validation evidence;
- `dev/` and named `*_smoke` families: unresolved development and smoke runs.

## Validation

- All ten KUKA expert `SHA256SUMS` checks passed.
- All 478 retained JSON files parse successfully.
- All 157 retained NPZ files open and expose their member lists.
- All five KUKA experts required by the current five-task pipeline resolve.
- All five Sawyer experts required by current DiffCRL configs resolve.
- The recovery archive passed `zstd -t` and contains 327 entries.
- No active training/evaluation process was visible and no file had been
  modified within the 15-minute safety window before cleanup.
- Targeted tests: 90 passed. One unrelated pre-existing assertion in
  `test_render_kuka_push_policy.py` fails because the renderer does not contain
  the exact source substring `finally:\n        env.close()`.

## Remaining uncertainty

Eight retained run roots still contain stale `RUNNING` markers. They were not
changed because their resume or diagnostic value is uncertain:

- three old KUKA development hammer/reach runs;
- the KUKA five-task failed-goal-bounds run;
- the compact pre-task-bank failure;
- incomplete top-down-button, peg-unplug, and reward-v3 push runs.

Several smoke suites contain byte-identical fixtures but were retained because
their named purpose documents separate code-contract validations. They can be
pruned later if that development history is no longer needed.
