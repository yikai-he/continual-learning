---
status: archived
original_date: 2026-10-30
archived_on: 2026-11-07
last_verified: unknown
superseded_by: docs/RUNS_INVENTORY.md
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

Archived because it records a completed repository-maintenance event rather than current operating guidance.

# Large Artifact Cleanup Report

Cleanup date: 2026-10-30.

| Metric | Value |
|---|---:|
| Repository before cleanup | 57,818,268,109 bytes |
| Repository after cleanup | 931,240,761 bytes |
| Repository reduction | 56,887,027,348 bytes |
| External archive deleted | 12,213,510,782 bytes |
| Inventoried source data deleted | 69,231,151,654 bytes |
| Lightweight records extracted first | 838 files / 47,145,947 bytes |
| Original SAC ZIP checkpoints | 640 |
| Canonical SAC policies retained | 19 |
| Redundant original SAC ZIP checkpoints removed | 621 |
| Canonical continual-learning `.pt` checkpoints retained | 71 |

The source-data figure counts the exact 36 inventoried trees before deletion.
Extracted reports duplicate a small subset, so net storage reduction is
slightly smaller. Filesystem free space rose from about 14 GiB to 78 GiB.

Kept intact: `runs/experts/`, `runs/formal/`, and `runs/formal_40traj/`.
Reports were extracted before deleting reward runs, ablations, exploratory and
development runs, pilots, validation, diagnostics, and KUKA failure evidence.
Disposable smoke, pipeline, visualization, rendered-video, and temporary trees
were deleted completely. The prior sibling archive was verified, its
lightweight records were extracted, and it was permanently deleted.

The cleanup did not alter algorithms, losses, normalization, replay behavior,
hyperparameters, trajectory formats, or evaluation semantics. Formal five-task
DiffCRL/no-replay evidence and supported resume artifacts remain. Historical
reports are result records only where large checkpoints/data were removed.
