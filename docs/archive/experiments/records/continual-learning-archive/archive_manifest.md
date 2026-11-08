# Continual Learning External Archive Manifest

Created on 2026-10-30. Paths are relative to the `continual-learning/`
repository and its sibling `continual-learning-archive/` directory.

| Original path | Archive path | Category | Reason | Related experiment | Integrity verification |
|---|---|---|---|---|---|
| `.runs_quarantine_20261011/` | `legacy/.runs_quarantine_20261011/` | Legacy experiment recovery tree | A prior audited cleanup had already removed 12 redundant run roots from active `runs/`, but its quarantine was still inside the source repository. Moving it completes that cleanup without deleting research data. | Nine exported KUKA expert source runs, one duplicate 40-trajectory DiffCRL run, one abandoned no-replay directory, and one redundant failed smoke run; exact paths remain in `continual-learning/docs/RUNS_DELETION_MANIFEST.md`. | 254 files; the prior cleanup recorded per-file SHA-256 values. Same-filesystem relocation completed successfully. The verified recovery bundle below contains the same recovery set. |
| `.runs_cleanup_backup_20261011.tar.zst` | `checkpoints/.runs_cleanup_backup_20261011.tar.zst` | Checkpoint/experiment recovery bundle | Verified compressed backup for the prior run cleanup; it is recovery material rather than an active runtime dependency. | Same 12 quarantined run roots. | `zstd -t` passed immediately before relocation. SHA-256: `22cd8aacf1e9775e461fec000f01c9fb39fc0e6b90a5b91bb492e34b53e5c5aa`; size: 751,322,549 bytes. |

Total archived: **255 files**, **12,213,509,075 bytes** (11.38 GiB of
logical file content). The quarantine and compressed bundle intentionally
overlap: one is browsable and the other is an independent recovery copy.

