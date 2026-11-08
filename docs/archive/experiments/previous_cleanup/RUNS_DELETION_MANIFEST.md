---
status: archived
original_date: 2026-10-11
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# Runs Cleanup Deletion Manifest

Generated before cleanup on 2026-10-11. All paths are relative to the
`continual-learning` repository. The listed directories must be archived and
the archive verified before their originals are removed.

Recovery archive: `.runs_cleanup_backup_20261011.tar.zst`

Quarantine: `.runs_quarantine_20261011/`

Cleanup status: the originals were moved out of `runs/` into the quarantine.
Permanent deletion was intentionally not performed because the safety review
treated a stale `RUNNING` marker in one failed smoke directory as potentially
resumable. The compressed archive was verified with `zstd -t`; its SHA-256 is
`22cd8aacf1e9775e461fec000f01c9fb39fc0e6b90a5b91bb492e34b53e5c5aa`.

| Original path | Bytes | Classification | Reason and duplicate evidence |
|---|---:|---|---|
| `runs/dev/reward_v2/kuka-handle-press-side-v3/seed_0` | 4,219,441,393 | Redundant | Complete source run. Its selected best-success checkpoint is byte-identical (SHA-256 `090824e5…`) to `runs/experts/kuka-v3/handle-press-side-v3/policy.zip`; exported config, evaluation history, summary, metadata, and manifest are retained. |
| `runs/dev/reward_v2/kuka-button-press-v3/seed_0` | 1,425,021,914 | Redundant | Complete source run; exported expert checkpoint is byte-identical (`ee92f679…`) and its scientific metadata/evaluations are retained. |
| `runs/dev/reward_v2/kuka-faucet-close-v3/seed_0` | 1,075,739,985 | Redundant | Complete source run; exported expert checkpoint is byte-identical (`a3d92008…`) and its scientific metadata/evaluations are retained. |
| `runs/dev/reward_v2/kuka-door-open-v3/seed_0` | 1,075,726,680 | Redundant | Complete source run; exported expert checkpoint is byte-identical (`73d6f429…`) and its scientific metadata/evaluations are retained. |
| `runs/dev/reward_v2/kuka-reach-v3/seed_0_20261008_094205_216332` | 726,450,294 | Redundant | Complete source run; exported expert checkpoint is byte-identical (`2be77e78…`) and its scientific metadata/evaluations are retained. |
| `runs/dev/reward_v2/kuka-faucet-open-v3/seed_0` | 726,431,620 | Redundant | Complete source run; exported expert checkpoint is byte-identical (`2c4ead70…`) and its scientific metadata/evaluations are retained. |
| `runs/dev/reward_v2/kuka-window-open-v3/seed_0` | 726,431,612 | Redundant | Complete source run; exported expert checkpoint is byte-identical (`6f3358cb…`) and its scientific metadata/evaluations are retained. |
| `runs/dev/reward_v2/kuka-window-close-v3/seed_0` | 726,418,392 | Redundant | Complete source run; exported expert checkpoint is byte-identical (`25cd01f2…`) and its scientific metadata/evaluations are retained. |
| `runs/dev/reward_v2/kuka-drawer-close-v3/seed_0` | 726,418,266 | Redundant | Complete source run; exported expert checkpoint is byte-identical (`135c3c07…`) and its scientific metadata/evaluations are retained. |
| `runs/formal_40traj/diffcrl_5task/haha` | 32,287,276 | Redundant | Scientifically equivalent duplicate of `formal_seed0`: all learned policy, diffusion, trajectory, and dataset artifacts are byte-identical. JSON/config differences contain only run identity/path text. |
| `runs/kuka_diffcrl_smoke.failed-2026-10-09` | 1,816,663 | Unnecessary | Abandoned failed smoke run duplicated by the retained sampled smoke artifacts; the smaller, explicitly named `kuka_diffcrl_smoke_failed_pre_taskbank_fix` remains as bug-fix evidence. |
| `runs/formal_40traj/no_replay_5task/haha` | 2,431 | Unnecessary | Abandoned configuration-only directory with no stages, model, trajectories, evaluation matrix, or resumable checkpoint. |

Total candidate size: **11,436,187,452 bytes** (10.651 GiB).

No directory with a recent write, visible active process, or unresolved
`RUNNING` state is included. Stale/incomplete runs remain in place when their
resumability or diagnostic value is uncertain.
