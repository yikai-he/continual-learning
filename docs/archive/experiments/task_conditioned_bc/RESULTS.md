---
status: archived
original_date: unknown
archived_on: 2026-11-07
last_verified: unknown
superseded_by: docs/DEVELOPMENT_STATUS.md
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

Archived as a completed experimental result rather than current pipeline documentation.

# Task-Conditioned Behavioral Cloning Results

## Experimental setup

This audit compares five-task MetaWorld-v3 continual BC with no replay or DDPM diffusion replay, and with no explicit task input or a five-way one-hot task input. Each condition has training seeds 0, 1, and 2. Success is evaluated on the same fixed bank of 50 environment instances per task after each stage.

Repository definitions are used unchanged: `AP_k = mean(R[k, 0:k+1])`; endpoint forgetting is `F_k = mean(diag(R)[0:k+1] - R[k, 0:k+1])`. Thus negative forgetting means improvement, and the current task contributes zero.

### Authoritative source paths

- **No Replay + None:** `runs/formal/diffcrl_5task_no_replay_200traj/diffcrl_5task_no_replay_200traj_seed{seed}`
- **No Replay + One-hot:** `runs/formal/no_replay_5task_task_conditioned/seed_{seed}`
- **DiffCRL + None:** `runs/formal/diffcrl_5task_200traj/diffcrl_5task_200traj_seed{seed}`
- **DiffCRL + One-hot:** `runs/formal/diffcrl_5task_task_conditioned/seed_{seed}`

## Integrity verification

- PASS: All 12 run directories, root matrices, and five stage reports are present.
- PASS: Every lower-triangular matrix has exactly 15 valid measured cells and 10 unmeasured future-task cells.
- PASS: Task order and fixed evaluation task-bank identities match across all runs.
- PASS: Evaluation protocol matches: fixed-tasks, 50 episodes/cell, reset seed 10000, task-bank seed 10000.
- PASS: All final checkpoints report the expected conditioning mode; one-hot checkpoints record the full task order.
- **FAIL — conditioning-only configuration match:** the complete three-seed task-agnostic baselines are legacy two-hidden-layer runs, while the one-hot runs use four hidden layers. Their saved expert paths also differ; the legacy baseline paths no longer resolve in this checkout, so checkpoint identity cannot be compared. Consequently all conditioning effects and the replay × conditioning interaction below are descriptive, not controlled causal estimates.

Per-seed discrepancies:

- No Replay seed 0: task-agnostic policy has hidden sizes (256, 256), but one-hot has (256, 256, 256, 256); saved expert paths also differ and the legacy baseline expert files are not present at those paths.
- No Replay seed 1: task-agnostic policy has hidden sizes (256, 256), but one-hot has (256, 256, 256, 256); saved expert paths also differ and the legacy baseline expert files are not present at those paths.
- No Replay seed 2: task-agnostic policy has hidden sizes (256, 256), but one-hot has (256, 256, 256, 256); saved expert paths also differ and the legacy baseline expert files are not present at those paths.
- DiffCRL seed 0: task-agnostic policy has hidden sizes (256, 256), but one-hot has (256, 256, 256, 256); saved expert paths also differ and the legacy baseline expert files are not present at those paths.
- DiffCRL seed 1: task-agnostic policy has hidden sizes (256, 256), but one-hot has (256, 256, 256, 256); saved expert paths also differ and the legacy baseline expert files are not present at those paths.
- DiffCRL seed 2: task-agnostic policy has hidden sizes (256, 256), but one-hot has (256, 256, 256, 256); saved expert paths also differ and the legacy baseline expert files are not present at those paths.

## Main comparison

| Condition | Final AP | Average forgetting |
|---|---:|---:|
| No Replay + None | 29.73 ± 7.21% | 69.87 ± 7.61 pp |
| No Replay + One-hot | 24.40 ± 4.61% | 75.47 ± 4.82 pp |
| DiffCRL + None | 96.80 ± 1.39% | 2.13 ± 2.34 pp |
| DiffCRL + One-hot | 96.27 ± 1.22% | 3.20 ± 1.44 pp |

## Individual seed results

| Condition | Seed | Final AP | Forgetting |
|---|---:|---:|---:|
| No Replay + None | 0 | 22.80% | 77.20 pp |
| No Replay + None | 1 | 29.20% | 70.40 pp |
| No Replay + None | 2 | 37.20% | 62.00 pp |
| No Replay + One-hot | 0 | 24.00% | 76.00 pp |
| No Replay + One-hot | 1 | 29.20% | 70.40 pp |
| No Replay + One-hot | 2 | 20.00% | 80.00 pp |
| DiffCRL + None | 0 | 95.20% | 4.80 pp |
| DiffCRL + None | 1 | 97.60% | 0.40 pp |
| DiffCRL + None | 2 | 97.60% | 1.20 pp |
| DiffCRL + One-hot | 0 | 95.20% | 4.40 pp |
| DiffCRL + One-hot | 1 | 96.00% | 3.60 pp |
| DiffCRL + One-hot | 2 | 97.60% | 1.60 pp |

## Full continual evaluation matrices

Entries are success percentages; `—` denotes intentionally unmeasured future tasks.

### No Replay + None, seed 0

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 2 | 100 | — | — | — |
| 2 | 0 | 0 | 100 | — | — |
| 3 | 34 | 6 | 10 | 100 | — |
| 4 | 0 | 0 | 14 | 0 | 100 |

### No Replay + None, seed 1

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 26 | 100 | — | — | — |
| 2 | 2 | 2 | 100 | — | — |
| 3 | 16 | 0 | 0 | 98 | — |
| 4 | 16 | 30 | 0 | 0 | 100 |

### No Replay + None, seed 2

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 44 | 100 | — | — | — |
| 2 | 0 | 0 | 96 | — | — |
| 3 | 0 | 0 | 0 | 100 | — |
| 4 | 26 | 40 | 16 | 4 | 100 |

### No Replay + One-hot, seed 0

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 12 | 100 | — | — | — |
| 2 | 0 | 0 | 100 | — | — |
| 3 | 2 | 0 | 0 | 100 | — |
| 4 | 0 | 0 | 0 | 20 | 100 |

### No Replay + One-hot, seed 1

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 100 | 100 | — | — | — |
| 2 | 6 | 0 | 98 | — | — |
| 3 | 0 | 0 | 8 | 100 | — |
| 4 | 44 | 0 | 0 | 2 | 100 |

### No Replay + One-hot, seed 2

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 4 | 100 | — | — | — |
| 2 | 0 | 0 | 100 | — | — |
| 3 | 0 | 0 | 4 | 100 | — |
| 4 | 0 | 0 | 0 | 0 | 100 |

### DiffCRL + None, seed 0

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 100 | 100 | — | — | — |
| 2 | 100 | 100 | 100 | — | — |
| 3 | 100 | 100 | 88 | 100 | — |
| 4 | 100 | 100 | 80 | 96 | 100 |

### DiffCRL + None, seed 1

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 100 | 100 | — | — | — |
| 2 | 100 | 100 | 90 | — | — |
| 3 | 100 | 100 | 84 | 100 | — |
| 4 | 100 | 100 | 96 | 92 | 100 |

### DiffCRL + None, seed 2

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 100 | 100 | — | — | — |
| 2 | 100 | 100 | 98 | — | — |
| 3 | 100 | 100 | 94 | 96 | — |
| 4 | 100 | 100 | 88 | 100 | 100 |

### DiffCRL + One-hot, seed 0

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 100 | 100 | — | — | — |
| 2 | 100 | 100 | 98 | — | — |
| 3 | 100 | 100 | 90 | 100 | — |
| 4 | 100 | 100 | 78 | 98 | 100 |

### DiffCRL + One-hot, seed 1

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 100 | 100 | — | — | — |
| 2 | 100 | 100 | 98 | — | — |
| 3 | 100 | 100 | 92 | 100 | — |
| 4 | 100 | 100 | 80 | 100 | 100 |

### DiffCRL + One-hot, seed 2

| Stage | Button | Faucet | Hammer | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| 0 | 100 | — | — | — | — |
| 1 | 100 | 100 | — | — | — |
| 2 | 100 | 100 | 100 | — | — |
| 3 | 100 | 100 | 92 | 96 | — |
| 4 | 100 | 100 | 88 | 100 | 100 |

## Final per-task success

| Condition | Button | Faucet | Hammer | Handle | Window |
|---|---:|---:|---:|---:|---:|
| No Replay + None | 14.00 ± 13.11% | 23.33 ± 20.82% | 10.00 ± 8.72% | 1.33 ± 2.31% | 100.00 ± 0.00% |
| No Replay + One-hot | 14.67 ± 25.40% | 0.00 ± 0.00% | 0.00 ± 0.00% | 7.33 ± 11.02% | 100.00 ± 0.00% |
| DiffCRL + None | 100.00 ± 0.00% | 100.00 ± 0.00% | 88.00 ± 8.00% | 96.00 ± 4.00% | 100.00 ± 0.00% |
| DiffCRL + One-hot | 100.00 ± 0.00% | 100.00 ± 0.00% | 82.00 ± 5.29% | 99.33 ± 1.15% | 100.00 ± 0.00% |

## Stage-wise performance

| Condition | Metric | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---:|---:|---:|---:|---:|
| No Replay + None | AP | 100.00 ± 0.00% | 62.00 ± 10.54% | 33.33 ± 1.33% | 30.33 ± 6.45% | 29.73 ± 7.21% |
| No Replay + None | Forgetting | 0.00 ± 0.00 pp | 38.00 ± 10.54 pp | 66.22 ± 0.77 pp | 69.17 ± 5.97 pp | 69.87 ± 7.61 pp |
| No Replay + One-hot | AP | 100.00 ± 0.00% | 69.33 ± 26.63% | 33.78 ± 0.77% | 26.17 ± 0.76% | 24.40 ± 4.61% |
| No Replay + One-hot | Forgetting | 0.00 ± 0.00 pp | 30.67 ± 26.63 pp | 66.00 ± 1.15 pp | 73.67 ± 1.04 pp | 75.47 ± 4.82 pp |
| DiffCRL + None | AP | 100.00 ± 0.00% | 100.00 ± 0.00% | 98.67 ± 1.76% | 96.83 ± 0.76% | 96.80 ± 1.39% |
| DiffCRL + None | Forgetting | 0.00 ± 0.00 pp | 0.00 ± 0.00 pp | 0.00 ± 0.00 pp | 1.83 ± 1.04 pp | 2.13 ± 2.34 pp |
| DiffCRL + One-hot | AP | 100.00 ± 0.00% | 100.00 ± 0.00% | 99.56 ± 0.38% | 97.50 ± 0.50% | 96.27 ± 1.22% |
| DiffCRL + One-hot | Forgetting | 0.00 ± 0.00 pp | 0.00 ± 0.00 pp | 0.00 ± 0.00 pp | 1.83 ± 0.29 pp | 3.20 ± 1.44 pp |

## Controlled comparisons

Positive AP differences favor the second condition; positive forgetting differences mean more forgetting.

| Contrast | Paired Final AP difference | Paired forgetting difference |
|---|---:|---:|
| Task conditioning, no replay | -5.33 ± 10.29 pp | 5.60 ± 10.76 pp |
| Task conditioning, DiffCRL | -0.53 ± 0.92 pp | 1.07 ± 1.89 pp |
| Replay, task-agnostic | 67.07 ± 6.11 pp | -67.73 ± 6.12 pp |
| Replay, one-hot | 71.87 ± 5.43 pp | -72.27 ± 5.83 pp |

The two replay contrasts compare runs with the same architecture within each conditioning setting, but expert provenance still differs between the legacy task-agnostic and new one-hot suites. The two task-conditioning contrasts additionally change policy depth, so they are not controlled ablations.

## Replay × Task Conditioning interaction

Paired difference-in-differences for Final AP: **4.80 ± 10.74 pp** (seed values: -1.20, -1.60, 17.20 pp).

Paired difference-in-differences for forgetting: **-4.53 ± 11.38 pp** (seed values: 0.80, 3.20, -17.60 pp).

These interactions are descriptive only because task conditioning is confounded with network depth and unverified expert identity.

## Interpretation

- The one-hot no-replay suite ended at 24.40 ± 4.61% AP and 75.47 ± 4.82 pp forgetting. It therefore did not eliminate forgetting in every seed, although its descriptive aggregate is much stronger than the legacy no-replay suite (29.73 ± 7.21% AP).
- Diffusion replay remained descriptively beneficial in both information settings: 67.07 ± 6.11 pp AP without task IDs and 71.87 ± 5.43 pp AP with one-hot task IDs.
- The best aggregate is DiffCRL + One-hot at 96.27 ± 1.22% AP. The task-agnostic DiffCRL suite reached 96.80 ± 1.39%.
- Per-task and matrix tables distinguish acquisition (diagonal cells) from retention (later rows). Any mechanism-level explanations—task ambiguity, parameter interference, or replay quality—remain hypotheses because this experiment measures success outcomes rather than those mechanisms directly.
- Negative transfer is reported only where later success falls below the task's diagonal acquisition value; improvements correspond to negative repository-defined forgetting.

## Limitations and conclusions

All 12 runs are artifact-complete and internally valid, but the four-condition study is **not a valid conditioning-only controlled experiment** against the available baselines. The baseline policy-depth mismatch and unverifiable expert equivalence prevent attributing observed differences solely to task identity. The results support descriptive statements about these saved runs and show that replay remains useful in the one-hot suite; a clean causal conclusion requires rerunning task-agnostic four-layer baselines with the same expert checkpoints and otherwise identical current configurations.

Machine-readable values are provided in `summary.csv`, `per_seed_results.csv`, `per_task_results.csv`, and `stagewise_results.csv`.
