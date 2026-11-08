---
status: archived
original_date: unknown
archived_on: 2026-11-07
last_verified: unknown
superseded_by: docs/FLOW_REPLAY_ARCHITECTURE.md
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

Archived as a completed integration-step experiment record after removal of its dedicated study tooling.

# Flow Matching Integration Steps: Continual-Learning Results

## Experimental setup and provenance

Experiment B changes only the fixed-step Euler integration count used to sample
Flow Matching replay trajectories: 10, 50, or 100 steps. Each setting has
training seeds 0, 1, and 2. All runs use the task sequence Hammer, Button Press,
Faucet Close, Handle Press Side, and Window Close; a fixed total historical
replay budget of 50; and 50 deterministic evaluation episodes per task from the
fixed task bank with seed 10000.

The resolved scientific configurations are identical apart from integration
steps and seed/run identity. The expert checkpoint hashes, task order, model and
BC settings, replay settings, normalization, and evaluation protocol agree
across all nine runs. The expert manifests describe the SAC checkpoints as
legacy/unverified; this is shared provenance, not a between-condition
difference.

## Validation

All nine runs passed validation. Each has `RUN_COMPLETE`, a complete manifest,
five stage markers, and a complete 5-stage matrix. Every marker-referenced
artifact and every replay-source Flow Matching checkpoint matches its recorded
SHA-256. The root matrix equals the stage-4 matrix. Every generated replay
record has integration steps and neural function evaluations equal to its run
condition. Ordered evaluation-task hashes, evaluation seeds 10000--10049, task
set seed 10000, and the 50-episode fixed-task protocol agree exactly.

## Individual results

Performance values are percentages; forgetting is in percentage points. Times
are seconds. Flow training time sums the five recorded
`generator_training_seconds` fields. Duration is manifest wall-clock elapsed
time and includes the whole experiment.

| Steps | Seed | Final AP | Forgetting | Hammer | Button | Faucet | Handle | Window | Replay sampling | Flow training | Duration |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 73.2 | 24.4 | 24 | 100 | 100 | 42 | 100 | 1.642 | 770.263 | 958.790 |
| 10 | 1 | 86.0 | 12.8 | 38 | 98 | 100 | 94 | 100 | 1.614 | 775.546 | 970.230 |
| 10 | 2 | 80.8 | 18.8 | 8 | 100 | 100 | 96 | 100 | 1.711 | 810.556 | 1009.417 |
| 50 | 0 | 85.2 | 14.0 | 44 | 92 | 100 | 90 | 100 | 8.317 | 801.755 | 1008.992 |
| 50 | 1 | 87.2 | 10.8 | 40 | 100 | 100 | 96 | 100 | 8.348 | 805.799 | 1010.092 |
| 50 | 2 | 78.4 | 19.2 | 24 | 100 | 100 | 68 | 100 | 8.164 | 783.406 | 980.839 |
| 100 | 0 | 71.2 | 27.6 | 40 | 46 | 100 | 70 | 100 | 16.041 | 776.065 | 979.498 |
| 100 | 1 | 83.2 | 16.8 | 26 | 100 | 92 | 98 | 100 | 15.835 | 765.286 | 968.090 |
| 100 | 2 | 82.4 | 15.2 | 22 | 100 | 90 | 100 | 100 | 16.556 | 818.320 | 1031.324 |

## Aggregate comparison

Values are mean +/- sample standard deviation across three seeds (`ddof=1`).

| Metric | 10 steps | 50 steps | 100 steps |
|---|---:|---:|---:|
| Final AP | 80.00 +/- 6.44% | 83.60 +/- 4.61% | 78.93 +/- 6.71% |
| Average forgetting | 18.67 +/- 5.80 pp | 14.67 +/- 4.24 pp | 19.87 +/- 6.74 pp |
| Hammer success | 23.33 +/- 15.01% | 36.00 +/- 10.58% | 29.33 +/- 9.45% |
| Button success | 99.33 +/- 1.15% | 97.33 +/- 4.62% | 82.00 +/- 31.18% |
| Faucet success | 100.00 +/- 0.00% | 100.00 +/- 0.00% | 94.00 +/- 5.29% |
| Handle success | 77.33 +/- 30.62% | 84.67 +/- 14.74% | 89.33 +/- 16.77% |
| Window success | 100.00 +/- 0.00% | 100.00 +/- 0.00% | 100.00 +/- 0.00% |
| Replay generation runtime | 1.656 +/- 0.050 s | 8.276 +/- 0.099 s | 16.144 +/- 0.371 s |
| Flow Matching training time | 785.455 +/- 21.898 s | 796.987 +/- 11.934 s | 786.557 +/- 28.031 s |
| Total experiment duration | 979.479 +/- 26.551 s | 999.974 +/- 16.581 s | 992.971 +/- 33.701 s |

## Paired seed comparisons

Differences below are percentage points (`higher steps - lower steps`).

| Comparison | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Direction consistent? |
|---|---|---:|---:|---:|---:|---|
| 50 - 10 | Final AP | +12.0 | +1.2 | -2.4 | +3.60 | No |
| | Forgetting | -10.4 | -2.0 | +0.4 | -4.00 | No |
| | Hammer | +20 | +2 | +16 | +12.67 | Yes, improves |
| | Handle | +48 | +2 | -28 | +7.33 | No |
| 100 - 10 | Final AP | -2.0 | -2.8 | +1.6 | -1.07 | No |
| | Forgetting | +3.2 | +4.0 | -3.6 | +1.20 | No |
| | Hammer | +16 | -12 | +14 | +6.00 | No |
| | Handle | +28 | +4 | +4 | +12.00 | Yes, improves |
| 100 - 50 | Final AP | -14.0 | -4.0 | +4.0 | -4.67 | No |
| | Forgetting | +13.6 | +6.0 | -4.0 | +5.20 | No |
| | Hammer | -4 | -14 | -2 | -6.67 | Yes, regresses |
| | Handle | -20 | +2 | +32 | +4.67 | No |

Three seeds are insufficient for claims of statistical significance.

## Mean evaluation matrices

Rows are stages and columns are tasks in sequence. Values are mean success
percentages; dashes are tasks not yet learned.

### 10 integration steps

| Stage | Hammer | Button | Faucet | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| Hammer | 98.00 | -- | -- | -- | -- |
| Button | 84.00 | 100.00 | -- | -- | -- |
| Faucet | 56.67 | 99.33 | 100.00 | -- | -- |
| Handle | 27.33 | 96.00 | 100.00 | 95.33 | -- |
| Window | 23.33 | 99.33 | 100.00 | 77.33 | 100.00 |

### 50 integration steps

| Stage | Hammer | Button | Faucet | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| Hammer | 98.00 | -- | -- | -- | -- |
| Button | 92.00 | 100.00 | -- | -- | -- |
| Faucet | 51.33 | 100.00 | 100.00 | -- | -- |
| Handle | 29.33 | 95.33 | 100.00 | 93.33 | -- |
| Window | 36.00 | 97.33 | 100.00 | 84.67 | 100.00 |

### 100 integration steps

| Stage | Hammer | Button | Faucet | Handle | Window |
|---:|---:|---:|---:|---:|---:|
| Hammer | 98.00 | -- | -- | -- | -- |
| Button | 88.00 | 100.00 | -- | -- | -- |
| Faucet | 61.33 | 98.00 | 100.00 | -- | -- |
| Handle | 42.00 | 94.67 | 100.00 | 96.00 | -- |
| Window | 29.33 | 82.00 | 94.00 | 89.33 | 100.00 |

## Forgetting and Hammer retention

Hammer is the most severely forgotten task under every setting. Its mean
trajectory is 98.0 -> 84.0 -> 56.7 -> 27.3 -> 23.3% at 10 steps,
98.0 -> 92.0 -> 51.3 -> 29.3 -> 36.0% at 50 steps, and
98.0 -> 88.0 -> 61.3 -> 42.0 -> 29.3% at 100 steps. One hundred steps retains
Hammer better at the Faucet and Handle stages, but loses that advantage after
Window. Final Hammer retention is lower than at 50 steps in every paired seed.

Current-task acquisition is nearly saturated: mean diagonal performance is
`[98, 100, 100, 95.3, 100]%`, `[98, 100, 100, 93.3, 100]%`, and
`[98, 100, 100, 96, 100]%` for 10, 50, and 100 steps. The principal differences
therefore arise after later training rather than when acquiring each task.

Possible positive backward transfer appears for Handle at 50 steps/seed 1
(90% at acquisition to 96% final), at 100 steps/seed 2 (90% to 100%), and for
Button at 100 steps/seeds 1 and 2 after small intermediate losses. These are
matrix observations and not mechanistic evidence.

## Computational cost

Mean replay-sampling runtime ratios are 4.999x for 50/10, 9.752x for 100/10,
and 1.951x for 100/50. Flow Matching training-time ratios are 1.015x, 1.001x,
and 0.987x respectively, confirming that ordinary timing variation dominates
training time. Overall wall-clock ratios are only 1.021x, 1.014x, and 0.993x
because replay sampling is a small part of these fixed-budget runs.

The sampling cost is approximately linear, but its absolute contribution is
small here. This may not hold for larger replay budgets or GPU/CPU regimes with
different relative costs.

## Relationship to Experiment A

Experiment A established numerical convergence toward the 500-step Euler
reference, with diminishing numerical changes around 50--100 steps. It did not
show improving trajectory quality: expert-action disagreement, action OOB
rates, and state-delta discrepancies generally worsened as integration became
more accurate.

Experiment B likewise shows that numerical accuracy is not monotonic with
downstream utility. Fifty steps has the best mean AP, lowest forgetting, and
best final Hammer retention. Moving to 100 steps, despite being closer to the
ODE solution, reduces mean AP by 4.67 points, increases forgetting by 5.20
points, and reduces Hammer by 6.67 points relative to 50. The 100-step seed-0
Button failure (46%) is the largest anomaly and drives substantial variance,
but the paired Hammer regression occurs in all three seeds.

These results are consistent with the hypothesis that coarse Euler error can
act as a beneficial sampling bias. They do not prove that mechanism: learned
model error, replay composition, BC optimization, and seed interactions remain
possible explanations.

## Limitations and conclusions

- There are only three training seeds, so uncertainty estimates are coarse and
  no significance claim is warranted.
- Evaluation success is quantized in two-point increments because each cell has
  50 episodes.
- Experiment A used one intermediate-stage Hammer checkpoint; its quality
  diagnostics do not cover every checkpoint and task used here.
- Environment-dynamics consistency of generated trajectories was not measured.
- Total duration is wall-clock provenance, not a controlled benchmark.

Established finding: 50 steps is the best aggregate setting among those tested.
It improves mean AP and forgetting over 10 steps and improves final Hammer
retention in every paired seed. One hundred steps does not extend that benefit:
it costs about twice as much to sample as 50 steps, has lower AP, higher
forgetting, and lower Hammer retention. The joint Experiment A/B evidence
supports a practical plateau around 50 steps and provides no present reason to
prefer 100 steps for this fixed-budget pipeline.

Recommended follow-up: retain 10 steps as the inexpensive baseline and 50 steps
as the primary setting. Before attributing the difference to integration
accuracy, repeat trajectory-quality diagnostics across stages/tasks and test
whether the 50-step retention advantage survives additional seeds or a larger
replay budget.

## Generated artifacts

- Machine-readable aggregate: `FLOW_MATCHING_INTEGRATION_STEPS_EXPERIMENT_B_SUMMARY.json`
- Final AP: `flow_matching_steps_final_ap.png`
- Average forgetting: `flow_matching_steps_forgetting.png`
- Hammer retention: `flow_matching_steps_hammer_retention.png`
- Replay runtime: `flow_matching_steps_replay_runtime.png`
