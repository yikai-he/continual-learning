---
status: current
last_verified: 2026-11-09
applies_to: continual-learning
---

# KUKA trajectory-representation ablation: seeds 0 and 1

## 1. Experiment setup

This report analyzes the ten completed seed-0 and seed-1 runs for the sequence
`kuka-hammer-v3` → `kuka-handle-press-side-v3` under
`runs/representation_ablation/two_task/`. The methods are DDPM with
`structured_22d`, `goal_conditioned_40d`, or `full_43d`, Real Replay, and No
Replay. Seeds 2–4 are not present and are treated as missing, never as zero.

The common protocol uses KUKA-v3 reward v2, 100 deterministic expert
trajectories per task, 100 historical trajectories for replay arms, 100 DDPM
epochs and diffusion steps, 100 BC epochs, seed-paired current-task data and
splits, raw generated actions without projection, and 20 fixed-task evaluation
episodes with reset/task-bank seed 10000. Historical replay is training-only,
so the stage-1 validation set contains current-task Handle data rather than
historical Hammer replay.

Average performance (AP) and forgetting use `src/continual/metrics.py` without
redefinition. In particular, forgetting includes the current task's zero term;
a Hammer-only decline of 0.15 therefore yields reported forgetting of 0.075 in
this two-task experiment.

The existing analyzer was executed in the `metaworld` Conda environment. No
correctness problem was found, so it was not modified for this analysis.

## 2. Completion and comparability audit

All ten requested runs are complete and valid. For every run:

- `RUN_COMPLETE`, `stage_0_COMPLETE.json`, and `stage_1_COMPLETE.json` exist.
- `completed_stage_markers` validated both contiguous markers and every
  referenced artifact SHA-256.
- `EvaluationMatrix.load` accepted the final matrix, with `R[0,0]`, `R[1,0]`,
  and `R[1,1]` finite and `R[0,1]` correctly unmeasured.
- The saved configuration equals the corresponding canonical YAML with only
  the expected runtime seed and output identity overrides.

Expert identities are identical across every method and seed:

- Hammer SAC: SHA-256
  `1900fe4dd8d3150f7d484f7f8886f4e65616bf11fa681d5a29ae6b7a06696c8c`.
- Handle SAC: SHA-256
  `090824e59d5dcfe3a3be0700a020877b97e5545b3bf9e507a02d1b132dc47049`.

Current-task inputs are exactly paired within each seed:

| Seed | Stage-0 Hammer array SHA-256 | Stage-1 Handle array SHA-256 |
|---:|---|---|
| 0 | `ce5940cee3ada91a7c1ca8a164531683c194e2bd7cb913f5bafed66c8b0bde00` | `31a02741ccfd5e05ddde4ede48ce445e276b5f3fd1d56b744e7f46ed40a16494` |
| 1 | `6e1009022b6b4e988cc176e167dd5144af278db218be6210f00831aa160a4973` | `a6f216c9cfb6716e6055824627ef26e0cf56e6779a7a74655b9bd2e541a1765d` |

For each seed, all five methods have the same current-task train/validation IDs
at both stages. Evaluation mode, episode/reset seeds, fixed task-bank identities,
episode count, backend, reward version, task order, budgets, BC settings, and
common generator hyperparameters also match. Intended differences are replay
mode, DDPM representation, output identity, and the absence of historical data
and historical-training-only semantics for No Replay. No unintended difference
was detected.

Seeds 2, 3, and 4 have no run directories for any method and remain explicitly
missing (15 runs total).

## 3. Per-seed continual performance

All values are endpoint success rates except AP and forgetting.

| Method | Seed | Hammer S0 | Hammer S1 | Handle S1 | Final AP | Forgetting |
|---|---:|---:|---:|---:|---:|---:|
| DDPM 22D | 0 | 1.00 | 0.85 | 1.00 | 0.925 | 0.075 |
| DDPM 22D | 1 | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| DDPM 40D | 0 | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| DDPM 40D | 1 | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| DDPM 43D | 0 | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| DDPM 43D | 1 | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| Real Replay | 0 | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| Real Replay | 1 | 1.00 | 1.00 | 0.95 | 0.975 | 0.000 |
| No Replay | 0 | 1.00 | 0.00 | 1.00 | 0.500 | 0.500 |
| No Replay | 1 | 1.00 | 0.00 | 1.00 | 0.500 | 0.500 |

The 20-episode evaluation resolves success in increments of 0.05. Most replay
cells are at 1.00, so the endpoint comparison is strongly ceiling-limited.

## 4. Mean ± sample standard deviation

Sample standard deviation uses `ddof=1`; `n=2` for every row.

| Method | Hammer S1 | Handle S1 | Final AP | Forgetting | n |
|---|---:|---:|---:|---:|---:|
| DDPM 22D | 0.9250 ± 0.1061 | 1.0000 ± 0.0000 | 0.9625 ± 0.0530 | 0.0375 ± 0.0530 | 2 |
| DDPM 40D | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 2 |
| DDPM 43D | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 2 |
| Real Replay | 1.0000 ± 0.0000 | 0.9750 ± 0.0354 | 0.9875 ± 0.0177 | 0.0000 ± 0.0000 | 2 |
| No Replay | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 2 |

These two observations per method provide a replication check, not sufficient
evidence for statistical significance or a stable population variance.

## 5. Paired comparisons

Each cell is `seed 0; seed 1; mean paired difference`. Differences are left
method minus right method and use only matched seeds.

| Comparison | Δ Hammer S1 | Δ Final AP |
|---|---:|---:|
| DDPM 40D − DDPM 22D | +0.150; +0.000; **+0.075** | +0.075; +0.000; **+0.0375** |
| DDPM 43D − DDPM 22D | +0.150; +0.000; **+0.075** | +0.075; +0.000; **+0.0375** |
| DDPM 43D − DDPM 40D | +0.000; +0.000; **+0.000** | +0.000; +0.000; **+0.000** |
| DDPM 22D − Real Replay | −0.150; +0.000; **−0.075** | −0.075; +0.025; **−0.0250** |
| DDPM 40D − Real Replay | +0.000; +0.000; **+0.000** | +0.000; +0.025; **+0.0125** |
| DDPM 43D − Real Replay | +0.000; +0.000; **+0.000** | +0.000; +0.025; **+0.0125** |
| DDPM 22D − No Replay | +0.850; +1.000; **+0.925** | +0.425; +0.500; **+0.4625** |
| DDPM 40D − No Replay | +1.000; +1.000; **+1.000** | +0.500; +0.500; **+0.5000** |
| DDPM 43D − No Replay | +1.000; +1.000; **+1.000** | +0.500; +0.500; **+0.5000** |

For the three representation comparisons, the sample SDs of the paired
differences are 0.1061 Hammer / 0.0530 AP for 40D−22D and 43D−22D, and zero for
43D−40D. These descriptive paired differences do not establish significance.

## 6. Stage-1 replay consistency diagnostics

Errors are Euclidean norms. Goal deviation and variation are timestep means;
previous-state error excludes timestep zero. Previous-state violations use the
implemented tolerance `1e-3`. Action OOB component fraction counts scalar action
coordinates; timestep fraction counts steps with at least one OOB coordinate.
No projection was applied.

| Representation | Seed | Goal mean / max deviation | Goal variation | Previous-state mean error | Previous-state violations | Action OOB components | OOB timesteps | Max violation | Sampling time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 22D | 0 | 0 / 0 | 0 | 0 | 0% | 11.3063% | 36.5750% | 0.797593 | 14.6594 s |
| 22D | 1 | 0 / 0 | 0 | 0 | 0% | 5.9700% | 17.9950% | 0.722806 | 13.4922 s |
| 40D | 0 | 0 / 0 | 0 | 0.0148745 | 100% | 7.3513% | 20.8550% | 0.649380 | 15.3718 s |
| 40D | 1 | 0 / 0 | 0 | 0.0182404 | 100% | 9.6225% | 30.2500% | 0.584683 | 13.5938 s |
| 43D | 0 | 2.69329e-5 / 1.05114e-4 | 6.01627e-5 | 0.0150218 | 100% | 15.2375% | 51.8700% | 1.038789 | 14.4568 s |
| 43D | 1 | 2.72056e-5 / 1.28959e-4 | 4.76673e-5 | 0.0191841 | 100% | 14.6987% | 52.4200% | 1.102423 | 13.6334 s |

The seed-to-seed structural pattern is stable: 22D enforces exact previous-state
and goal reconstruction; 40D enforces goals but generates inconsistent previous
states; 43D generates both previous states and small goal drift. The legacy 22D
stage reports omit the nested diagnostics object, so the existing analyzer
recomputed those rows read-only from saved replay arrays and stage-0 reference
goals using the repository formulas. No trajectories were regenerated.

Action validity is less stable across seeds. The 22D component OOB fraction
drops from 11.31% to 5.97%; 40D rises from 7.35% to 9.62%; 43D remains highest
at about 15%. Despite those differences, all three representations attain 1.00
Hammer retention at seed 1. At seed 0, 43D has the highest OOB rate but perfect
retention, while exact 22D has a lower OOB rate and only 0.85 retention. Neither
previous-state inconsistency nor action OOB frequency orders retention across
these six runs.

## 7. BC training observations

Losses below are the selected checkpoint's training and validation losses. The
selected epoch is shown in parentheses. Stage-1 duration is BC fitting time,
not generator fitting or replay sampling time.

| Method | Seed | Stage-0 train / validation (epoch) | Stage-1 train / validation (epoch) | Stage-1 BC duration |
|---|---:|---:|---:|---:|
| DDPM 22D | 0 | 0.00147667 / 0.00301352 (96) | 0.00435681 / 0.00808409 (98) | 104.610 s |
| DDPM 22D | 1 | 0.00144036 / 0.00351293 (93) | 0.00334660 / 0.01104492 (91) | 96.573 s |
| DDPM 40D | 0 | 0.00147667 / 0.00301352 (96) | 0.00355635 / 0.00754288 (98) | 101.812 s |
| DDPM 40D | 1 | 0.00144036 / 0.00351293 (93) | 0.00353429 / 0.01168469 (95) | 103.746 s |
| DDPM 43D | 0 | 0.00147667 / 0.00301352 (96) | 0.00397299 / 0.00780030 (99) | 105.733 s |
| DDPM 43D | 1 | 0.00144036 / 0.00351293 (93) | 0.00471250 / 0.01112265 (88) | 101.522 s |
| Real Replay | 0 | 0.00147667 / 0.00301352 (96) | 0.00417115 / 0.00793683 (98) | 98.193 s |
| Real Replay | 1 | 0.00144036 / 0.00351293 (93) | 0.00306949 / 0.01105262 (99) | 95.386 s |
| No Replay | 0 | 0.00147667 / 0.00301352 (96) | 0.00989344 / 0.01205332 (95) | 45.872 s |
| No Replay | 1 | 0.00144036 / 0.00351293 (93) | 0.00779770 / 0.01427134 (99) | 40.928 s |

Within a seed, all five Stage-0 loss values and selected epochs are exactly
identical, and all Stage-0 policies achieve 1.00 Hammer success. This is strong
artifact-level evidence that representation-dependent generator construction
did not perturb Stage-0 policy initialization or BC training in these runs.
Wall-clock durations differ slightly, but timing is not a numerical trajectory.

Stage-1 validation is the same real Handle holdout because historical replay is
training-only. Its loss should not be interpreted as a measure of Hammer
retention. Replay arms also have twice the Stage-1 training samples of No Replay,
which explains much of their longer BC duration by design.

## 8. Interpretation

### Observed findings

1. **Does 40D outperform 22D on both seeds?** No. It improves Hammer retention
   by 0.15 and AP by 0.075 at seed 0, but ties 22D at seed 1.
2. **Does 43D outperform 22D on both seeds?** No. The pattern is identical:
   improvement at seed 0 and a tie at seed 1.
3. **Is 40D measurably different from 43D?** Not in these endpoint metrics.
   Hammer, Handle, AP, and forgetting are identical for both seeds.
4. **Does Real Replay consistently outperform generated replay?** No. It ties
   40D/43D at seed 0. At seed 1 all DDPM arms have 1.00 Handle success while
   Real Replay has 0.95, giving DDPM 40D/43D 0.025 higher AP. Real Replay does
   tie all DDPM arms on seed-1 Hammer retention.
5. **Does No Replay consistently forget Hammer severely?** Yes in both observed
   seeds: Hammer falls from 1.00 to 0.00, with AP 0.50 and forgetting 0.50.
6. **Does previous-state inconsistency predict retention?** No descriptive
   ordering appears. Exact 22D is worse at seed 0 but ties inconsistent 40D/43D
   at seed 1. This does not imply inconsistency is beneficial.
7. **Does action OOB frequency predict retention?** No. The highest-OOB 43D arm
   retains Hammer perfectly at both seeds; OOB ordering does not match retention.
8. **Are there ceiling effects?** Yes. All DDPM Handle endpoints, all 40D/43D
   Hammer endpoints, and most other replay cells are 1.00. With only 20 episodes,
   small differences are quantized in 0.05 increments.

### Hypotheses

The seed-0 22D deficit could reflect generated state/action distribution details,
temporal structure not captured by the listed diagnostics, or ordinary training
seed variation. The current evidence cannot identify a mechanism. Similarly,
Real Replay's one Handle failure at seed 1 may be finite-evaluation variation or
a policy difference; two seeds cannot distinguish those explanations.

## 9. Limitations

- Two training seeds are insufficient for stable uncertainty estimation or
  significance claims.
- Endpoint success is ceiling-censored and based on only 20 episodes per cell.
- The reported SD has one degree of freedom.
- Replay diagnostics measure representation consistency and action bounds, not
  simulator feasibility, temporal realism, or causal replay utility.
- BC validation measures the current Handle holdout at Stage 1, not retained
  Hammer competence.
- Replay and No Replay differ in Stage-1 dataset size by the intended protocol.
- The task order is fixed, so conclusions do not establish order robustness.

## 10. Recommendation for seeds 2–4

Proceed with seeds 2–4 using the already prepared paired launcher and unchanged
protocol. Seed 1 removes the seed-0 22D retention deficit, demonstrating exactly
why further replication is necessary. Three additional seeds will bring the
planned total to five, improve the estimate of seed variation, and provide more
matched points for assessing whether replay diagnostics covary with retention.
Do not change the evaluation bank, replay allocation, projection, checkpoints,
or hyperparameters between seeds.

## Artifact references

- `docs/ARCHITECTURE.md`
- `docs/experiments/kuka_two_task_representation_ablation.md`
- `docs/experiments/kuka_trajectory_representation_analysis.md`
- `scripts/evaluation/analyze_kuka_representation_multiseed.py`
- Each seed-0/1 run's `config.yaml`, `run_manifest.json`, stage completion
  markers, `evaluation_matrix.json`, stage reports, saved stage arrays, and
  `timing_summary.json` under `runs/representation_ablation/two_task/`
