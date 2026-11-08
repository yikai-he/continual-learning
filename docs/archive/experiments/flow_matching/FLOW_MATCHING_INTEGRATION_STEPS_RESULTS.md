---
status: archived
original_date: unknown
archived_on: 2026-11-07
last_verified: unknown
superseded_by: docs/FLOW_REPLAY_ARCHITECTURE.md
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

Archived as a completed sampling study after removal of its dedicated analysis tooling.

# Flow Matching Integration Steps: Sampling Results

## Experiment setup and validation

This experiment sampled 50 Hammer trajectories from the Flow Matching model
after continual stage 2 (`faucet-close-v3`) using fixed-step Euler integration
with 10, 20, 50, 100, 200, and 500 neural function evaluations (NFE). All runs
used sampling seed 0, task condition 0 (`hammer-v3`), the same 50 goal
conditions, paired initial Gaussian noise, CPU legacy execution, and the same
model weights.

All six generated arrays exist, have shape `(50, 200, 43)` and dtype float32,
contain only finite values, and contain identical goal conditions. Each metrics
file exists and reports NFE equal to the requested integration-step count.

- Checkpoint: `flow_matching_after_task_2_faucet-close-v3.pt`
- Checkpoint SHA-256: `93d486d9e5e9a46cd594250fae322653db22fe643d054430eae66b9ec79f74e4`
- Checkpoint model: horizon 200, 22 generated features, width 64, five task
  conditions, stored default of 10 Euler steps
- Real reference: stage-0 Hammer `data_task_0.npy`, `(50, 200, 43)`
- Reference SHA-256: `ff63695d7258a9ffcd81b85ae51b27126067c0ab477e7b2a4421d8e061a3fec5`
- SAC expert SHA-256: `4322881959f1dd3f03a3d4be8db83b28c433dc68a66cfd869bb1cd85eba23349`

The metadata hashes agree with the previously validated continual stage
artifacts. No model was retrained and no sample was regenerated for this
analysis.

## Main comparison

Rates are fractions rather than percentages. State/action mean and standard
deviation discrepancies are mean absolute differences between the generated
and real per-coordinate statistics. Expert metrics compare each generated
action with the deterministic SAC expert action at the generated observation.

| Steps | Runtime (s) | NFE | Action OOB components | Action OOB timesteps | State mean MAD | State std MAD | Action mean MAD | Action std MAD | Expert MSE | Expert MAE | Expert mean L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.1962 | 10 | 0.033850 | 0.1107 | 0.006045 | 0.006352 | 0.029627 | 0.125259 | 0.248233 | 0.401520 | 0.954247 |
| 20 | 0.3443 | 20 | 0.035425 | 0.1158 | 0.006096 | 0.006469 | 0.029959 | 0.121239 | 0.253281 | 0.404484 | 0.962166 |
| 50 | 0.9042 | 50 | 0.036700 | 0.1201 | 0.006084 | 0.006627 | 0.029697 | 0.118798 | 0.256360 | 0.406241 | 0.966806 |
| 100 | 1.8253 | 100 | 0.037350 | 0.1224 | 0.006073 | 0.006695 | 0.029536 | 0.118018 | 0.257417 | 0.406904 | 0.968348 |
| 200 | 3.6216 | 200 | 0.037775 | 0.1239 | 0.006068 | 0.006729 | 0.029451 | 0.117634 | 0.257879 | 0.407182 | 0.968978 |
| 500 | 9.2922 | 500 | 0.037900 | 0.1243 | 0.006065 | 0.006750 | 0.029398 | 0.117404 | 0.258137 | 0.407331 | 0.969315 |

Runtime is a single wall-clock measurement, not a repeated timing estimate.
The per-NFE cost was approximately 17.2–19.6 ms and total runtime was close to
linear in NFE. Relative to 10 steps, 50, 100, 200, and 500 steps cost about
4.61x, 9.30x, 18.46x, and 47.37x respectively.

## State-transition and support diagnostics

The real state-delta norm has mean 0.009502, standard deviation 0.011721, p95
0.035821, and maximum 0.074400. The tail column below is the fraction of
generated deltas above the real p95. These are distributional diagnostics, not
simulator transition checks.

| Steps | Generated delta mean | Mean absolute error vs real | Delta std | Delta p95 | Delta max | Fraction > real p95 | Action components >0.99 | Mean action outside real p01–p99 | Max action overshoot |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.024961 | 0.015460 | 0.014993 | 0.057187 | 0.128759 | 0.164623 | 0.036750 | 0.037750 | 0.649036 |
| 20 | 0.029038 | 0.019537 | 0.016316 | 0.063857 | 0.135298 | 0.228643 | 0.038700 | 0.039800 | 0.669801 |
| 50 | 0.033619 | 0.024118 | 0.017294 | 0.069877 | 0.146164 | 0.325126 | 0.040325 | 0.041325 | 0.683108 |
| 100 | 0.035246 | 0.025744 | 0.017629 | 0.071817 | 0.149937 | 0.367638 | 0.040775 | 0.041675 | 0.687480 |
| 200 | 0.036046 | 0.026544 | 0.017803 | 0.072789 | 0.151776 | 0.390754 | 0.041075 | 0.042025 | 0.689659 |
| 500 | 0.036521 | 0.027020 | 0.017908 | 0.073554 | 0.152865 | 0.403819 | 0.041175 | 0.042100 | 0.690964 |

All generated trajectories have exact previous-state reconstruction and exact
constant-goal reconstruction: mean and maximum errors are zero for every step
count. Those equalities are guaranteed by decoding and establish schema
consistency only. They do not show physical validity.

The full JSON diagnostics additionally retain per-timestep action OOB rates,
per-timestep expert disagreement, per-timestep state-delta comparisons,
per-coordinate state/action support checks, saturation thresholds, and largest
individual overshoots. Six state coordinates are constant in the real
reference under the diagnostic's `1e-6` threshold; approximately 99.1–99.2% of
generated values in those coordinates differ by more than that very strict
tolerance. This indicates imperfect reproduction of constant reference
features, but the tolerance makes it unsuitable as a standalone physical
validity criterion.

## Paired numerical convergence toward 500 steps

The comparison uses corresponding trajectory, timestep, and feature indices.
Observations are reduced to the generated 22D representation (18 current-state
plus four action coordinates) and normalized with the normalizer stored in the
Flow checkpoint. Goals and reconstructed previous-state fields are excluded
from MSE because they are supplied or reconstructed exactly.

| Steps vs 500 | Normalized MSE | Normalized RMSE | Normalized max abs | Normalized state MSE | Normalized action MSE | Raw 22D MSE | Raw max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.52554e-3 | 0.0390581 | 0.532625 | 1.51460e-3 | 1.57474e-3 | 9.47416e-5 | 0.106124 |
| 20 | 3.89204e-4 | 0.0197283 | 0.269465 | 3.89574e-4 | 3.87540e-4 | 2.41582e-5 | 0.052236 |
| 50 | 5.37967e-5 | 0.00733462 | 0.105266 | 5.43814e-5 | 5.11653e-5 | 3.17847e-6 | 0.019246 |
| 100 | 1.05863e-5 | 0.00325366 | 0.047284 | 1.07161e-5 | 1.00025e-5 | 6.18026e-7 | 0.008533 |
| 200 | 1.48704e-6 | 0.00121944 | 0.017824 | 1.50642e-6 | 1.39983e-6 | 8.62887e-8 | 0.003197 |

Successive normalized MSEs were `3.85368e-4` (10–20), `1.55314e-4`
(20–50), `1.66646e-5` (50–100), `4.13846e-6` (100–200), and
`1.48704e-6` (200–500). Thus the Euler solution is clearly converging. The
largest gains occur by 50–100 steps, with diminishing changes beyond 100 and a
strong numerical plateau by 200 steps relative to this 500-step reference.
The 500-step solution is only a higher-resolution numerical reference, not an
exact solution or evidence of physical correctness.

## Quality versus computational cost

Increasing the number of steps did not produce a general quality improvement:

- State-coordinate mean discrepancy stayed almost flat near 0.0061.
- Action-coordinate standard-deviation discrepancy improved from 0.1253 at 10
  steps to 0.1174 at 500 steps, but most of the gain occurred by 50–100 steps.
- Expert-action MSE worsened monotonically from 0.2482 to 0.2581.
- Action OOB component rate worsened from 3.385% to 3.790%; OOB timestep rate
  worsened from 11.07% to 12.43%.
- Generated state deltas moved farther from the real distribution as steps
  increased. Their mean error rose from 0.01546 to 0.02702 and the fraction
  above the real p95 rose from 16.46% to 40.38%.
- Numerical differences to the 500-step trajectory fell rapidly and smoothly.

These results separate two effects: more Euler steps converge toward the ODE
defined by the learned velocity field, while that converged ODE solution is not
better under most available generative-quality proxies. Ten steps actually
looks best for action validity, expert consistency, and state-delta realism in
this single paired Hammer experiment. This may reflect beneficial coarse-step
numerical bias rather than a better learned distribution.

## Limitations

- There is one checkpoint, one task condition, one sampling seed, and 50 paired
  trajectories. No uncertainty across sampling seeds or checkpoints is known.
- Runtime has one observation per setting and includes ordinary timing noise.
- Distribution summaries, expert disagreement, and state-delta statistics do
  not test whether the environment would produce the proposed next state after
  the proposed action.
- Actual simulator dynamics consistency is unavailable because no transition
  oracle was run.
- Generated task success and episode return are unavailable because generated
  state sequences were not environment rollouts.
- The 500-step result is not ground truth. Convergence to it measures numerical
  stability only.

## Conclusions and Experiment B recommendation

There is a numerical convergence plateau: 50 steps is already close to the
500-step result (normalized RMSE 0.00733), 100 steps is closer (0.00325), and
200 steps is very close (0.00122). Computational cost remains nearly linear.
Quality proxies do not justify 200 or 500 steps and mostly favor 10 steps.

For Experiment B, retain 10 steps as the primary low-cost/default condition,
use 50 steps as the main numerically better-resolved compromise, and include
100 steps as the practical convergence-control condition. Twenty steps is
useful for resolving the early part of the curve. The 200- and 500-step runs
remain scientifically useful controls in the predeclared full ablation, but
the Experiment A evidence gives no quality-based reason to prefer them and
predicts large runtime costs. Downstream continual performance must determine
whether the small numerical differences between 10, 50, and 100 steps matter.
