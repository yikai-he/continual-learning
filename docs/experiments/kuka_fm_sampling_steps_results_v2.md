---
status: completed
last_verified: 2026-11-10
applies_to: continual-learning
---

# KUKA Flow Matching sampling-step ablation

## Executive summary

All 60 expected cells are complete and valid. The six step conditions are
exactly paired within all 60 method/seed/stage/task groups by generator
checkpoint, model-state hash, sampling seed, task, training-configuration
indices and hashes. No NaN, Inf, invalid shape, allocation error, or provenance
disagreement was found.

The saved evidence does **not** show that higher Euler resolution repairs FM
43D. From 10 to 400 steps, trajectory-weighted 43D goal error rises from
0.02490 to 0.02603, previous-state alignment error from 0.4671 to 0.4754, and
action-component OOB from 16.56% to 16.96%. Most change occurs below 100 steps;
100 and 400 are nearly indistinguishable on these aggregate diagnostics. FM
22D retains exact goal and previous-state reconstruction at every setting.

Ten steps is therefore a strong efficiency candidate for a follow-up replay/BC
experiment: it is 9.63× faster than 100 steps for 22D and 8.96× faster for 43D,
with slightly *better* recorded structural/action diagnostics. It cannot yet be
declared equivalent in expert-action agreement or downstream retention. The
diagnostic files contain trajectory hashes rather than trajectory arrays or
expert-action metrics, so those quantities cannot be reconstructed without the
prohibited resampling/evaluation.

## Setup and aggregation

The sweep covers FM structured 22D and full 43D, seeds 0–4, and 10, 25, 50,
100, 200, and 400 fixed Euler steps. Each method/seed/step cell contains Stage
1 Hammer (20 trajectories), Stage 2 Hammer/Handle (10/10), and Stage 3
Hammer/Handle/Button (7/7/6): 60 trajectories and 12,000 decoded timesteps per
cell. Each method/step summary below contains 300 trajectories and 60,000
timesteps.

Unless marked otherwise, means are trajectory/timestep weighted. Every
trajectory has horizon 200, so trajectory and global-timestep weights coincide.
“Max violation” in the tables is the trajectory-count-weighted mean of the
per-group maximum; the largest individual recorded group maximum is described
separately. Sample-time totals cover the six ODE calls per seed and exclude
checkpoint loading and analysis.

## Completion, integrity, and provenance

| Audit item | Result |
|---|---:|
| Expected/valid cells | 60/60 |
| Missing / incomplete / invalid / incomparable | 0 / 0 / 0 / 0 |
| Cross-step matched groups | 60/60 |
| Groups per method/step | 30 |
| Trajectories per method/step | 300 |
| Timesteps per method/step | 60,000 |
| NaN / Inf / invalid-shape flags | 0 / 0 / 0 |

Stage 1 uses the post-Stage-0 Hammer generator, Stage 2 the post-Stage-1
Handle generator, and Stage 3 the post-Stage-2 Button generator. Every saved
checkpoint path, file SHA-256, and expected model-state hash agrees across the
six matched step conditions. Identical checkpoint files also establish
identical saved normalization state. The sampling seed and complete ordered
training-configuration identities agree both with the original replay report
and across steps. The sampler deterministically constructs its initial
Gaussian from that seed, so exact initial-noise pairing is established by code
and metadata, although the noise tensor itself was not serialized.

The linked source manifests contain identical frozen expert hashes for all ten
runs: Hammer `1900fe4d…96c8c`, Handle `090824e5…47049`, Button
`ee92f679…2f0`, Reach `2be77e78…2792`. Experts are not invoked by this sampling
diagnostic. Reference goals were reconstructed from the paired training-bank
configuration hashes in memory; the reference tensors were not serialized.
The sampling script runtime-validated `(N,200,43)` decoded groups, corresponding
to 39 observation and 4 action features, before writing each result; the JSON
stores the validity flag but not an explicit shape vector.

## Expert-action disagreement

| Euler steps | FM 22D expert MSE | FM 43D expert MSE |
|---:|---:|---:|
| 10 | unavailable | unavailable |
| 25 | unavailable | unavailable |
| 50 | unavailable | unavailable |
| 100 | unavailable | unavailable |
| 200 | unavailable | unavailable |
| 400 | unavailable | unavailable |

The completed JSON files save structural diagnostics and a SHA-256 of each
generated tensor, but not tensor values, expert predictions, aggregate expert
MSE, or per-action MSE. Consequently mean, median, SD, 95th percentile,
maximum, per-dimension MSE, global weighting, and the requested paired
`E_10-E_100` expert comparison are not identifiable. The earlier factorial
report's 100-step expert disagreement used a separate analysis and cannot be
assigned to the newly sampled 100-step tensors merely because settings match.

## Structural and action diagnostics

### FM structured 22D

| Steps | Goal mean | Previous mean | Action comp. OOB | Action timestep OOB | Max violation |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | .11064 | .35867 | 1.0735 |
| 25 | 0 | 0 | .11534 | .37187 | 1.0945 |
| 50 | 0 | 0 | .11693 | .37663 | 1.1017 |
| 100 | 0 | 0 | .11765 | .37882 | 1.1055 |
| 200 | 0 | 0 | .11803 | .37993 | 1.1074 |
| 400 | 0 | 0 | .11823 | .38052 | 1.1084 |

Goal maximum, within-trajectory variation, goal tolerance violations, previous
maximum, previous tolerance violations, and timestep-zero reset error are also
exactly zero in every 22D group. This matches reconstruction by construction.
Action violations rise slightly with resolution and plateau by 100 steps.

### FM full 43D

| Steps | Goal mean | Goal max | Goal variation | Goal >1e-3 | Previous mean | Previous max | Previous >1e-3 | Reset error | Action comp. OOB | Action timestep OOB | Max violation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | .02490 | .08408 | .03591 | .89142 | .46706 | 1.10935 | 1.000 | .50002 | .16555 | .50482 | 1.5579 |
| 25 | .02559 | .08631 | .03691 | .89220 | .47223 | 1.11959 | 1.000 | .50510 | .16798 | .51048 | 1.5782 |
| 50 | .02582 | .08704 | .03724 | .89245 | .47395 | 1.12285 | 1.000 | .50679 | .16878 | .51212 | 1.5836 |
| 100 | .02594 | .08740 | .03741 | .89268 | .47480 | 1.12444 | 1.000 | .50764 | .16923 | .51293 | 1.5863 |
| 200 | .02600 | .08758 | .03749 | .89277 | .47523 | 1.12530 | 1.000 | .50807 | .16947 | .51347 | 1.5877 |
| 400 | .02603 | .08767 | .03753 | .89282 | .47545 | 1.12580 | 1.000 | .50828 | .16961 | .51378 | 1.5883 |

All 43D previous-state transitions exceed the existing tolerance. The largest
recorded per-group action violation is about 2.498 at every setting. Increasing
resolution does not reduce goal drift, previous-state inconsistency, or action
violations; the metrics move slightly in the opposite direction toward a
stable high-step limit. The 18D previous-state norm is a feature-space norm,
not a distance in metres.

## Ten versus 100 steps

Trajectory-weighted per-seed differences use `10 - 100`; SD is the sample SD
with `ddof=1` over five paired seeds.

| Representation | Metric | Per-seed differences | Mean ± SD | Relative to 100 |
|---|---|---|---:|---:|
| 22D | Action component OOB | −.01094, −.00485, −.00583, −.00696, −.00648 | −.00701 ± .00233 | −5.96% |
| 22D | Goal / previous mean | 0 in all seeds | 0 ± 0 | n/a |
| 43D | Goal mean | −.00109, −.00104, −.00098, −.00127, −.00082 | −.00104 ± .00017 | −4.01% |
| 43D | Previous mean | −.00834, −.00755, −.00746, −.00800, −.00740 | −.00775 ± .00040 | −1.63% |
| 43D | Action component OOB | −.00198, −.00515, −.00379, −.00344, −.00402 | −.00368 ± .00114 | −2.17% |

Thus 10 steps consistently improves these recorded diagnostics relative to 100
in every seed; it does not establish equivalent expert agreement.

## Stage- and task-wise findings

For 43D, trajectory-weighted Stage 1/2/3 goal means change from
`.00129/.03884/.03457` at 10 steps to `.00130/.04030/.03650` at 400.
Previous-state means change from `.4007/.5420/.4584` to
`.4023/.5514/.4727`. The same mild worsening therefore occurs at every stage,
including the high-error Stage 2 and Stage 3 regimes.

Handle is not rescued by resolution. For 43D at Stage 2, 10→100→400 gives goal
mean `.03776→.03838→.03844`, previous mean `.5219→.5256→.5259`, and action
component OOB `.1446→.1481→.1485`. At Stage 3 the corresponding values are
`.03527→.03734→.03751`, `.4629→.4774→.4786`, and
`.1147→.1194→.1198`. For 22D Handle remains structurally exact; its action OOB
also rises from `.1093/.0863` at Stage 2/3 with 10 steps to
`.1204/.0966` with 400. Expert disagreement and numerical tensor distances are
unavailable, so no claim about improved Handle retention follows.

Hammer is the counterexample with strong stage dependence: 43D goal error is
only `.00129` at Stage 1 but roughly `.040–.043` after later generator updates.
Button at Stage 3 is intermediate (goal mean `.02750` at 10 and `.02806` at
400). These task differences are far larger than the effect of integration
resolution.

## Numerical convergence

Every group records a trajectory SHA-256, which establishes output identity
but not distance. Hashes differ across step conditions, so outputs are not
bitwise identical. Mean observation difference, mean action difference, and
maximum absolute difference for adjacent pairs or against 400 cannot be
recovered from hashes. No trajectories were regenerated, as required.

Aggregate diagnostics approach a plateau: for 43D, 100→400 changes goal mean
by only +0.000088 (+0.34%), previous mean by +0.000641 (+0.14%), and action
component OOB by +0.000379 (+0.22%). This is evidence of convergence of these
summaries, not proof of sample-level ODE convergence. The 400-step output is a
higher-resolution reference, not ground truth.

## Sampling efficiency

| Steps | 22D time, s | 22D NFE | 22D timesteps/s | 43D time, s | 43D NFE | 43D timesteps/s |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.4500 | 300 | 41,380 | 1.4936 | 300 | 40,171 |
| 25 | 3.5022 | 750 | 17,132 | 3.7437 | 750 | 16,027 |
| 50 | 7.1628 | 1,500 | 8,377 | 7.0927 | 1,500 | 8,459 |
| 100 | 13.9705 | 3,000 | 4,295 | 13.3785 | 3,000 | 4,485 |
| 200 | 27.9182 | 6,000 | 2,149 | 29.2286 | 6,000 | 2,053 |
| 400 | 58.6550 | 12,000 | 1,023 | 59.2875 | 12,000 | 1,012 |

Across the five seeds, per-seed total sampling time is `0.2900±0.0217` seconds
at 10 versus `2.7941±0.1966` at 100 for 22D, and `0.2987±0.0193` versus
`2.6757±0.1571` for 43D. Ratio-of-total-times speedups are 9.63×/40.45× for
10 versus 100/400 in 22D and 8.96×/39.69× in 43D. These are recorded model/ODE
sampling times on the experiment host; hardware metadata was not serialized.

## Interpretation and research questions

Observed facts:

1. Ten steps preserves or slightly improves every available aggregate quality
   diagnostic relative to 100, while reducing recorded sampling time by about
   ninefold.
2. Increasing resolution does not reduce 43D goal drift, previous-state error,
   or action OOB. The 100- and 400-step summaries are already very close.
3. Stage/task effects, particularly later Hammer and Handle replay, dominate
   the small step-count effect.
4. All outputs are finite and structurally valid as tensors; 43D remains badly
   inconsistent as a physical observation sequence.

The data therefore argue against insufficient Euler resolution as the primary
explanation for the recorded FM 43D structural failure. A learned-velocity-field
or training/calibration limitation is more plausible: the higher-resolution
solver converges toward essentially the same inconsistent distribution.
This is descriptive evidence, not causal identification; sample-level
convergence and expert disagreement remain unmeasured.

Ten steps is acceptable as a candidate—not yet a validated replacement. The
most informative next experiment is a paired 10-versus-100 replay/BC study that
stores expert-disagreement summaries and sample-pair distance statistics at
generation time, then trains matched BC policies to measure retention. Saving
only those summaries would avoid retaining raw trajectories while answering
both numerical and downstream questions.

## Limitations

- Expert-action statistics and per-action MSE were not saved.
- Generated tensors were not retained; hashes cannot measure numerical distance.
- No BC retraining or policy evaluation was performed, so downstream utility is unknown.
- The exact decoded shape is runtime-validated but not serialized explicitly.
- Hardware/runtime metadata is absent.
- Five seeds cover one task order, data budget, FM architecture, and Euler solver.
- Aggregate closeness does not establish distributional equivalence.

The machine-readable companion is
[`kuka_fm_sampling_steps_results_v2.json`](kuka_fm_sampling_steps_results_v2.json).
