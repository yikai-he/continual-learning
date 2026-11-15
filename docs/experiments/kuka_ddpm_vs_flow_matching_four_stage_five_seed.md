---
status: completed
last_verified: 2026-11-10
applies_to: continual-learning
---

# DDPM versus Flow Matching: four-stage KUKA five-seed factorial analysis

## Executive summary

All 20 intended runs are complete, valid, and exactly paired within seed. The DDPM result does **not** generalize to Flow Matching: DDPM 43D beats DDPM 22D in final AP in 5/5 seeds (+10.25±4.37 percentage points), whereas FM 43D beats FM 22D in only 1/5 seeds and is worse on average (−6.75±11.85 points). The AP interaction is negative in all five seeds, −17.00±10.02 points. DDPM 43D has the highest final AP (.8100±.0704) and lowest forgetting (.0700±.1018); FM 43D has the lowest AP (.5600±.1184) and highest forgetting (.3475±.1102).

FM replay has markedly worse expert-action agreement than DDPM, especially in 43D. FM 43D also has much larger goal drift, previous-state inconsistency, and action-bound violations. These diagnostics align with the poor FM 43D aggregate result, but task/seed counterexamples remain and do not establish causality.

## Experiment and audit

The common sequence is Hammer → Handle Press Side → Button Press → Reach. Every arm uses 20 current expert trajectories, fixed-total historical replay 20, training-only historical replay, 100 generator epochs, 100 BC epochs, raw actions, no projection, and 20 fixed-bank evaluations (seed 10000). DDPM uses noise prediction and 100 reverse steps. FM uses linear conditional-flow velocity regression and 100 fixed-step Euler substeps from Gaussian noise over `[0,1]`. These steps and objectives are not computationally equivalent.

All 20 runs have `RUN_COMPLETE`, four hash-validated stage markers, generator/GeneralPolicy checkpoints, resolved configuration, valid matrices, reports, and timings. Saved expert hashes, current-task data hashes, split IDs, evaluation-task hashes, BC settings, task order, replay allocation/budget, and action settings match across all four arms within each seed. No run is missing, incomplete, invalid, or scientifically incomparable.

## Full success and return matrices

Each cell contains four rows (stages 0–3), with columns Hammer, Handle, Button, Reach. The right column is mean return in the same layout.

| Seed | Arm | Success matrix | Return matrix |
|---:|---|---|---|
| 0 | DDPM 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.80,.95,1,—]`<br>`[.95,.50,1,.65]` | `[397.8,—,—,—]`<br>`[231.3,871.4,—,—]`<br>`[261.4,926.5,1478.0,—]`<br>`[203.6,551.0,1467.5,1714.3]` |
| 1 | DDPM 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.95,.95,1,—]`<br>`[1,1,.80,.55]` | `[388.7,—,—,—]`<br>`[379.3,883.4,—,—]`<br>`[376.8,738.4,1485.4,—]`<br>`[272.2,722.9,1308.9,1609.1]` |
| 2 | DDPM 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.70,.95,1,—]`<br>`[0,.70,1,.70]` | `[347.7,—,—,—]`<br>`[319.6,956.9,—,—]`<br>`[405.9,938.9,1478.5,—]`<br>`[361.5,887.8,1420.6,1653.3]` |
| 3 | DDPM 22D | `[.95,—,—,—]`<br>`[.90,.70,—,—]`<br>`[.70,.80,1,—]`<br>`[.85,.65,.60,.45]` | `[372.9,—,—,—]`<br>`[410.3,792.6,—,—]`<br>`[286.0,687.8,1466.6,—]`<br>`[292.4,835.3,1336.3,1652.5]` |
| 4 | DDPM 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.75,.90,1,—]`<br>`[.45,.55,.95,.80]` | `[390.0,—,—,—]`<br>`[347.9,918.2,—,—]`<br>`[331.3,794.7,1488.9,—]`<br>`[267.5,426.2,1382.6,1651.2]` |
| 0 | DDPM 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.85,.85,1,—]`<br>`[1,.45,1,.95]` | `[397.8,—,—,—]`<br>`[269.4,919.7,—,—]`<br>`[278.3,828.1,1482.8,—]`<br>`[324.0,640.1,1485.3,1761.1]` |
| 1 | DDPM 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[1,.95,1,—]`<br>`[.80,1,1,.75]` | `[388.7,—,—,—]`<br>`[315.3,911.5,—,—]`<br>`[379.0,982.1,1491.1,—]`<br>`[198.4,1030.4,1392.5,1675.5]` |
| 2 | DDPM 43D | `[1,—,—,—]`<br>`[.80,1,—,—]`<br>`[.95,1,1,—]`<br>`[.95,.35,.90,.60]` | `[347.7,—,—,—]`<br>`[353.0,826.1,—,—]`<br>`[388.8,837.7,1466.5,—]`<br>`[406.0,649.8,1449.4,1541.8]` |
| 3 | DDPM 43D | `[.95,—,—,—]`<br>`[1,.65,—,—]`<br>`[1,.90,1,—]`<br>`[.95,.90,1,.35]` | `[372.9,—,—,—]`<br>`[304.7,803.2,—,—]`<br>`[342.7,854.4,1493.2,—]`<br>`[353.2,996.1,1477.6,1429.3]` |
| 4 | DDPM 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[1,1,1,—]`<br>`[1,.90,1,.35]` | `[390.0,—,—,—]`<br>`[306.6,967.3,—,—]`<br>`[251.9,904.9,1476.6,—]`<br>`[282.2,631.3,1502.9,1670.7]` |
| 0 | FM 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.60,.25,1,—]`<br>`[1,.15,1,1]` | `[397.8,—,—,—]`<br>`[287.2,872.1,—,—]`<br>`[213.1,220.0,1482.5,—]`<br>`[292.3,154.9,1295.0,1757.4]` |
| 1 | FM 22D | `[1,—,—,—]`<br>`[.60,1,—,—]`<br>`[.75,.25,1,—]`<br>`[.85,0,.95,.75]` | `[388.7,—,—,—]`<br>`[404.1,951.1,—,—]`<br>`[165.2,411.1,1474.6,—]`<br>`[346.7,116.3,1394.3,1711.9]` |
| 2 | FM 22D | `[1,—,—,—]`<br>`[.50,1,—,—]`<br>`[.80,.25,1,—]`<br>`[.95,0,.95,.75]` | `[347.7,—,—,—]`<br>`[202.7,885.6,—,—]`<br>`[240.1,344.4,1466.7,—]`<br>`[314.7,171.9,1106.3,1624.4]` |
| 3 | FM 22D | `[.95,—,—,—]`<br>`[1,.95,—,—]`<br>`[1,.30,1,—]`<br>`[1,.05,.90,.65]` | `[372.9,—,—,—]`<br>`[391.5,917.2,—,—]`<br>`[245.9,237.7,1469.8,—]`<br>`[203.5,137.2,1107.7,1553.5]` |
| 4 | FM 22D | `[1,—,—,—]`<br>`[.25,1,—,—]`<br>`[.70,.35,1,—]`<br>`[1,0,.25,.35]` | `[390.0,—,—,—]`<br>`[358.6,924.0,—,—]`<br>`[247.0,362.0,1456.7,—]`<br>`[309.8,146.0,890.6,1361.0]` |
| 0 | FM 43D | `[1,—,—,—]`<br>`[.95,1,—,—]`<br>`[.80,0,.90,—]`<br>`[.90,.05,1,1]` | `[397.8,—,—,—]`<br>`[307.5,977.5,—,—]`<br>`[180.6,88.0,1435.2,—]`<br>`[176.3,75.9,550.6,1707.9]` |
| 1 | FM 43D | `[1,—,—,—]`<br>`[.90,1,—,—]`<br>`[.35,.10,1,—]`<br>`[.70,.10,.35,.70]` | `[388.7,—,—,—]`<br>`[194.0,931.6,—,—]`<br>`[179.6,94.8,1491.6,—]`<br>`[155.0,97.7,76.7,1647.4]` |
| 2 | FM 43D | `[1,—,—,—]`<br>`[.85,1,—,—]`<br>`[.20,.10,1,—]`<br>`[.45,0,.70,.75]` | `[347.7,—,—,—]`<br>`[280.2,899.1,—,—]`<br>`[186.3,140.7,1468.4,—]`<br>`[169.9,69.9,697.4,1755.4]` |
| 3 | FM 43D | `[.95,—,—,—]`<br>`[.90,1,—,—]`<br>`[.60,0,1,—]`<br>`[.95,0,1,.55]` | `[372.9,—,—,—]`<br>`[322.8,880.4,—,—]`<br>`[341.2,72.5,1498.3,—]`<br>`[334.7,79.5,720.4,1601.6]` |
| 4 | FM 43D | `[1,—,—,—]`<br>`[.90,1,—,—]`<br>`[.95,.10,1,—]`<br>`[.90,.10,.70,.30]` | `[390.0,—,—,—]`<br>`[305.0,928.8,—,—]`<br>`[187.0,88.0,1475.3,—]`<br>`[358.7,96.6,908.6,1607.5]` |

FM causes early Handle collapse after Button, often followed by no recovery. Hammer sometimes recovers by Stage 3, demonstrating non-monotonicity.

## Stage-wise AP and Forgetting

| Seed | Arm | AP0 | AP1 | AP2 | AP3 | F0 | F1 | F2 | F3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | DDPM 22D | 1.000 | 1.000 | .917 | .775 | 0 | 0 | .083 | .138 |
| 0 | DDPM 43D | 1.000 | 1.000 | .900 | .850 | 0 | 0 | .100 | .138 |
| 0 | FM 22D | 1.000 | 1.000 | .617 | .787 | 0 | 0 | .383 | .212 |
| 0 | FM 43D | 1.000 | .975 | .567 | .738 | 0 | .025 | .400 | .237 |
| 1 | DDPM 22D | 1.000 | 1.000 | .967 | .837 | 0 | 0 | .033 | .050 |
| 1 | DDPM 43D | 1.000 | 1.000 | .983 | .887 | 0 | 0 | .017 | .050 |
| 1 | FM 22D | 1.000 | .800 | .667 | .637 | 0 | .200 | .333 | .300 |
| 1 | FM 43D | 1.000 | .950 | .483 | .462 | 0 | .050 | .517 | .463 |
| 2 | DDPM 22D | 1.000 | 1.000 | .883 | .600 | 0 | 0 | .117 | .325 |
| 2 | DDPM 43D | 1.000 | .900 | .983 | .700 | 0 | .100 | .017 | .200 |
| 2 | FM 22D | 1.000 | .750 | .683 | .662 | 0 | .250 | .317 | .275 |
| 2 | FM 43D | 1.000 | .925 | .433 | .475 | 0 | .075 | .567 | .463 |
| 3 | DDPM 22D | .950 | .800 | .833 | .638 | 0 | .025 | .050 | .137 |
| 3 | DDPM 43D | .950 | .825 | .967 | .800 | 0 | −.025 | −.100 | −.062 |
| 3 | FM 22D | .950 | .975 | .767 | .650 | 0 | −.025 | .200 | .237 |
| 3 | FM 43D | .950 | .950 | .533 | .625 | 0 | .025 | .450 | .250 |
| 4 | DDPM 22D | 1.000 | 1.000 | .883 | .688 | 0 | 0 | .117 | .263 |
| 4 | DDPM 43D | 1.000 | 1.000 | 1.000 | .812 | 0 | 0 | 0 | .025 |
| 4 | FM 22D | 1.000 | .625 | .683 | .400 | 0 | .375 | .317 | .438 |
| 4 | FM 43D | 1.000 | .950 | .683 | .500 | 0 | .050 | .317 | .325 |

| Arm | AP0 | AP1 | AP2 | AP3 | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DDPM 22D | .990±.022 | .960±.089 | .897±.049 | .708±.098 | 0 | .005±.011 | .080±.038 | .183±.110 |
| DDPM 43D | .990±.022 | .945±.080 | .967±.039 | .810±.070 | 0 | .015±.049 | .007±.071 | .070±.102 |
| FM 22D | .990±.022 | .830±.158 | .683±.054 | .628±.141 | 0 | .160±.170 | .310±.067 | .293±.088 |
| FM 43D | .990±.022 | .950±.018 | .540±.095 | .560±.118 | 0 | .045±.021 | .450±.098 | .348±.110 |

## Final comparison

| Metric | DDPM 22D | DDPM 43D | FM 22D | FM 43D |
|---|---:|---:|---:|---:|
| Final AP | .7075±.0979 | **.8100±.0704** | .6275±.1407 | .5600±.1184 |
| Final F | .1825±.1099 | **.0700±.1018** | .2925±.0878 | .3475±.1102 |
| Hammer | .6500±.4228 | .9400±.0822 | **.9600±.0652** | .7800±.2080 |
| Handle | .6800±.1956 | **.7200±.2971** | .0400±.0652 | .0500±.0500 |
| Button | .8700±.1718 | **.9800±.0447** | .8100±.3150 | .7500±.2693 |
| Reach | .6300±.1351 | .6000±.2598 | **.7000±.2345** | .6600±.2584 |
| n | 5 | 5 | 5 | 5 |

FM 22D's relatively strong Hammer/Reach acquisition cannot offset near-total Handle loss. FM 43D additionally loses Hammer and Button. DDPM 43D's advantage is historical retention, not Reach acquisition.

## Paired effects

Values are percentage points; negative F differences are better.

| Contrast | Metric | Per-seed differences | Mean±SD | +/tie/− |
|---|---|---|---:|---:|
| DDPM 43D−22D | AP | +7.5,+5.0,+10.0,+16.3,+12.5 | +10.3±4.4 | 5/0/0 |
|  | F | 0,0,−12.5,−20.0,−23.8 | −11.3±11.0 | 0/2/3 |
| FM 43D−22D | AP | −5.0,−17.5,−18.8,−2.5,+10.0 | −6.8±11.8 | 1/0/4 |
|  | F | +2.5,+16.3,+18.8,+1.3,−11.3 | +5.5±12.2 | 4/0/1 |
| FM−DDPM, 22D | AP | +1.3,−20.0,+6.3,+1.3,−28.8 | −8.0±15.4 | 3/0/2 |
|  | F | +7.5,+25.0,−5.0,+10.0,+17.5 | +11.0±11.3 | 4/0/1 |
| FM−DDPM, 43D | AP | −11.3,−42.5,−22.5,−17.5,−31.3 | −25.0±12.2 | 0/0/5 |
|  | F | +10.0,+41.3,+26.3,+31.3,+30.0 | +27.8±11.4 | 5/0/0 |

Task-wise representation effects differ sharply. DDPM 43D−22D is Hammer +29±46, Handle +4±27, Button +11±19, Reach −3±30. FM 43D−22D is Hammer −18±18, Handle +1±9, Button −6±39, Reach −4±4. FM versus DDPM at 43D reduces Handle by 67±27 points in every seed.

## Representation × generator interaction

| Metric | Per-seed interaction | Mean±SD |
|---|---|---:|
| Final AP | −12.5,−22.5,−28.8,−18.8,−2.5 | **−17.0±10.0 pp** |
| Final F | +2.5,+16.3,+31.3,+21.3,+12.5 | **+16.8±10.6 pp** |
| Hammer | −15,+5,−145,−15,−65 | −47±61 pp |
| Handle | −5,+10,+35,−30,−25 | −3±27 pp |
| Button | 0,−80,−15,−30,+40 | −17±44 pp |
| Reach | −30,−25,+10,0,+40 | −1±28 pp |

The AP interaction is negative in 5/5 seeds: the 43D advantage is generator-dependent and reverses under FM. Exploratory paired Student-t 95% intervals are `[−29.4,−4.6]` points for AP interaction and `[+3.6,+29.9]` for F interaction (n=5, approximate normal independent seed effects). These are exploratory, not universal causal estimates.

## Replay quality

22D exactly reconstructs goal and previous state for both generators. The following 43D values equally average recorded historical groups within stage:

| Arm/stage | Goal mean | Goal max | >1e−3 | Previous mean | Action component OOB | Timestep OOB | Max violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| DDPM 43D S1 | .000079 | .000275 | 0 | .0316 | .0926 | .2651 | .3796 |
| DDPM 43D S2 | .007973 | .026625 | .9860 | .0434 | .1282 | .4269 | .4742 |
| DDPM 43D S3 | .009485 | .027494 | .9941 | .0430 | .1194 | .3788 | .5718 |
| FM 43D S1 | .001295 | .003610 | .6783 | .4022 | .1828 | .5368 | 1.6448 |
| FM 43D S2 | .040183 | .136687 | 1.0000 | .5506 | .1808 | .5478 | 1.7414 |
| FM 43D S3 | .035952 | .120644 | .9998 | .4677 | .1428 | .4510 | 1.3703 |

FM 43D already violates goal tolerance frequently at Stage 1 and has roughly 4–5× DDPM's later mean goal error, about 11× previous-state error, and larger action violations. Goal drift is task/seed dependent; full per-group recorded diagnostics are in the combined JSON. Every saved group is finite with valid shape. This establishes diagnostic differences, not physical validity or their causal origin.

## Expert-action disagreement

The scalar is timestep squared L2 error summed over four action coordinates. Values below equally weight saved task/stage/seed groups, not all timesteps globally.

| Metric | DDPM 22D | DDPM 43D | FM 22D | FM 43D |
|---|---:|---:|---:|---:|
| Raw | .722122 | **.601515** | 1.683554 | 2.668947 |
| Both structural corrections | .722122 | **.635362** | 1.683554 | 2.704345 |
| Clipped generated action | .677479 | **.570650** | 1.599721 | 2.410102 |

FM does not improve agreement: it is worse than DDPM for both representations, and FM 43D is worst. Under DDPM, 43D improves raw agreement; under FM, it worsens it. Structural correction does not improve either 43D aggregate. Stage-wise raw disagreement rises from S1 to later stages: DDPM 22D `.397/.703/.843`, DDPM 43D `.323/.602/.694`, FM 22D `1.324/1.878/1.674`, FM 43D `1.750/2.966/2.778`.

Handle disagreement is highest for both FM representations (2.421 in 22D; 3.595 in 43D), aligning with near-zero retention. Counterexamples remain: FM 22D has exact structure but catastrophic Handle forgetting, and its Hammer retention is excellent despite high disagreement. Expert agreement is not dynamics feasibility or a complete retention predictor.

See [FM disagreement report](kuka_fm_expert_action_disagreement.md), [FM disagreement JSON](kuka_fm_expert_action_disagreement.json), and the existing DDPM disagreement report.

## Efficiency

Five-seed mean recorded seconds by stage:

| Arm | Generator train S0/S1/S2/S3 | Sampling S0/S1/S2/S3 | BC S0/S1/S2/S3 | Mean total run | Replay NFE S0/S1/S2/S3 |
|---|---|---|---|---:|---|
| DDPM 22D | 11.55/23.86/24.11/24.44 | 0/5.07/7.99/11.02 | 7.66/17.16/16.79/16.67 | 256.68 | 0/100/200/300 |
| DDPM 43D | 12.26/25.36/25.06/25.80 | 0/5.19/8.27/11.18 | 7.75/17.33/17.12/16.90 | 263.08 | 0/100/200/300 |
| FM 22D | 11.80/24.44/24.65/24.59 | 0/5.08/7.93/11.00 | 7.73/17.24/16.52/16.00 | 256.56 | 0/100/200/300 |
| FM 43D | 12.57/25.92/26.00/25.99 | 0/5.10/8.11/11.04 | 7.77/17.19/16.12/15.79 | 260.63 | 0/100/200/300 |

Recorded CPU timings and forward-call totals are similar. NFE totals sum one 100-step sample call per represented historical task (1/2/3 tasks); they do not imply equal numerical operations or objectives. Runtime comparisons are specific to this software/hardware environment.

## Scientific interpretation

Direct observations:

1. 43D wins under DDPM but loses under FM; the representation advantage is not generator-invariant.
2. FM does not outperform DDPM reliably. At 43D it is worse in AP and F in all five seeds. At 22D its AP effect is mixed and mean negative.
3. DDPM 43D is the best combination on AP/F. FM 43D is the worst.
4. Handle is the persistent FM failure: final success is .04/.05 for FM 22D/43D.
5. FM 43D has worse structural diagnostics and expert agreement than DDPM 43D and FM 22D.
6. Current Reach acquisition does not explain the main differences; historical retention dominates.

Descriptive mechanism hypotheses:

- The unchanged FM architecture/hyperparameters may be poorly calibrated for unconstrained 43D generation at this small data budget.
- Euler integration error, velocity regression difficulty, or normalization across extra generated fields may contribute.
- Fixed-total replay dilution may amplify weak historical samples, especially Handle.

None is identified causally by this factorial experiment. The generator families differ in objective and sampler, so the interaction is an empirical pipeline interaction, not a pure architectural main effect.

## Limitations

- Five seeds, one sequence, one replay budget, and one hyperparameter setting per generator.
- Twenty evaluation episodes give 5-point success resolution.
- Exploratory intervals rely on approximate normality of five paired seed effects.
- Equal-group disagreement summaries are not global timestep-weighted estimates.
- Expert agreement can be unreliable on off-distribution generated observations.
- Structural diagnostics do not test simulator transition feasibility.
- Nominal 100-step settings are not equivalent across generators.

## Recommended follow-up

Before a larger replay-budget study, run a **small FM generator-quality calibration on saved Stage-0/Stage-1 data** without continual-policy conclusions: compare Euler step counts and a supported higher-order solver while holding representation and training data fixed. The immediate controlled CL follow-up should then compare DDPM 22D/43D and the best calibrated FM setting under fixed-per-task replay. This separates FM underfitting/integration quality from replay dilution and representation effects.
