---
status: completed
last_verified: 2026-11-10
applies_to: continual-learning
---

# KUKA 200+200 DDPM versus Flow Matching, seed 0

## Executive summary

All four runs are complete, artifact-valid, and strictly paired. DDPM 22D
finishes perfectly (`AP=.1000`, all task success 1.00); DDPM 43D loses only
five points of Handle success (`AP=.9875`). Ten-step FM improves substantially
over its historical seed-0 20+20 condition, but remains below DDPM: FM 22D has
final AP `.8875` and FM 43D `.9250`. FM 43D is favored over FM 22D by 3.75
percentage points (pp), while DDPM 43D is 1.25 pp below DDPM 22D. The resulting
single-seed AP interaction is +5.00 pp.

Handle retention is no longer the dominant failure: final Handle is
`.95/.80` for FM 22D/43D versus `.15/.05` in historical seed 0. FM's remaining
deficit is primarily Hammer (`.60/.90` versus 1.00 for both DDPM arms). Saved
replay arrays show FM expert-action disagreement remains about three times
DDPM despite improved retention. This is a descriptive one-seed result; the
new condition changes both data budgets and, for FM, sampling steps.

## Configuration and integrity audit

Each run has `RUN_COMPLETE`, four hash-validated stage markers, generator and
GeneralPolicy checkpoints, resolved configuration, reports, evaluation matrix,
and complete timing summary. All four use the same task order, SAC hashes,
current trajectory hashes, 160/40 current split IDs, fixed evaluation bank,
BC settings, reward v2, raw actions, and no generated-action projection. No
comparability warning was found.

The saved reports confirm 200 current trajectories per stage and fixed-total
historical allocations of `200`, `100+100`, and `67+67+66`. Historical replay
is training-only. DDPM uses 100 reverse steps; FM uses 10 Euler steps. These
are different objectives and sampling algorithms, not equivalent step units.

## Evaluation matrices

Rows are stages 0–3; columns are Hammer, Handle, Button, Reach.

| Method | Success matrix | Mean-return matrix |
|---|---|---|
| DDPM 22D | `[1,—,—,—]`<br>`[1,.95,—,—]`<br>`[1,1,1,—]`<br>`[1,1,1,1]` | `[406.2,—,—,—]`<br>`[352.5,884.0,—,—]`<br>`[356.2,956.8,1488.0,—]`<br>`[288.9,913.7,1495.4,1843.4]` |
| DDPM 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[1,1,1,—]`<br>`[1,.95,1,1]` | `[406.2,—,—,—]`<br>`[299.5,881.1,—,—]`<br>`[280.6,711.3,1490.0,—]`<br>`[250.3,708.8,1488.5,1750.4]` |
| FM 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.75,.80,1,—]`<br>`[.60,.95,1,1]` | `[406.2,—,—,—]`<br>`[304.4,932.2,—,—]`<br>`[366.3,744.6,1483.6,—]`<br>`[378.0,594.7,1485.7,1765.3]` |
| FM 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.95,.80,1,—]`<br>`[.90,.80,1,1]` | `[406.2,—,—,—]`<br>`[356.3,891.5,—,—]`<br>`[324.9,652.2,1486.2,—]`<br>`[273.9,570.5,1392.8,1803.7]` |

Both DDPM arms acquire and retain nearly perfectly. FM forgetting appears
after Button: Hammer/Handle fall to `.75/.80` in 22D and `.95/.80` in 43D.
After Reach, FM 22D Hammer falls further to `.60` while Handle recovers to
`.95`; FM 43D Hammer falls slightly to `.90` and Handle remains `.80`.
Forgetting is therefore non-monotonic and task-specific.

## AP, forgetting, and final retention

| Method | AP0 | AP1 | AP2 | AP3 | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DDPM 22D | 1.000 | .975 | 1.000 | 1.000 | 0 | 0 | −.0167 | −.0125 |
| DDPM 43D | 1.000 | 1.000 | 1.000 | .9875 | 0 | 0 | 0 | .0125 |
| FM 22D | 1.000 | 1.000 | .8500 | .8875 | 0 | 0 | .1500 | .1125 |
| FM 43D | 1.000 | 1.000 | .9167 | .9250 | 0 | 0 | .0833 | .0750 |

Negative forgetting reflects later performance exceeding the earlier maximum
under the repository's implemented definition.

| Metric | DDPM 22D | DDPM 43D | FM 22D | FM 43D |
|---|---:|---:|---:|---:|
| Final AP | 1.000 | .9875 | .8875 | .9250 |
| Final F | −.0125 | .0125 | .1125 | .0750 |
| Hammer | 1.00 | 1.00 | .60 | .90 |
| Handle | 1.00 | .95 | .95 | .80 |
| Button | 1.00 | 1.00 | 1.00 | 1.00 |
| Reach | 1.00 | 1.00 | 1.00 | 1.00 |

## Contrasts and interaction

Success differences are percentage points; positive forgetting differences
are worse.

| Contrast (left − right) | AP | F | Hammer | Handle | Button | Reach |
|---|---:|---:|---:|---:|---:|---:|
| DDPM 43D − 22D | −1.25 | +2.50 | 0 | −5 | 0 | 0 |
| FM 43D − 22D | +3.75 | −3.75 | +30 | −15 | 0 | 0 |
| FM − DDPM, 22D | −11.25 | +12.50 | −40 | −5 | 0 | 0 |
| FM − DDPM, 43D | −6.25 | +6.25 | −10 | −15 | 0 | 0 |

The single-seed descriptive representation × generator interaction is +5.00
pp for final AP and −6.25 pp for forgetting. No inferential claim is warranted.

## Replay structural quality

Structured 22D has exactly zero goal and previous-state error for every saved
group. Its action-component OOB range is `.0543–.1386` for DDPM and
`.0447–.0716` for FM; maximum violations are `.256–.690` and `.949–1.590`.

For full 43D, the principal task/stage records are:

| Method | Stage/task | Goal mean / max | Goal >1e−3 | Previous mean / max | Action component / timestep OOB | Max violation |
|---|---|---:|---:|---:|---:|---:|
| DDPM 43D | S1 Hammer | .000018/.000066 | 0 | .0142/.1262 | .1347/.4212 | .3722 |
| DDPM 43D | S2 Hammer | .002624/.01634 | .8055 | .0162/.1594 | .0823/.1779 | .4647 |
| DDPM 43D | S2 Handle | .006467/.02788 | .9896 | .0128/.0578 | .0426/.1589 | .4111 |
| DDPM 43D | S3 Hammer | .006119/.02368 | .9937 | .0191/.1985 | .1617/.3911 | .2720 |
| DDPM 43D | S3 Handle | .007877/.03110 | .9949 | .0149/.0772 | .0472/.1781 | .4414 |
| DDPM 43D | S3 Button | .004854/.02294 | .9663 | .0111/.1989 | .0403/.1187 | .6546 |
| FM 43D | S1 Hammer | .000204/.000744 | 0 | .0825/.2877 | .0800/.2471 | .9937 |
| FM 43D | S2 Hammer | .004738/.02454 | .9706 | .0994/.2611 | .0905/.2799 | .8205 |
| FM 43D | S2 Handle | .006763/.02169 | .9933 | .0586/.1637 | .0897/.3029 | 1.6044 |
| FM 43D | S3 Hammer | .007271/.02871 | .9933 | .0958/.2498 | .0562/.1814 | .8114 |
| FM 43D | S3 Handle | .010427/.02476 | .9992 | .0565/.1512 | .0746/.2555 | 1.3005 |
| FM 43D | S3 Button | .005644/.02011 | .9858 | .0496/.1460 | .0304/.0982 | 1.5431 |

Within-trajectory goal variation and reset errors are preserved in the JSON.
All replay arrays have expected `(N,200,43)` shape and zero NaN/Inf. Full 43D
previous-state tolerance violations are essentially 100% for both generators.
FM 43D previous-state errors remain several times DDPM's, while FM action-OOB
frequency is not uniformly worse; FM's violation severity is much larger.

Handle replay does not explain retention monotonically. FM 22D has exact
structure but much worse expert agreement than DDPM 22D, yet final Handle is
the same `.95` as DDPM 43D and only five points below DDPM 22D. FM 43D has
worse structure and final Handle `.80`. Replay diagnostics are correlates, not
causal retention measures.

## Expert-action disagreement

The saved arrays were analyzed with the deterministic frozen expert for each
historical task. The primary value is timestep mean squared L2, summing squared
error over four action coordinates. The aggregate below equally weights the
six stage/task groups.

| Method | Raw expert MSE | Both structure-corrected | Raw observation, clipped action |
|---|---:|---:|---:|
| DDPM 22D | .4103 | .4103 | .4050 |
| DDPM 43D | .3743 | .3810 | .3681 |
| FM 22D | 1.1820 | 1.1820 | 1.1616 |
| FM 43D | 1.1576 | 1.1347 | 1.1336 |

Handle raw disagreement at S2/S3 is `.644/.628` (DDPM 22D), `.497/.500`
(DDPM 43D), `1.799/1.680` (FM 22D), and `1.559/1.378` (FM 43D). FM remains
substantially less expert-consistent even under 200+200. Structural correction
does not close the gap. The full median, SD, P95, maximum, per-coordinate MSE,
task/stage records, and expert identities are in the JSON report.

## Efficiency

| Method | Generator training | Replay sampling | BC training | Total stage time | Replay NFE |
|---|---:|---:|---:|---:|---:|
| DDPM 22D | 1019.6 s | 76.5 s | 626.7 s | 1939.5 s | 600 |
| DDPM 43D | 1077.8 s | 85.3 s | 658.3 s | 2044.1 s | 600 |
| FM 22D | 1054.9 s | 33.4 s | 646.6 s | 1958.1 s | 60 |
| FM 43D | 1149.1 s | 34.1 s | 675.5 s | 2087.9 s | 60 |

Ten-step FM replay sampling is 2.29× faster than DDPM for 22D and 2.50× for
43D, not 10×. Total runtime is similar because generator training, collection,
BC, and evaluation dominate; FM is 1.01×/1.02× slower in total for 22D/43D.
Times are CPU, unsynchronized-device records from the same software setup;
total-stage scope excludes final artifact serialization.

## Historical 20+20 comparison

| Method | Historical five-seed mean AP | Historical seed-0 AP | New seed-0 AP | Historical seed-0 final H/Hd/B/R | New final H/Hd/B/R |
|---|---:|---:|---:|---|---|
| DDPM 22D | .7075 | .7750 | 1.0000 | .95/.50/1/.65 | 1/1/1/1 |
| DDPM 43D | .8100 | .8500 | .9875 | 1/.45/1/.95 | 1/.95/1/1 |
| FM 22D, old 100 steps | .6275 | .7875 | .8875 | 1/.15/1/1 | .60/.95/1/1 |
| FM 43D, old 100 steps | .5600 | .7375 | .9250 | .90/.05/1/1 | .90/.80/1/1 |

All new arms improve AP descriptively, especially Handle. DDPM changes current
and replay budgets. FM changes both budgets and Euler steps from 100 to 10.
These comparisons therefore cannot isolate budget, sample count, or solver
resolution. The reversal from the historical five-seed mean—43D hurt FM there
but helps in this seed—also cannot establish a robust representation effect.

## Interpretation, limitations, and next experiment

At 200+200, ten-step FM is competitive in acquisition and perfect on Button
and Reach, but it remains below DDPM in historical retention and expert-action
agreement. FM 43D is the better FM arm in final AP because it retains Hammer,
despite worse Handle and structural consistency. This is a useful counterexample
to treating any one replay diagnostic as a sufficient predictor.

Limitations are one seed, one task order, 20 evaluation episodes (five-point
resolution), different generator objectives/samplers, and a historical
comparison that changes multiple variables. Expert disagreement on generated
observations is not simulator feasibility. Negative forgetting and recovery
are metric behavior, not evidence that forgetting never occurred.

The next controlled experiment should add seeds 1–4 with identical 200+200
settings. If compute is constrained, prioritize the two FM arms plus the best
DDPM reference. A separate paired FM 10-versus-100 experiment at fixed 200+200
would isolate sampling resolution; it should reuse saved current data and split
IDs and retain expert-disagreement summaries.

Machine-readable results: [kuka_generator_200plus200_seed0_results.json](kuka_generator_200plus200_seed0_results.json).
