---
status: completed
last_verified: 2026-11-10
applies_to: continual-learning
---

# Five-seed KUKA four-stage 20+20 representation ablation

## Experiment overview

This report extends the four-stage Hammer → Handle Press Side → Button Press → Reach ablation to five matched seeds for DDPM `structured_22d` and `full_43d`. DDPM `goal_conditioned_40d` remains a supplementary three-seed result. The protocol is unchanged: 20 current expert trajectories, a fixed total of 20 historical generated trajectories, 100 DDPM epochs/steps, 100 BC epochs, raw actions, no generated-action projection, historical replay training-only, and 20 fixed-bank evaluations per learned task (task-bank/evaluation seed 10000).

The existing analyzers were run read-only:

```bash
conda run -n metaworld python scripts/evaluation/analyze_kuka_representation_four_stage.py --seeds 0,1,2,3,4
conda run -n metaworld python scripts/evaluation/analyze_kuka_expert_disagreement.py --seeds 0,1,2,3,4 \
  --output docs/experiments/kuka_expert_action_disagreement_five_seed.md \
  --json-output docs/experiments/kuka_expert_action_disagreement_five_seed.json
```

AP and Forgetting are taken directly from `src/continual/metrics.py`. In particular, Forgetting is the mean acquisition endpoint minus final endpoint including the current task's zero—not maximum-past forgetting. Standard deviations are sample SD (`ddof=1`).

## Completion and reproducibility audit

All 13 intended runs are complete and valid: 22D seeds 0–4, 43D seeds 0–4, and 40D seeds 0–2. The absent 40D seeds 3–4 are planned omissions, reported as missing and excluded from paired comparisons.

Every completed run has `RUN_COMPLETE`, four hash-validated stage markers, four generator and GeneralPolicy checkpoints, a valid finite lower triangle in its 4×4 evaluation matrix, resolved configuration, reports, and timing summary. Across each matched seed, saved hashes and metadata verify identical task order, KUKA backend/reward version, frozen SAC experts, current-task arrays, train/validation split IDs, BC settings, DDPM settings except representation width, replay budget/allocation, raw-action/no-projection settings, evaluation bank, and 20 evaluation episodes. Seeds 3–4 use the same scientific protocol as seeds 0–2. No comparability discrepancy was found.

Replay allocation is 20 Hammer at Stage 1; 10 Hammer + 10 Handle at Stage 2; and 7 Hammer + 7 Handle + 6 Button at Stage 3. Each later BC stage has 36 training and four current-task validation trajectories; historical replay is training-only.

## Full success matrices

Rows are stages; columns are Hammer, Handle, Button, Reach. `—` is unmeasured future-task performance.

| Seed | Representation | Success matrix |
|---:|---|---|
| 0 | 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.80,.95,1,—]`<br>`[.95,.50,1,.65]` |
| 1 | 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.95,.95,1,—]`<br>`[1,1,.80,.55]` |
| 2 | 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.70,.95,1,—]`<br>`[0,.70,1,.70]` |
| 3 | 22D | `[.95,—,—,—]`<br>`[.90,.70,—,—]`<br>`[.70,.80,1,—]`<br>`[.85,.65,.60,.45]` |
| 4 | 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.75,.90,1,—]`<br>`[.45,.55,.95,.80]` |
| 0 | 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.85,.85,1,—]`<br>`[1,.45,1,.95]` |
| 1 | 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[1,.95,1,—]`<br>`[.80,1,1,.75]` |
| 2 | 43D | `[1,—,—,—]`<br>`[.80,1,—,—]`<br>`[.95,1,1,—]`<br>`[.95,.35,.90,.60]` |
| 3 | 43D | `[.95,—,—,—]`<br>`[1,.65,—,—]`<br>`[1,.90,1,—]`<br>`[.95,.90,1,.35]` |
| 4 | 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[1,1,1,—]`<br>`[1,.90,1,.35]` |
| 0 | 40D | `[1,—,—,—]`<br>`[.85,1,—,—]`<br>`[.80,.95,1,—]`<br>`[1,.40,1,1]` |
| 1 | 40D | `[1,—,—,—]`<br>`[.85,1,—,—]`<br>`[.80,.85,1,—]`<br>`[.75,.15,.85,.55]` |
| 2 | 40D | `[1,—,—,—]`<br>`[.50,1,—,—]`<br>`[.50,.85,1,—]`<br>`[.90,.70,1,.70]` |

Performance is non-monotonic: seed-3 43D Hammer improves from .95 to 1.00 and finishes .95; seed-2 40D Hammer falls to .50 then recovers to .90; seed-2 22D Hammer collapses only at the final stage.

## Complete mean-return matrices

| Seed | Representation | Return matrix |
|---:|---|---|
| 0 | 22D | `[397.8,—,—,—]`<br>`[231.3,871.4,—,—]`<br>`[261.4,926.5,1478.0,—]`<br>`[203.6,551.0,1467.5,1714.3]` |
| 1 | 22D | `[388.7,—,—,—]`<br>`[379.3,883.4,—,—]`<br>`[376.8,738.4,1485.4,—]`<br>`[272.2,722.9,1308.9,1609.1]` |
| 2 | 22D | `[347.7,—,—,—]`<br>`[319.6,956.9,—,—]`<br>`[405.9,938.9,1478.5,—]`<br>`[361.5,887.8,1420.6,1653.3]` |
| 3 | 22D | `[372.9,—,—,—]`<br>`[410.3,792.6,—,—]`<br>`[286.0,687.8,1466.6,—]`<br>`[292.4,835.3,1336.3,1652.5]` |
| 4 | 22D | `[390.0,—,—,—]`<br>`[347.9,918.2,—,—]`<br>`[331.3,794.7,1488.9,—]`<br>`[267.5,426.2,1382.6,1651.2]` |
| 0 | 43D | `[397.8,—,—,—]`<br>`[269.4,919.7,—,—]`<br>`[278.3,828.1,1482.8,—]`<br>`[324.0,640.1,1485.3,1761.1]` |
| 1 | 43D | `[388.7,—,—,—]`<br>`[315.3,911.5,—,—]`<br>`[379.0,982.1,1491.1,—]`<br>`[198.4,1030.4,1392.5,1675.5]` |
| 2 | 43D | `[347.7,—,—,—]`<br>`[353.0,826.1,—,—]`<br>`[388.8,837.7,1466.5,—]`<br>`[406.0,649.8,1449.4,1541.8]` |
| 3 | 43D | `[372.9,—,—,—]`<br>`[304.7,803.2,—,—]`<br>`[342.7,854.4,1493.2,—]`<br>`[353.2,996.1,1477.6,1429.3]` |
| 4 | 43D | `[390.0,—,—,—]`<br>`[306.6,967.3,—,—]`<br>`[251.9,904.9,1476.6,—]`<br>`[282.2,631.3,1502.9,1670.7]` |
| 0 | 40D | `[397.8,—,—,—]`<br>`[300.8,947.5,—,—]`<br>`[352.6,855.2,1472.0,—]`<br>`[289.6,581.7,1462.9,1732.1]` |
| 1 | 40D | `[388.7,—,—,—]`<br>`[357.4,920.8,—,—]`<br>`[290.8,596.4,1487.3,—]`<br>`[192.1,450.9,1381.5,1646.8]` |
| 2 | 40D | `[347.7,—,—,—]`<br>`[244.9,909.5,—,—]`<br>`[254.5,875.6,1474.1,—]`<br>`[187.1,736.5,1457.3,1660.0]` |

## Stage-wise AP and Forgetting

| Seed | Rep | AP0 | AP1 | AP2 | AP3 | F0 | F1 | F2 | F3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22D | 1.000 | 1.000 | .917 | .775 | 0 | 0 | .083 | .138 |
| 1 | 22D | 1.000 | 1.000 | .967 | .837 | 0 | 0 | .033 | .050 |
| 2 | 22D | 1.000 | 1.000 | .883 | .600 | 0 | 0 | .117 | .325 |
| 3 | 22D | .950 | .800 | .833 | .638 | 0 | .025 | .050 | .137 |
| 4 | 22D | 1.000 | 1.000 | .883 | .688 | 0 | 0 | .117 | .263 |
| 0 | 43D | 1.000 | 1.000 | .900 | .850 | 0 | 0 | .100 | .138 |
| 1 | 43D | 1.000 | 1.000 | .983 | .887 | 0 | 0 | .017 | .050 |
| 2 | 43D | 1.000 | .900 | .983 | .700 | 0 | .100 | .017 | .200 |
| 3 | 43D | .950 | .825 | .967 | .800 | 0 | −.025 | −.100 | −.062 |
| 4 | 43D | 1.000 | 1.000 | 1.000 | .812 | 0 | 0 | 0 | .025 |
| 0 | 40D | 1.000 | .925 | .917 | .850 | 0 | .075 | .083 | .150 |
| 1 | 40D | 1.000 | .925 | .883 | .575 | 0 | .075 | .117 | .312 |
| 2 | 40D | 1.000 | .750 | .783 | .825 | 0 | .250 | .217 | .100 |

Five-seed stage aggregates:

| Rep | AP0 | AP1 | AP2 | AP3 | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 22D | .990±.022 | .960±.089 | .897±.049 | .708±.098 | 0±0 | .005±.011 | .080±.038 | .183±.110 |
| 43D | .990±.022 | .945±.080 | .967±.039 | .810±.070 | 0±0 | .015±.048 | .007±.075 | .070±.102 |

Supplementary 40D (n=3): AP `1.000±0/.867±.101/.861±.069/.750±.152`; F `0±0/.133±.101/.139±.069/.188±.111` for stages 0–3.

## Final task-wise performance

| Seed | Rep | Hammer | Handle | Button | Reach | Final AP | Final F |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 22D | .95 | .50 | 1.00 | .65 | .775 | .138 |
| 1 | 22D | 1.00 | 1.00 | .80 | .55 | .837 | .050 |
| 2 | 22D | 0 | .70 | 1.00 | .70 | .600 | .325 |
| 3 | 22D | .85 | .65 | .60 | .45 | .638 | .137 |
| 4 | 22D | .45 | .55 | .95 | .80 | .688 | .263 |
| 0 | 43D | 1.00 | .45 | 1.00 | .95 | .850 | .138 |
| 1 | 43D | .80 | 1.00 | 1.00 | .75 | .887 | .050 |
| 2 | 43D | .95 | .35 | .90 | .60 | .700 | .200 |
| 3 | 43D | .95 | .90 | 1.00 | .35 | .800 | −.062 |
| 4 | 43D | 1.00 | .90 | 1.00 | .35 | .812 | .025 |

### Primary five-seed aggregation

| Metric | Structured 22D | Full 43D |
|---|---:|---:|
| Final AP | .7075±.0979 | **.8100±.0704** |
| Final Forgetting | .1825±.1099 | **.0700±.1018** |
| Hammer | .6500±.4228 | **.9400±.0822** |
| Handle | .6800±.1956 | .7200±.2971 |
| Button | .8700±.1718 | **.9800±.0447** |
| Reach | **.6300±.1351** | .6000±.2598 |
| Valid n | 5 | 5 |

Acquisition-to-final changes show where AP differences arise:

| Rep | Hammer | Handle | Button | Reach |
|---|---:|---:|---:|---:|
| 22D | −.340±.429 | −.260±.227 | −.130±.172 | 0±0 |
| 43D | −.050±.087 | −.210±.380 | −.020±.045 | 0±0 |

Thus the aggregate advantage is mainly historical retention—especially Hammer (+29 pp final difference) and Button (+11 pp)—not current Reach acquisition, where 43D is 3 pp lower on average. Handle contributes only +4 pp with large variability.

### Supplementary three-seed 40D

| Metric | 40D mean±SD (n=3) |
|---|---:|
| Final AP | .7500±.1521 |
| Final F | .1875±.1111 |
| Hammer / Handle / Button / Reach | .8833±.1258 / .4167±.2754 / .9500±.0866 / .7500±.2291 |

Unequal seed coverage prevents an equal-footing three-method ranking.

## Paired 43D − 22D comparison

Differences are percentage points; negative Forgetting favors 43D.

| Seed | Δ Final AP | Δ F | Δ Hammer | Δ Handle | Δ Button | Δ Reach |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +7.5 | 0.0 | +5 | −5 | 0 | +30 |
| 1 | +5.0 | 0.0 | −20 | 0 | +20 | +20 |
| 2 | +10.0 | −12.5 | +95 | −35 | −10 | −10 |
| 3 | +16.3 | −20.0 | +10 | +25 | +40 | −10 |
| 4 | +12.5 | −23.8 | +55 | +35 | +5 | −45 |
| **Mean±SD** | **+10.3±4.4** | **−11.3±11.0** | +29.0±45.7 | +4.0±27.5 | +11.0±19.5 | −3.0±29.5 |

43D has higher final AP in 5/5 seeds, with no ties or 22D wins. It has equal F in seeds 0–1 and lower F in seeds 2–4; it never has higher final F. The direction is consistent for AP but not for individual tasks. Hammer is the largest mean contributor but also contains the largest outlier (seed 2, +95 pp). The new seeds remain favorable without that seed: seed 3 is +16.3 pp and seed 4 +12.5 pp.

An exploratory paired Student-t interval (independent seeds, approximately normal paired differences) gives a 95% CI of **[+4.8,+15.7] pp** for mean ΔAP and **[−25.0,+2.5] pp** for mean ΔF. With n=5 these intervals are descriptive and assumption-sensitive; the Forgetting interval includes zero.

## Structural replay diagnostics

22D reconstructs goals and previous state exactly. Its original reports omit structural fields, so zero goal/previous errors are contract-derived and action statistics are recomputed read-only from saved arrays. 43D fields below are recorded. All saved groups have zero NaNs/Infs and valid shapes.

Seeds 3–4 reproduce 43D goal drift:

| Seed | Stage/task | Goal mean | Goal max | Within variation | >1e−3 | Previous mean | Previous max | Action component/step OOB | Max violation |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 3 | S1 Hammer | .000080 | .000271 | .000085 | 0 | .03335 | .27877 | .0941/.2455 | .5720 |
| 3 | S2 Hammer | .005347 | .020253 | .006218 | .9535 | .05132 | .14784 | .0959/.3760 | .1839 |
| 3 | S2 Handle | .009006 | .029032 | .011180 | .9965 | .04041 | .09465 | .1534/.4555 | .6492 |
| 3 | S3 Hammer | .006101 | .014617 | .006942 | .9607 | .05731 | .13862 | .1107/.4150 | .1305 |
| 3 | S3 Handle | .010475 | .023757 | .010974 | .9979 | .04464 | .10302 | .1500/.4250 | .7254 |
| 3 | S3 Button | .009908 | .033210 | .014913 | .9992 | .03447 | .24115 | .0346/.0892 | .5186 |
| 4 | S1 Hammer | .000073 | .000324 | .000213 | 0 | .03870 | .10039 | .0944/.2990 | .4054 |
| 4 | S2 Hammer | .005083 | .013134 | .013183 | 1.0000 | .05306 | .08961 | .0973/.3890 | .0751 |
| 4 | S2 Handle | .014077 | .042744 | .012799 | .9980 | .04067 | .09323 | .1400/.4245 | .8179 |
| 4 | S3 Hammer | .006036 | .010638 | .011506 | .9943 | .05228 | .09903 | .0695/.2779 | .0365 |
| 4 | S3 Handle | .016036 | .048490 | .010596 | 1.0000 | .04333 | .12687 | .1554/.4829 | .8381 |
| 4 | S3 Button | .012643 | .027484 | .015791 | 1.0000 | .02990 | .16621 | .0613/.1833 | .7742 |

Stage 1 remains below tolerance in both seeds. Later task-specific mean deviations are .0051–.0160 and 95.4–100% of timesteps violate the strict threshold. This replicates seeds 0–2. It establishes generated-goal error, not a causal self-replay mechanism. Previous-state tolerance violation is 100% for every listed 43D group; magnitude is not interpreted as metres.

## Expert-action disagreement

The metric averages timestep squared L2 error, where each timestep error **sums across four action coordinates**. Aggregate values below equally weight the 30 saved task/stage/seed groups per representation; because every timestep has equal horizon but groups contain 6–20 trajectories, this is not globally timestep-weighted. Full distributions and per-dimension MSE are in the JSON companion.

| Rep | Raw observation/action | Both structural corrections | Raw observation/clipped action |
|---|---:|---:|---:|
| 22D | .722122 | .722122 | .677479 |
| 43D | **.601515** | .635362 | **.570650** |
| 40D (n=3 supplementary) | .693236 | .716620 | .640106 |

| Task | 22D raw | 43D raw | Difference 43D−22D |
|---|---:|---:|---:|
| Hammer | .505098 | .337480 | −.167618 |
| Handle | .972523 | .894899 | −.077624 |
| Button | .872391 | .806855 | −.065536 |

43D retains its aggregate expert-agreement advantage and is lower for each task aggregate. Structural correction leaves 22D unchanged and increases 43D's aggregate disagreement; it therefore does not explain the advantage. Clipping improves both, without reversing their ordering. Real deterministic expert data agrees to numerical precision (maximum group mean approximately `2.3e−12`).

Seeds 3–4 contain supporting and opposing examples. Seed-4 43D Hammer has lower disagreement than 22D at every stage and retains 1.00 versus .45. Conversely, seed-3 Stage-2 Handle disagreement is slightly higher for 43D (.9312 versus .9182), yet final Handle success is .90 versus .65. Expert agreement is informative descriptively, not a complete predictor.

See [five-seed disagreement report](kuka_expert_action_disagreement_five_seed.md) and [machine-readable disagreement results](kuka_expert_action_disagreement_five_seed.json).

## Relationship between replay quality and retention

Observed evidence does not support structural consistency or action OOB as sufficient predictors. Exact 22D structure coexists with lower mean AP and higher F. 43D's recorded drift/inconsistency coexists with better historical retention. Action OOB overlaps heavily and has counterexamples: seed-3 43D Handle has higher component OOB than 22D at Stage 2 but better retention; seed-4 43D Hammer has comparable OOB yet far better retention.

Expert-action agreement aligns better in aggregate: 43D is lower overall and for every historical task while retaining Hammer/Button better. It still fails at group level and cannot explain Reach, which has no historical replay group. The association may reflect more useful generated current-state/action combinations, but off-distribution expert predictions, task interference, BC checkpoint selection, and replay dilution remain confounded.

## Scientific interpretation

1. **Final AP:** 43D is higher in all five seeds. The original three-seed direction replicates and strengthens descriptively.
2. **Forgetting:** 43D ties 22D in two seeds and is lower in three; it is never higher. The exploratory paired CI still includes zero.
3. **Task contributions:** Hammer and Button drive the mean advantage; Handle is variable; Reach favors 22D on average. Seed-2 Hammer is influential but does not determine the AP conclusion.
4. **Acquisition versus retention:** Stage-0 acquisition is matched, and 43D's Reach acquisition is not better. The five-seed advantage is principally historical retention.
5. **Goal drift:** The Stage-1-small/later-large 43D pattern replicates in seeds 3–4, with later violations of 95.4–100%.
6. **Expert agreement:** 43D retains lower aggregate disagreement, but task/group counterexamples rule out a deterministic relationship with retention.
7. **Structural and action diagnostics:** Neither exact reconstruction nor lower OOB consistently predicts downstream performance.
8. **Seed variation:** 22D Hammer SD is .423 and paired task effects have large SDs. Five seeds reduce, but do not eliminate, uncertainty.
9. **Scope:** These data support further investigation of 43D under this sequence/budget, not universal superiority or a causal benefit from inconsistency.

## Limitations

- Five seeds, one task order, one fixed-total replay budget, and 20 evaluation episodes per cell limit inference.
- Success rates have 5-point resolution; seed runs are the replication unit, not task/stage replay groups.
- Equal group weighting in disagreement differs from global timestep weighting; conclusions are unchanged only descriptively, not proven invariant.
- Expert agreement does not establish dynamics feasibility, reward quality, or closed-loop usefulness and may evaluate off-distribution observations.
- Current-task-only validation can trade new-task selection against historical retention.
- Fixed-total replay simultaneously reduces per-task counts as the number of historical tasks grows.
- No corrected/clipped replay method was trained; diagnostic correction cannot imply performance improvement.

## Recommended follow-up

The next controlled experiment should compare 22D and 43D under **fixed-per-task replay**, using the same five paired seeds if affordable. Holding historical support constant would separate representation effects from fixed-total replay dilution. A smaller secondary ablation could compare current-task-only versus mixed historical/current validation, but it should not replace the replay-budget control.
