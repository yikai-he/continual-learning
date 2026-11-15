---
status: completed
last_verified: 2026-11-09
applies_to: continual-learning
---

# KUKA four-stage 20+20 trajectory-representation ablation (seeds 0–2)

## Experiment overview

This report analyzes the completed sequence Hammer → Handle Press Side → Button Press → Reach for DDPM `structured_22d`, `goal_conditioned_40d`, and `full_43d`. Each stage collected 20 current-task expert trajectories. Stages 1–3 added a fixed total of 20 generated historical trajectories, used for training only. DDPM and BC used 100 epochs; sampling used 100 diffusion steps; actions were raw and generated-action projection was disabled. Each learned task was evaluated on 20 fixed-bank episodes (evaluation/task-bank seed 10000). Training seeds were 0, 1, and 2.

All values below come from existing artifacts. The repository analyzer was run read-only as:

```bash
conda run -n metaworld python scripts/evaluation/analyze_kuka_representation_four_stage.py
```

Average Performance (AP) and Forgetting (F) are the implementations in `src/continual/metrics.py`; they are not redefined here. Standard deviations are sample SD (`ddof=1`).

## Completion and reproducibility audit

All 9 planned runs are **complete and valid**. Every run has `RUN_COMPLETE`, four validated stage markers (including artifact hashes), four DDPM checkpoints, four GeneralPolicy checkpoints, a finite lower-triangular 4×4 evaluation matrix, resolved configuration, reports, and timing summary.

| Seed | 22D | 40D | 43D |
|---:|---|---|---|
| 0 | complete/valid | complete/valid | complete/valid |
| 1 | complete/valid | complete/valid | complete/valid |
| 2 | complete/valid | complete/valid | complete/valid |

Within each seed, saved hashes and metadata confirm identical SAC expert checkpoints, task order, current-task trajectory arrays, episode split IDs, BC settings, DDPM settings apart from representation dimensions, evaluation bank, replay budgets, and action settings. Expert hashes also match across seeds. No unintended comparability difference was detected.

Recorded allocation is identical in every run:

| Stage | Current | Historical | Historical allocation | BC train | Validation |
|---:|---:|---:|---|---:|---:|
| 0 | 20 | 0 | none | 16 | 4 |
| 1 | 20 | 20 | Hammer 20 | 36 | 4 |
| 2 | 20 | 20 | Hammer 10, Handle 10 | 36 | 4 |
| 3 | 20 | 20 | Hammer 7, Handle 7, Button 6 | 36 | 4 |

Historical replay is training-only in all cases. Validation contains four current-task episodes.

## Full success matrices

Rows are stages 0–3; columns are Hammer, Handle, Button, Reach. `—` denotes a future task.

| Seed | Representation | Success matrix |
|---:|---|---|
| 0 | 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.80,.95,1,—]`<br>`[.95,.50,1,.65]` |
| 0 | 40D | `[1,—,—,—]`<br>`[.85,1,—,—]`<br>`[.80,.95,1,—]`<br>`[1,.40,1,1]` |
| 0 | 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.85,.85,1,—]`<br>`[1,.45,1,.95]` |
| 1 | 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.95,.95,1,—]`<br>`[1,1,.80,.55]` |
| 1 | 40D | `[1,—,—,—]`<br>`[.85,1,—,—]`<br>`[.80,.85,1,—]`<br>`[.75,.15,.85,.55]` |
| 1 | 43D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[1,.95,1,—]`<br>`[.80,1,1,.75]` |
| 2 | 22D | `[1,—,—,—]`<br>`[1,1,—,—]`<br>`[.70,.95,1,—]`<br>`[0,.70,1,.70]` |
| 2 | 40D | `[1,—,—,—]`<br>`[.50,1,—,—]`<br>`[.50,.85,1,—]`<br>`[.90,.70,1,.70]` |
| 2 | 43D | `[1,—,—,—]`<br>`[.80,1,—,—]`<br>`[.95,1,1,—]`<br>`[.95,.35,.90,.60]` |

The non-monotonic cells matter: for example, seed-2 40D Hammer falls to 0.50 and recovers to 0.90, whereas seed-2 22D Hammer falls to 0.70 and then collapses to 0.00.

### Complete mean-return matrices

The corresponding mean episode returns (same layout) are:

| Seed | Representation | Return matrix |
|---:|---|---|
| 0 | 22D | `[397.8,—,—,—]`<br>`[231.3,871.4,—,—]`<br>`[261.4,926.5,1478.0,—]`<br>`[203.6,551.0,1467.5,1714.3]` |
| 0 | 40D | `[397.8,—,—,—]`<br>`[300.8,947.5,—,—]`<br>`[352.6,855.2,1472.0,—]`<br>`[289.6,581.7,1462.9,1732.1]` |
| 0 | 43D | `[397.8,—,—,—]`<br>`[269.4,919.7,—,—]`<br>`[278.3,828.1,1482.8,—]`<br>`[324.0,640.1,1485.3,1761.1]` |
| 1 | 22D | `[388.7,—,—,—]`<br>`[379.3,883.4,—,—]`<br>`[376.8,738.4,1485.4,—]`<br>`[272.2,722.9,1308.9,1609.1]` |
| 1 | 40D | `[388.7,—,—,—]`<br>`[357.4,920.8,—,—]`<br>`[290.8,596.4,1487.3,—]`<br>`[192.1,450.9,1381.5,1646.8]` |
| 1 | 43D | `[388.7,—,—,—]`<br>`[315.3,911.5,—,—]`<br>`[379.0,982.1,1491.1,—]`<br>`[198.4,1030.4,1392.5,1675.5]` |
| 2 | 22D | `[347.7,—,—,—]`<br>`[319.6,956.9,—,—]`<br>`[405.9,938.9,1478.5,—]`<br>`[361.5,887.8,1420.6,1653.3]` |
| 2 | 40D | `[347.7,—,—,—]`<br>`[244.9,909.5,—,—]`<br>`[254.5,875.6,1474.1,—]`<br>`[187.1,736.5,1457.3,1660.0]` |
| 2 | 43D | `[347.7,—,—,—]`<br>`[353.0,826.1,—,—]`<br>`[388.8,837.7,1466.5,—]`<br>`[406.0,649.8,1449.4,1541.8]` |

All evaluation episode lengths were 200.

## Stage-wise continual metrics

| Seed | Representation | AP0 | AP1 | AP2 | AP3 | F0 | F1 | F2 | F3 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22D | 1.000 | 1.000 | .917 | .775 | 0 | 0 | .083 | .138 |
| 0 | 40D | 1.000 | .925 | .917 | .850 | 0 | .075 | .083 | .150 |
| 0 | 43D | 1.000 | 1.000 | .900 | .850 | 0 | 0 | .100 | .138 |
| 1 | 22D | 1.000 | 1.000 | .967 | .837 | 0 | 0 | .033 | .050 |
| 1 | 40D | 1.000 | .925 | .883 | .575 | 0 | .075 | .117 | .312 |
| 1 | 43D | 1.000 | 1.000 | .983 | .887 | 0 | 0 | .017 | .050 |
| 2 | 22D | 1.000 | 1.000 | .883 | .600 | 0 | 0 | .117 | .325 |
| 2 | 40D | 1.000 | .750 | .783 | .825 | 0 | .250 | .217 | .100 |
| 2 | 43D | 1.000 | .900 | .983 | .700 | 0 | .100 | .017 | .200 |

### Stage aggregates (mean ± sample SD, n=3)

| Representation | AP0 | AP1 | AP2 | AP3 |
|---|---:|---:|---:|---:|
| 22D | 1.000±0 | 1.000±0 | .922±.042 | .738±.123 |
| 40D | 1.000±0 | .867±.101 | .861±.069 | .750±.152 |
| 43D | 1.000±0 | .967±.058 | .956±.048 | .813±.099 |

| Representation | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| 22D | 0±0 | 0±0 | .078±.042 | .171±.140 |
| 40D | 0±0 | .133±.101 | .139±.069 | .188±.111 |
| 43D | 0±0 | .033±.058 | .044±.048 | .129±.075 |

## Final performance and retention

| Seed | Representation | Hammer | Handle | Button | Reach | Final AP | Final F |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 22D | .95 | .50 | 1.00 | .65 | .775 | .138 |
| 0 | 40D | 1.00 | .40 | 1.00 | 1.00 | .850 | .150 |
| 0 | 43D | 1.00 | .45 | 1.00 | .95 | .850 | .138 |
| 1 | 22D | 1.00 | 1.00 | .80 | .55 | .837 | .050 |
| 1 | 40D | .75 | .15 | .85 | .55 | .575 | .312 |
| 1 | 43D | .80 | 1.00 | 1.00 | .75 | .887 | .050 |
| 2 | 22D | 0 | .70 | 1.00 | .70 | .600 | .325 |
| 2 | 40D | .90 | .70 | 1.00 | .70 | .825 | .100 |
| 2 | 43D | .95 | .35 | .90 | .60 | .700 | .200 |

| Metric | DDPM 22D | DDPM 40D | DDPM 43D |
|---|---:|---:|---:|
| Final AP | **.7375±.1231** | **.7500±.1521** | **.8125±.0992** |
| Final forgetting | .1708±.1405 | .1875±.1111 | **.1292±.0753** |
| Hammer | .6500±.5635 | .8833±.1258 | .9167±.1041 |
| Handle | .7333±.2517 | .4167±.2754 | .6000±.3500 |
| Button | .9333±.1155 | .9500±.0866 | .9667±.0577 |
| Reach | .6333±.0764 | .7500±.2291 | .7667±.1756 |
| n | 3 | 3 | 3 |

Acquisition-to-final changes (negative is loss) separate retention from current-task acquisition:

| Representation | Hammer | Handle | Button | Reach |
|---|---:|---:|---:|---:|
| 22D | −.350±.563 | −.267±.252 | −.067±.115 | 0±0 |
| 40D | −.117±.126 | −.583±.275 | −.050±.087 | 0±0 |
| 43D | −.083±.104 | −.400±.350 | −.033±.058 | 0±0 |

## Paired representation differences

Values are percentage points (pp); positive AP/task values favor the first representation, while negative F favors it. All comparisons have three matched seeds.

| Comparison | Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| 40D−22D | Final AP | +7.5 | −26.3 | +22.5 | +1.3±25.0 pp |
|  | Final F | +1.3 | +26.3 | −22.5 | +1.7±24.4 pp |
|  | Hammer | +5 | −25 | +90 | +23.3±59.7 pp |
|  | Handle | −10 | −85 | 0 | −31.7±46.5 pp |
|  | Button | 0 | +5 | 0 | +1.7±2.9 pp |
|  | Reach | +35 | 0 | 0 | +11.7±20.2 pp |
| 43D−22D | Final AP | +7.5 | +5.0 | +10.0 | +7.5±2.5 pp |
|  | Final F | 0 | 0 | −12.5 | −4.2±7.2 pp |
|  | Hammer | +5 | −20 | +95 | +26.7±60.5 pp |
|  | Handle | −5 | 0 | −35 | −13.3±18.9 pp |
|  | Button | 0 | +20 | −10 | +3.3±15.3 pp |
|  | Reach | +30 | +20 | −10 | +13.3±20.8 pp |
| 43D−40D | Final AP | 0 | +31.3 | −12.5 | +6.3±22.5 pp |
|  | Final F | −1.3 | −26.3 | +10.0 | −5.8±18.6 pp |
|  | Hammer | 0 | +5 | +5 | +3.3±2.9 pp |
|  | Handle | +5 | +85 | −35 | +18.3±61.1 pp |
|  | Button | 0 | +15 | −10 | +1.7±12.6 pp |
|  | Reach | −5 | +20 | −10 | +1.7±16.1 pp |

The 43D–22D final-AP difference has the most consistent sign, but three seeds are insufficient for a significance claim. Task-level differences are much more variable.

## Replay diagnostics

All saved generated arrays are finite and have valid shapes: NaN=0, Inf=0, invalid-shape=false for every available group. The concise table below aggregates each replay stage across its historical groups. Goal/previous-state values are means; action values are component-OOB/timestep-OOB/max violation. For 22D, original diagnostic fields were absent, so values marked `R` were reconstructed read-only from saved decoded arrays using the analyzer's diagnostic definitions. Exact goal and previous-state reconstruction are structural properties of 22D.

| Seed | Rep | Stage | Goal mean / violation | Previous mean / violation | Action OOB component / step / max | Source |
|---:|---|---:|---|---|---|---|
| 0 | 22D | 1 | 0 / 0 | 0 / 0 | .0341 / .1110 / .4780 | R |
| 0 | 22D | 2 | 0 / 0 | 0 / 0 | .2524 / .6645 / 1.1450 | R |
| 0 | 22D | 3 | 0 / 0 | 0 / 0 | .1927 / .5255 / 1.1409 | R |
| 1 | 22D | 1 | 0 / 0 | 0 / 0 | .0400 / .1170 / 1.4650 | R |
| 1 | 22D | 2 | 0 / 0 | 0 / 0 | .0795 / .2595 / .9758 | R |
| 1 | 22D | 3 | 0 / 0 | 0 / 0 | .0752 / .2305 / 1.0708 | R |
| 2 | 22D | 1 | 0 / 0 | 0 / 0 | .0661 / .2015 / .3088 | R |
| 2 | 22D | 2 | 0 / 0 | 0 / 0 | .0956 / .3253 / 1.0015 | R |
| 2 | 22D | 3 | 0 / 0 | 0 / 0 | .1783 / .4798 / 1.7325 | R |
| 0 | 40D | 1 | 0 / 0 | .0203 / 1 | .1008 / .2925 / .6131 | recorded |
| 0 | 40D | 2 | 0 / 0 | .0372 / 1 | .1380 / .3523 / .8628 | recorded |
| 0 | 40D | 3 | 0 / 0 | .0411 / 1 | .1425 / .3575 / .9811 | recorded |
| 1 | 40D | 1 | 0 / 0 | .0426 / 1 | .2868 / .5440 / .5217 | recorded |
| 1 | 40D | 2 | 0 / 0 | .0473 / 1 | .1557 / .3958 / .9409 | recorded |
| 1 | 40D | 3 | 0 / 0 | .0466 / 1 | .0910 / .2613 / 1.4218 | recorded |
| 2 | 40D | 1 | 0 / 0 | .0165 / 1 | 0 / 0 / 0 | recorded |
| 2 | 40D | 2 | 0 / 0 | .0328 / 1 | .0744 / .2168 / 1.0324 | recorded |
| 2 | 40D | 3 | 0 / 0 | .0370 / 1 | .1136 / .2800 / 1.5463 | recorded |
| 0 | 43D | 1 | .000087 / 0 | .0255 / 1 | .0482 / .1783 / .1484 | recorded |
| 0 | 43D | 2 | .008200 / .9893 | .0472 / 1 | .1633 / .5960 / .7476 | recorded |
| 0 | 43D | 3 | .009200 / .9953 | .0452 / 1 | .1456 / .4981 / .8440 | recorded |
| 1 | 43D | 1 | .000065 / 0 | .0351 / 1 | .1916 / .4833 / .5828 | recorded |
| 1 | 43D | 2 | .006500 / .9748 | .0439 / 1 | .1366 / .3880 / .7392 | recorded |
| 1 | 43D | 3 | .009623 / .9993 | .0475 / 1 | .1318 / .3796 / .7984 | recorded |
| 2 | 43D | 1 | .000089 / 0 | .0254 / 1 | .0349 / .1195 / .1893 | recorded |
| 2 | 43D | 2 | .008409 / .9918 | .0331 / 1 | .0977 / .3280 / 1.1115 | recorded |
| 2 | 43D | 3 | .008492 / .9925 | .0367 / 1 | .1308 / .3919 / 1.3561 | recorded |

Task-specific recorded diagnostics are preserved in each stage report and were printed by the analyzer for all 40D/43D groups. Their main patterns are:

- 40D goal deviation/variation/violation is exactly zero for every historical task. Its task-specific previous-state mean spans .0165–.0668, and every timestep exceeds the configured alignment tolerance.
- 43D stage-1 Hammer goal means are only 0.000065–0.000089 with zero tolerance violations. At stages 2–3, task-specific means range .00370–.01615 and violation fractions .956–1.000. Handle has the largest task-specific goal mean in all later-stage seed comparisons (.00919–.01615), while Hammer is generally lower (.00370–.00745).
- 43D task-specific previous-state means span .0254–.0636; all violation fractions are 1.0. Button is usually lowest at stage 3 (.0290–.0340).
- Component action OOB varies strongly by task and seed, including exact zero for seed-2 40D Hammer and values above .25 for seed-2 Handle. It is therefore not a representation-only property.

The full per-task tuple reported by the analyzer is `goal mean/max/within-trajectory variation/violation`, `previous mean/max/violation/reset error`, and `action component OOB/timestep OOB/max violation`. Component-wise previous-state diagnostics (end-effector, object position, gripper, orientation) were not persisted and cannot be recovered unambiguously from the aggregate reports without introducing a new analysis implementation; they are reported as unavailable rather than fabricated. The full 18D error is not interpreted as metres.

### Full-43D goal drift

The later-stage drift replicates in every seed. Stage-1 maximum deviations are .000205–.000330 and never cross the 1e−3 tolerance. Aggregate maximum deviations at stage 2 are .0359–.0391 and at stage 3 .0357–.0424; approximately 97.5–99.9% of later generated timesteps violate the tolerance. These are small absolute observation-coordinate errors but 4–16 mm-scale mean coordinate deviations if the goal coordinates are metre-valued, and maxima near 4 cm. The tolerance result alone exaggerates their practical meaning because 1e−3 is strict. The artifacts establish increasing generated-goal error, not that generator self-replay causes it.

### Previous-state consistency and action validity

40D previous-state mean increases from stage 1 to stage 3 in seeds 0 and 2, but seed 1 begins high and is nearly flat. 43D increases overall in all three seeds, though seed 0 peaks at stage 2. Thus growth is a descriptive tendency, not strictly monotonic behavior. By construction, 22D has exact alignment.

Structural consistency does not reliably predict retention: 22D is exact yet has the lowest mean Hammer success and a seed-2 collapse, while 43D has both goal drift and previous-state inconsistency but the highest mean AP and lowest mean F. Action validity also fails as a simple predictor: seed-2 40D Hammer has zero action OOB but only .90 final success; seed-0 22D has substantial later OOB yet .95 Hammer; and Handle retention varies sharply at overlapping OOB levels. These comparisons are correlational and do not establish causal irrelevance.

## BC training and checkpoint selection

Stage-0 current data and splits are identical within each seed, and the three Stage-0 BC outcomes are identical within numerical reporting precision. At later stages each BC dataset has 36 training trajectories (7,200 timesteps at horizon 200) and four current-task validation trajectories (800 timesteps). Reports retain complete `bc_losses` histories, selected best epoch, and per-stage BC timing.

Each cell below is selected train loss / selected current-task validation loss / best epoch / BC seconds:

| Seed | Rep | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---:|---|---|---|---|---|
| 0 | 22D | .0028/.1131/e95/7.91s | .0083/.0296/e72/19.06s | .0144/.0094/e92/17.84s | .0448/.1572/e30/18.52s |
| 0 | 40D | .0028/.1131/e95/8.07s | .0071/.0239/e98/18.74s | .0097/.0105/e98/17.87s | .0246/.1436/e67/17.47s |
| 0 | 43D | .0028/.1131/e95/7.93s | .0073/.0269/e100/18.65s | .0069/.0078/e100/19.73s | .0188/.1506/e40/19.00s |
| 1 | 22D | .0053/.0171/e95/7.78s | .0070/.0089/e95/17.39s | .0118/.0272/e86/16.95s | .0387/.0843/e19/16.91s |
| 1 | 40D | .0053/.0171/e95/7.76s | .0079/.0093/e98/17.57s | .0138/.0259/e52/16.71s | .0456/.1121/e12/17.44s |
| 1 | 43D | .0053/.0171/e95/7.62s | .0082/.0127/e85/16.92s | .0099/.0235/e79/16.36s | .0116/.1058/e72/16.16s |
| 2 | 22D | .0059/.0217/e98/7.45s | .0058/.0439/e95/16.58s | .0180/.0265/e96/16.15s | .0329/.1602/e34/15.83s |
| 2 | 40D | .0059/.0217/e98/7.57s | .0073/.0390/e78/16.78s | .0119/.0285/e98/16.15s | .0421/.1585/e32/15.92s |
| 2 | 43D | .0059/.0217/e98/7.65s | .0116/.0417/e68/17.03s | .0147/.0287/e80/16.42s | .0701/.1763/e22/16.96s |

Later-stage validation losses and selected epochs differ among representations, but no representation has a uniform advantage across seeds/stages. Because validation is current-task-only while training includes historical replay, selecting the minimum current-task validation checkpoint may trade historical retention against new-task fit. That is a plausible protocol mechanism, not a demonstrated causal explanation. The mixed Reach results and variable historical retention are consistent with such a tradeoff but do not identify it.

## Efficiency

Generator training seconds, replay sampling seconds, and total recorded stage time:

| Seed | Rep | DDPM train stages 0/1/2/3 (s) | Sampling stages 0/1/2/3 (s) | Total (s) |
|---:|---|---|---|---:|
| 0 | 22D | 11.992/25.775/25.522/27.457 | 0/5.308/8.426/11.453 | 273.516 |
| 0 | 40D | 12.370/27.594/29.557/27.485 | 0/5.410/9.232/11.993 | 281.516 |
| 0 | 43D | 12.567/26.855/27.698/29.511 | 0/5.318/9.271/12.310 | 285.474 |
| 1 | 22D | 11.923/24.239/24.285/24.503 | 0/5.151/8.127/11.249 | 260.963 |
| 1 | 40D | 12.156/25.174/25.064/25.594 | 0/5.237/8.238/11.235 | 264.222 |
| 1 | 43D | 12.015/24.671/24.632/24.648 | 0/5.116/8.048/10.956 | 256.444 |
| 2 | 22D | 11.199/23.417/23.522/23.478 | 0/4.984/7.931/10.844 | 249.757 |
| 2 | 40D | 11.856/24.337/24.503/24.456 | 0/5.109/7.946/10.920 | 254.029 |
| 2 | 43D | 12.169/24.977/24.785/26.320 | 0/5.139/8.310/11.135 | 262.700 |

BC timing is present per stage in the timing summaries; there is no practically important representation-level runtime separation relative to seed/run variability. No unrecorded timing is estimated.

## Interpretation

### Observed facts

1. 43D has the highest mean final AP (.8125) and lowest mean F (.1292). Its paired AP advantage over 22D is positive in all three seeds (+5 to +10 pp).
2. Representation rankings are not generally consistent. 43D ties 40D in seed 0, leads in seed 1, and trails 40D in seed 2. Task-level paired differences have large SDs.
3. Handle is the largest mean acquisition-to-final loss for 40D and 43D, but not 22D: the seed-2 22D Hammer collapse makes Hammer's mean loss larger. Handle drops mainly after learning Reach; seed-1 43D instead recovers from .95 to 1.00.
4. Hammer is not uniformly stable under 6–7 stage-3 replay trajectories: final values range 0–1.00. The low mean for 22D is driven by one complete collapse.
5. Button does not always remain perfect after Reach (22D seed 1=.80; 40D seed 1=.85; 43D seed 2=.90), though mean retention is high for all methods.
6. Seed-0 Reach ordering does not replicate: seed 1 is .55/.55/.75 and seed 2 .70/.70/.60 for 22D/40D/43D.
7. Full-43D later-stage goal drift and near-universal 1e−3 violations replicate across all seeds. Previous-state inconsistency generally grows for 40D/43D, but not monotonically in every run.
8. Neither exact structural reconstruction nor lower action OOB orders downstream retention in these nine runs.

### Descriptive interpretation and hypotheses

43D merits further investigation because its AP advantage over 22D is the only aggregate paired effect with the same direction in all three seeds, and it has the best mean forgetting despite worse structural diagnostics. This does **not** make it universally superior. Its advantage combines stronger mean Hammer retention, modestly stronger Button retention, and better mean Reach acquisition, while Handle is worse than 22D.

The fixed-total replay protocol reduces per-task support as history grows (20 → 10 → 6–7 trajectories). The larger and more variable late-stage losses are consistent with a capacity/data-allocation limitation, but sequence interference, BC selection, and generated data quality are confounded. A fixed-per-task replay control is needed to attribute the effect to replay dilution.

## Limitations

- Only three training seeds and one task order are available; no inferential significance is claimed.
- Evaluation success increments are 5 pp because there are 20 episodes.
- The benchmark jointly tests representation, generator evolution, BC optimization, task interference, and a shrinking per-task replay allocation.
- Diagnostics measure structural contracts, not environment-dynamics validity or policy usefulness.
- 22D diagnostics were not originally persisted and reconstructed values must be distinguished from recorded diagnostics.
- Current-task-only validation prevents direct selection for historical retention.
- Component-wise previous-state errors are unavailable in existing reports.

## Recommended follow-up experiments

1. Add at least two more paired seeds, preserving all data/evaluation hashes, with 43D versus 22D as the primary comparison.
2. Repeat with fixed-per-task replay to separate representation effects from dilution under a fixed total of 20.
3. Repeat one reversed/permuted task order to test whether Handle/Reach effects are order-specific.
4. Add a read-only dynamics/plausibility evaluation and persist component-wise previous-state diagnostics; do not post-process generated samples.
5. Compare current-task-only checkpoint selection with a prespecified mixed current/history validation protocol.
6. Treat goal tolerance as a sweep (1e−3 plus physically motivated thresholds) and relate drift to task-goal distributions.

The most informative immediate next experiment is a paired 43D-versus-22D fixed-per-task replay ablation with additional seeds. It tests whether 43D's consistent AP advantage survives when historical-task support no longer shrinks.
