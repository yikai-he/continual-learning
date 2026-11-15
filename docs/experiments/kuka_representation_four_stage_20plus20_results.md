---
status: current
last_verified: 2026-11-09
applies_to: continual-learning
---

# Four-stage KUKA trajectory-representation ablation: 20+20 results

## 1. Experimental setup

This report analyzes the completed runs under
`runs/representation_ablation/four_stage_20plus20/` for the sequence:

1. `kuka-hammer-v3`
2. `kuka-handle-press-side-v3`
3. `kuka-button-press-v3`
4. `kuka-reach-v3`

The compared DDPM trajectory representations are `structured_22d`,
`goal_conditioned_40d`, and `full_43d`. Each configuration collects 20 current
expert trajectories per stage and uses a fixed total of 20 generated historical
trajectories after Stage 0. Other shared settings are 100 DDPM epochs, 100
diffusion steps, width 64, batch size 8, learning rate 0.001, raw actions, no
action projection, 100 BC epochs, task-agnostic BC, historical replay
training-only, and 20 fixed-task evaluation episodes using reset/task-bank seed
10000.

Average Performance (AP) and forgetting use `src/continual/metrics.py` exactly.
Forgetting at stage `k` is the mean of `R[j,j] - R[k,j]` over `j=0..k`, including
the current task's zero term. No training, replay generation, or additional
evaluation was performed for this analysis.

## 2. Completion and reproducibility audit

Only seed 0 is available. Seeds 1 and 2 are missing for all representations.

| Seed | 22D | 40D | 43D |
|---:|---|---|---|
| 0 | complete and valid | complete and valid | complete and valid |
| 1 | missing | missing | missing |
| 2 | missing | missing | missing |

For each seed-0 run, the repository validator confirmed:

- `RUN_COMPLETE` and all four contiguous stage markers exist.
- Every marker-referenced generator, GeneralPolicy, stage state, report, and
  evaluation artifact matches its recorded SHA-256.
- The final `4×4` evaluation matrix has exactly the expected lower-triangular
  measured cells.
- The resolved configuration matches the canonical configuration and intended
  representation.
- Task order, backend (`kuka-v3`), reward version (`v2`), expert identities,
  DDPM/BC hyperparameters, replay allocation, action settings, and evaluation
  task banks agree across representations.

Expert checkpoint hashes are identical across all three runs:

| Task | Expert SHA-256 |
|---|---|
| Hammer | `1900fe4dd8d3150f7d484f7f8886f4e65616bf11fa681d5a29ae6b7a06696c8c` |
| Handle Press Side | `090824e59d5dcfe3a3be0700a020877b97e5545b3bf9e507a02d1b132dc47049` |
| Button Press | `ee92f679f342a3609fdc56c4379641d7e60123fc3443e558203685577674f2f0` |
| Reach | `2be77e7884b1fdf0b29d364ec0a5dcfdae51b0abb34b56f7ecaded5e6be32792` |

Current-task arrays are byte-identical across representations within seed 0:

| Task | Current-task array SHA-256 |
|---|---|
| Hammer | `b4d106996874329a2b3ae6d2571fa4d857e20b6f4c107cd5ccb3f95728a24643` |
| Handle | `050b4b6985bef8ea674e43e282fab0061c98eee7f87da1b97591757552583e4a` |
| Button | `1685b32de675951126b83903cef6c099416e4b9b652f77bbd18a9f4bb5ebc4bd` |
| Reach | `d20e6c8801c73e1a7b6862f6398f8f23e6645accc8a73e30fb6672431fa8c230` |

Saved split IDs also match across representations at every stage. No
comparability discrepancy was found among the three available runs.

## 3. Full evaluation matrices

Rows are training stages; columns are Hammer, Handle, Button, and Reach.

### Structured 22D

```text
[[1.00,    —,    —,    —],
 [1.00, 1.00,    —,    —],
 [0.80, 0.95, 1.00,    —],
 [0.95, 0.50, 1.00, 0.65]]
```

### Goal-conditioned 40D

```text
[[1.00,    —,    —,    —],
 [0.85, 1.00,    —,    —],
 [0.80, 0.95, 1.00,    —],
 [1.00, 0.40, 1.00, 1.00]]
```

### Full 43D

```text
[[1.00,    —,    —,    —],
 [1.00, 1.00,    —,    —],
 [0.85, 0.85, 1.00,    —],
 [1.00, 0.45, 1.00, 0.95]]
```

## 4. Evaluation returns and episode lengths

Each cell is `success; mean return ± episode-return SD`. Every measured cell has
mean episode length 200.0, so episode length does not distinguish methods.

### Structured 22D

| Stage | Hammer | Handle | Button | Reach |
|---:|---|---|---|---|
| 0 | 1.00; 397.8 ± 78.1 | — | — | — |
| 1 | 1.00; 231.3 ± 116.0 | 1.00; 871.4 ± 154.6 | — | — |
| 2 | 0.80; 261.4 ± 206.5 | 0.95; 926.5 ± 389.4 | 1.00; 1478.0 ± 39.6 | — |
| 3 | 0.95; 203.6 ± 183.4 | 0.50; 551.0 ± 303.2 | 1.00; 1467.5 ± 21.2 | 0.65; 1714.3 ± 196.7 |

### Goal-conditioned 40D

| Stage | Hammer | Handle | Button | Reach |
|---:|---|---|---|---|
| 0 | 1.00; 397.8 ± 78.1 | — | — | — |
| 1 | 0.85; 300.8 ± 113.8 | 1.00; 947.5 ± 83.4 | — | — |
| 2 | 0.80; 352.6 ± 72.7 | 0.95; 855.2 ± 206.9 | 1.00; 1472.0 ± 58.4 | — |
| 3 | 1.00; 289.6 ± 98.7 | 0.40; 581.7 ± 361.5 | 1.00; 1462.9 ± 11.5 | 1.00; 1732.1 ± 250.3 |

### Full 43D

| Stage | Hammer | Handle | Button | Reach |
|---:|---|---|---|---|
| 0 | 1.00; 397.8 ± 78.1 | — | — | — |
| 1 | 1.00; 269.4 ± 133.2 | 1.00; 919.7 ± 109.8 | — | — |
| 2 | 0.85; 278.3 ± 94.9 | 0.85; 828.1 ± 216.1 | 1.00; 1482.8 ± 22.7 | — |
| 3 | 1.00; 324.0 ± 93.8 | 0.45; 640.1 ± 401.3 | 1.00; 1485.3 ± 38.0 | 0.95; 1761.1 ± 226.7 |

## 5. Stage-wise AP and forgetting

| Representation | AP0 | AP1 | AP2 | AP3 | F0 | F1 | F2 | F3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 22D | 1.000 | 1.000 | 0.917 | 0.775 | 0.000 | 0.000 | 0.083 | 0.138 |
| 40D | 1.000 | 0.925 | 0.917 | 0.850 | 0.000 | 0.075 | 0.083 | 0.150 |
| 43D | 1.000 | 1.000 | 0.900 | 0.850 | 0.000 | 0.000 | 0.100 | 0.138 |

Immediate acquisition is perfect through Button for all arms. Reach acquisition
at Stage 3 differs: 0.65 for 22D, 1.00 for 40D, and 0.95 for 43D.

Task trajectories are non-monotonic:

- Hammer falls at Stage 2 to 0.80/0.80/0.85, then recovers at Stage 3 to
  0.95/1.00/1.00 for 22D/40D/43D. This is backward improvement after learning
  Reach, not monotonic forgetting.
- Handle falls from 1.00 immediately after learning to 0.95/0.95/0.85 at Stage
  2 and then to 0.50/0.40/0.45 at Stage 3. It is the most severely forgotten
  historical task.
- Button remains at 1.00 after learning and at the final stage in every arm.
- Reach is evaluated only at Stage 3, so its result measures acquisition, not
  retention.

## 6. Final retention and available-seed aggregation

| Representation | Hammer | Handle | Button | Reach | Final AP | Final forgetting | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| 22D | 0.950 | 0.500 | 1.000 | 0.650 | 0.775 | 0.1375 | 1 |
| 40D | 1.000 | 0.400 | 1.000 | 1.000 | 0.850 | 0.1500 | 1 |
| 43D | 1.000 | 0.450 | 1.000 | 0.950 | 0.850 | 0.1375 | 1 |

Only one seed is complete. These values are the available means, but sample SD
is unavailable and is deliberately not calculated. Multi-seed conclusions are
not possible yet.

## 7. Paired representation differences

Differences are percentage-point equivalents of success-rate differences. All
comparisons contain only seed 0 (`matched n=1`); sample SD is unavailable.

| Comparison | Δ Final AP | Δ Final F | Δ Hammer | Δ Handle | Δ Button | Δ Reach |
|---|---:|---:|---:|---:|---:|---:|
| 40D − 22D | +0.075 | +0.0125 | +0.050 | −0.100 | 0.000 | +0.350 |
| 43D − 22D | +0.075 | 0.0000 | +0.050 | −0.050 | 0.000 | +0.300 |
| 43D − 40D | 0.000 | −0.0125 | 0.000 | +0.050 | 0.000 | −0.050 |

Stage-wise AP differences further show that the final advantage is not
consistent throughout training:

| Comparison | Δ AP0 | Δ AP1 | Δ AP2 | Δ AP3 |
|---|---:|---:|---:|---:|
| 40D − 22D | 0.000 | −0.075 | 0.000 | +0.075 |
| 43D − 22D | 0.000 | 0.000 | −0.017 | +0.075 |
| 43D − 40D | 0.000 | +0.075 | −0.017 | 0.000 |

The 40D/43D final AP advantage over 22D is driven mostly by Reach performance,
not consistently better retention: 22D has the best final Handle result and
ties 43D for the lowest formal forgetting.

## 8. Verified replay and dataset allocation

Every stage report contains the intended fixed-total allocation:

| Stage | Collected current | Historical allocation | Physical total | Training trajectories | Validation trajectories |
|---:|---:|---|---:|---:|---:|
| 0 | 20 | none | 20 | 16 | 4 |
| 1 | 20 | Hammer 20 | 40 | 36 | 4 |
| 2 | 20 | Hammer 10, Handle 10 | 40 | 36 | 4 |
| 3 | 20 | Hammer 7, Handle 7, Button 6 | 40 | 36 | 4 |

For all historical groups, all generated trajectory IDs are in training and
their validation ID lists are empty. Each current group has 16 training and 4
validation IDs. Thus historical replay is training-only and the four current
validation episodes are not used for gradient updates. Both generator and BC
use the resulting 16 trajectories at Stage 0 and 36 at later stages.

## 9. Aggregate replay diagnostics by stage

Stage aggregates weight task means by generated trajectory count. Maxima are
the maximum across task-specific groups. Goal and state errors are Euclidean
norms; tolerance is `1e-3`.

| Representation | Stage | Goal mean drift | Goal max | Goal variation | Goal violations | Previous-state mean | Previous violations | Action component OOB | OOB timesteps | Max action violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 22D | 1 | 0 | 0 | 0 | 0% | 0 | 0% | 3.406% | 11.100% | 0.478 |
| 22D | 2 | 0 | 0 | 0 | 0% | 0 | 0% | 25.238% | 66.450% | 1.145 |
| 22D | 3 | 0 | 0 | 0 | 0% | 0 | 0% | 19.269% | 52.550% | 1.141 |
| 40D | 1 | 0 | 0 | 0 | 0% | 0.02034 | 100% | 10.081% | 29.250% | 0.613 |
| 40D | 2 | 0 | 0 | 0 | 0% | 0.03718 | 100% | 13.800% | 35.225% | 0.863 |
| 40D | 3 | 0 | 0 | 0 | 0% | 0.04113 | 100% | 14.250% | 35.750% | 0.981 |
| 43D | 1 | 8.67e-5 | 2.48e-4 | 1.43e-4 | 0% | 0.02546 | 100% | 4.819% | 17.825% | 0.148 |
| 43D | 2 | 0.00820 | 0.03910 | 0.01957 | 98.925% | 0.04718 | 100% | 16.325% | 59.600% | 0.748 |
| 43D | 3 | 0.00920 | 0.04245 | 0.01880 | 99.525% | 0.04515 | 100% | 14.563% | 50.875% | 0.844 |

All saved/generated arrays are finite, have physical shape `(N,200,43)`, and
contain zero NaNs and zero infinities. The 22D reports predate storage of the
nested structural-diagnostics object. Their rows above were recovered read-only
from saved stage arrays using the repository's diagnostic formulas; exact goal
and previous-state values follow from the saved decoded arrays and the 22D
reconstruction contract. No samples were regenerated.

Quality changes materially with stage:

- 22D keeps exact structure but its action OOB component rate jumps from 3.4%
  at Stage 1 to 25.2% at Stage 2, then remains high at 19.3%.
- 40D retains exact goals, while previous-state mean error grows from 0.0203 to
  0.0411 and action OOB grows from 10.1% to 14.3%.
- 43D goal drift changes from negligible at Stage 1 to approximately 0.008–0.009
  mean error at Stages 2–3, with about 99% of timesteps exceeding tolerance.
  Previous-state error and action OOB also increase.

These observations show replay quality deteriorating in several diagnostics as
the generator is updated across stages, but not uniformly across representations
or metrics.

## 10. Task-specific replay diagnostics

Each cell is `goal mean drift; previous-state mean error; action-component OOB`.

| Stage | Historical task | 22D | 40D | 43D |
|---:|---|---|---|---|
| 1 | Hammer (20) | 0; 0; 3.41% | 0; 0.0203; 10.08% | 8.67e-5; 0.0255; 4.82% |
| 2 | Hammer (10) | 0; 0; 31.76% | 0; 0.0311; 12.83% | 0.00721; 0.0600; 22.35% |
| 2 | Handle (10) | 0; 0; 18.71% | 0; 0.0433; 14.78% | 0.00919; 0.0344; 10.30% |
| 3 | Hammer (7) | 0; 0; 30.46% | 0; 0.0462; 15.89% | 0.00745; 0.0591; 22.09% |
| 3 | Handle (7) | 0; 0; 19.13% | 0; 0.0422; 16.37% | 0.01174; 0.0421; 10.55% |
| 3 | Button (6) | 0; 0; 6.38% | 0; 0.0340; 9.85% | 0.00827; 0.0323; 10.46% |

Per-task diagnostics complicate a direct quality–retention link:

- Hammer replay often has the worst action OOB (22D/43D) or previous-state
  error (40D/43D), yet Hammer recovers to 95–100% at the final stage.
- Handle is forgotten most severely, but its action OOB is lower than Hammer's
  for 22D and 43D. For 40D its Handle OOB is slightly higher, but the difference
  is too small and unreplicated to establish an explanation.
- Button replay has relatively low errors and Button remains at 100%, a
  compatible association but not causal evidence.

## 11. BC training analysis

Losses are from the selected validation checkpoint. Historical trajectories are
training-only, so validation loss measures the current task and is not a direct
measure of historical retention.

| Representation | Stage | Selected epoch | Training loss | Validation loss | BC time | Training / validation trajectories |
|---|---:|---:|---:|---:|---:|---:|
| 22D | 0 | 95 | 0.00277 | 0.11306 | 7.906 s | 16 / 4 |
| 22D | 1 | 72 | 0.00832 | 0.02965 | 19.062 s | 36 / 4 |
| 22D | 2 | 92 | 0.01440 | 0.00941 | 17.838 s | 36 / 4 |
| 22D | 3 | 30 | 0.04479 | 0.15719 | 18.519 s | 36 / 4 |
| 40D | 0 | 95 | 0.00277 | 0.11306 | 8.074 s | 16 / 4 |
| 40D | 1 | 98 | 0.00711 | 0.02386 | 18.739 s | 36 / 4 |
| 40D | 2 | 98 | 0.00966 | 0.01054 | 17.870 s | 36 / 4 |
| 40D | 3 | 67 | 0.02456 | 0.14365 | 17.470 s | 36 / 4 |
| 43D | 0 | 95 | 0.00277 | 0.11306 | 7.932 s | 16 / 4 |
| 43D | 1 | 100 | 0.00726 | 0.02691 | 18.652 s | 36 / 4 |
| 43D | 2 | 100 | 0.00690 | 0.00781 | 19.726 s | 36 / 4 |
| 43D | 3 | 40 | 0.01885 | 0.15058 | 18.999 s | 36 / 4 |

Stage-0 losses, checkpoint epoch, data, and performance are exactly identical
across representations, ruling out observed Stage-0 policy divergence. At Stage
3, all validation losses are high relative to Stages 1–2; 22D has the highest
loss and earliest checkpoint and also the weakest Reach acquisition. This is a
descriptive alignment, not proof that representation caused the Reach gap.

## 12. Efficiency

| Representation | Generator training by stage (s) | Replay sampling by stage (s) | Total recorded stage time (s) |
|---|---|---|---:|
| 22D | 11.992, 25.775, 25.522, 27.457 | 0.000, 5.308, 8.426, 11.453 | 273.516 |
| 40D | 12.370, 27.594, 29.557, 27.485 | 0.000, 5.410, 9.232, 11.993 | 281.516 |
| 43D | 12.567, 26.855, 27.698, 29.511 | 0.000, 5.318, 9.271, 12.310 | 285.474 |

Timing differences are modest and come from one CPU run, so they should not be
treated as stable performance estimates.

## 13. Comparison with the earlier two-task experiment

In the previous two-task 100+100 experiment, 40D and 43D retained Hammer at
100% in both completed seeds; 22D retained 85% at seed 0 and 100% at seed 1.
The short benchmark was largely ceiling-saturated.

The four-stage seed-0 run reveals substantially more temporal structure:
temporary Hammer forgetting followed by recovery, severe final Handle
forgetting, stable Button retention, representation-dependent Reach acquisition,
and diagnostic degradation over generator updates. However, the new experiment
simultaneously changes the sequence from two to four tasks and the budget from
100+100 to 20+20. The differences cannot be attributed to task count alone,
replay budget alone, or representation without the missing seeds and further
controlled ablations.

## 14. Scientific interpretation

### Direct observations

1. **Highest final AP:** 40D and 43D tie at 0.850; 22D reaches 0.775.
2. **Lowest forgetting:** 22D and 43D tie at 0.1375; 40D is 0.1500. Thus the
   highest AP is not synonymous with the lowest forgetting.
3. **Seed consistency:** unavailable—only seed 0 is complete.
4. **Most forgotten historical task:** Handle, ending at 0.40–0.50 after being
   learned at 1.00. Hammer recovers and Button does not forget.
5. **Structural consistency and retention:** exact 22D structure does not yield
   uniformly better retention. It has the best Handle endpoint but weaker Reach
   acquisition and slightly weaker Hammer retention.
6. **Action OOB and retention:** OOB does not order final performance. Hammer
   has high task-specific OOB yet recovers; Handle has lower OOB in 22D/43D yet
   is forgotten most.
7. **Additional stages:** the longer run exposes temporary forgetting, recovery,
   task-specific vulnerability, and generator degradation hidden by the
   two-task endpoint. Whether these patterns replicate is unknown.
8. **Fixed replay budget:** Handle degradation is consistent with pressure from
   sharing a fixed budget, but Hammer receives the same Stage-3 allocation and
   recovers. Budget dilution alone does not explain task outcomes.
9. **Transfer:** Hammer shows backward improvement from Stage 2 to Stage 3.
   Reach acquisition differs substantially, suggesting representation-dependent
   current-task learning or checkpoint selection. There is no clean evidence
   that later tasks uniformly harm or help earlier tasks.
10. **Reach ceiling:** the expert evaluation previously showed an initial 100%
    Reach success rate, and 40D/43D attain 95–100% here. Reach is ceiling-prone,
    limiting fine discrimination. The 22D result of 65% nevertheless shows that
    acquisition is not automatically saturated under this continual protocol.

### Suggested patterns and unsupported mechanisms

The data suggest that representation choice affects the joint BC training
outcome after repeated generator updates, especially current Reach acquisition.
They do not show that generated previous-state inconsistency is beneficial, that
goal drift causes forgetting, or that action OOB causes policy failure. The
three representations differ simultaneously in generated feature space and
learned replay distribution; one seed cannot identify a mechanism.

## 15. Limitations and recommended next experiments

- Seeds 1 and 2 are missing, so sample SD and stable paired comparisons are
  unavailable.
- Evaluation has 20 episodes per cell and 0.05 success-rate resolution.
- Full 43D has severe goal drift after Stage 1, but only one seed tests whether
  that pattern persists.
- Structural and action diagnostics do not measure simulator transition
  feasibility or temporal dynamics consistency.
- Reach is intrinsically ceiling-prone in the saved expert evaluation.
- This experiment confounds increased sequence length with reduced data/replay
  budget relative to the earlier benchmark by design.

The immediate next step is to complete seeds 1 and 2 unchanged. This is more
valuable than modifying the architecture because the seed-0 ranking separates
AP, forgetting, historical retention, and current Reach acquisition in different
ways. After three seeds, consider a controlled budget ablation at the same
four-task sequence (for example 20+20 versus 100+100) and a larger evaluation
episode count to reduce ceiling quantization. A task-conditioned analysis of
generator dynamics validity would be a separate follow-up, not a substitute for
the current structural diagnostics.

## Source references

- `docs/ARCHITECTURE.md`
- `docs/experiments/kuka_trajectory_representation_seed01_analysis.md`
- `configs/experiments/kuka_trajectory_representations/four_stage_20plus20/`
- `scripts/evaluation/analyze_kuka_representation_four_stage.py`
- Each available run's `config.yaml`, `run_manifest.json`, four completion
  markers, stage reports, saved stage arrays, `evaluation_matrix.json`, and
  `timing_summary.json` under
  `runs/representation_ablation/four_stage_20plus20/`
