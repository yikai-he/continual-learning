---
status: completed
last_verified: 2026-11-10
applies_to: continual-learning
---

# KUKA expert-action disagreement analysis

## Objective and metric

This read-only analysis compares each saved generated action with the deterministic frozen SAC expert action at the same saved generated observation. The primary scalar is the **sum over four action dimensions** of squared coordinate errors, averaged over all trajectory timesteps. It is an expert-policy agreement diagnostic, not environment-dynamics validity or a true policy-quality score.

Raw actions are also clipped in memory to `[-1,1]`; saved arrays and actual BC inputs are unchanged. Observation variants independently replace goals with saved reference goals and/or previous state with shifted generated current state.

## Provenance and completion

| Seed | Representation | Status | Detail |
|---:|---|---|---|
| 0 | 22D | complete | validated |
| 0 | 43D | complete | validated |
| 1 | 22D | complete | validated |
| 1 | 43D | complete | validated |
| 2 | 22D | complete | validated |
| 2 | 43D | complete | validated |
| 3 | 22D | complete | validated |
| 3 | 43D | complete | validated |
| 4 | 22D | complete | validated |
| 4 | 43D | complete | validated |

All expert checkpoints were manifest-validated for task, backend, reward version, horizon, observation/action shapes, and SHA-256 identity against each run manifest. Task identity comes from the stage report's replay-group key.

## Real deterministic-expert reference

| Seed | Task | Mean squared L2 | P95 | Maximum |
|---:|---|---:|---:|---:|
| 0 | kuka-hammer-v3 | 0.000000 | 0.000000 | 0.000000 |
| 0 | kuka-handle-press-side-v3 | 0.000000 | 0.000000 | 0.000000 |
| 0 | kuka-button-press-v3 | 0.000000 | 0.000000 | 0.000000 |
| 1 | kuka-hammer-v3 | 0.000000 | 0.000000 | 0.000000 |
| 1 | kuka-handle-press-side-v3 | 0.000000 | 0.000000 | 0.000000 |
| 1 | kuka-button-press-v3 | 0.000000 | 0.000000 | 0.000000 |
| 2 | kuka-hammer-v3 | 0.000000 | 0.000000 | 0.000000 |
| 2 | kuka-handle-press-side-v3 | 0.000000 | 0.000000 | 0.000000 |
| 2 | kuka-button-press-v3 | 0.000000 | 0.000000 | 0.000000 |
| 3 | kuka-hammer-v3 | 0.000000 | 0.000000 | 0.000000 |
| 3 | kuka-handle-press-side-v3 | 0.000000 | 0.000000 | 0.000000 |
| 3 | kuka-button-press-v3 | 0.000000 | 0.000000 | 0.000000 |
| 4 | kuka-hammer-v3 | 0.000000 | 0.000000 | 0.000000 |
| 4 | kuka-handle-press-side-v3 | 0.000000 | 0.000000 | 0.000000 |
| 4 | kuka-button-press-v3 | 0.000000 | 0.000000 | 0.000000 |

## Generated replay: task-wise results

`Raw MSE` and `corrected MSE` use raw generated actions; `clipped MSE` clips only generated actions. Corrected means both goal and previous-state correction.

| Seed | Stage | Task | Rep | Raw MSE | Goal-only | Previous-only | Both corrected | Clipped MSE | Action OOB component | Final success | Δ acquisition→final |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | kuka-hammer-v3 | 22D | 1.079050 | 1.079050 | 1.079050 | 1.079050 | 0.981797 | 0.151938 | 1.00 | +0.00 |
| 0 | 2 | kuka-hammer-v3 | 22D | 1.006898 | 1.006898 | 1.006898 | 1.006898 | 0.912075 | 0.155125 | 1.00 | +0.00 |
| 0 | 2 | kuka-handle-press-side-v3 | 22D | 2.456658 | 2.456658 | 2.456658 | 2.456658 | 2.270696 | 0.158000 | 0.15 | -0.85 |
| 0 | 3 | kuka-hammer-v3 | 22D | 0.773790 | 0.773790 | 0.773790 | 0.773790 | 0.703911 | 0.154286 | 1.00 | +0.00 |
| 0 | 3 | kuka-handle-press-side-v3 | 22D | 2.354578 | 2.354578 | 2.354578 | 2.354578 | 2.231939 | 0.129107 | 0.15 | -0.85 |
| 0 | 3 | kuka-button-press-v3 | 22D | 1.972524 | 1.972524 | 1.972524 | 1.972524 | 1.941314 | 0.045208 | 1.00 | +0.00 |
| 0 | 1 | kuka-hammer-v3 | 43D | 1.780741 | 1.780719 | 1.747539 | 1.747551 | 1.508934 | 0.210687 | 0.90 | -0.10 |
| 0 | 2 | kuka-hammer-v3 | 43D | 2.422233 | 2.425261 | 2.404137 | 2.405976 | 1.868880 | 0.278875 | 0.90 | -0.10 |
| 0 | 2 | kuka-handle-press-side-v3 | 43D | 3.902006 | 3.914050 | 3.861347 | 3.864251 | 3.504881 | 0.164375 | 0.05 | -0.95 |
| 0 | 3 | kuka-hammer-v3 | 43D | 2.163768 | 2.164605 | 2.103233 | 2.104573 | 1.703279 | 0.269643 | 0.90 | -0.10 |
| 0 | 3 | kuka-handle-press-side-v3 | 43D | 3.580623 | 3.530247 | 3.479885 | 3.434363 | 3.332735 | 0.127143 | 0.05 | -0.95 |
| 0 | 3 | kuka-button-press-v3 | 43D | 3.221817 | 3.222592 | 3.184772 | 3.193591 | 3.030001 | 0.121250 | 1.00 | +0.10 |
| 1 | 1 | kuka-hammer-v3 | 22D | 1.366530 | 1.366530 | 1.366530 | 1.366530 | 1.297208 | 0.085938 | 0.85 | -0.15 |
| 1 | 2 | kuka-hammer-v3 | 22D | 1.413317 | 1.413317 | 1.413317 | 1.413317 | 1.358065 | 0.087250 | 0.85 | -0.15 |
| 1 | 2 | kuka-handle-press-side-v3 | 22D | 2.413848 | 2.413848 | 2.413848 | 2.413848 | 2.301759 | 0.111750 | 0.00 | -1.00 |
| 1 | 3 | kuka-hammer-v3 | 22D | 1.208696 | 1.208696 | 1.208696 | 1.208696 | 1.174558 | 0.072857 | 0.85 | -0.15 |
| 1 | 3 | kuka-handle-press-side-v3 | 22D | 2.239489 | 2.239489 | 2.239489 | 2.239489 | 2.152949 | 0.099821 | 0.00 | -1.00 |
| 1 | 3 | kuka-button-press-v3 | 22D | 1.033307 | 1.033307 | 1.033307 | 1.033307 | 1.013334 | 0.057292 | 0.95 | -0.05 |
| 1 | 1 | kuka-hammer-v3 | 43D | 1.715590 | 1.715607 | 1.651690 | 1.651737 | 1.538495 | 0.159188 | 0.70 | -0.30 |
| 1 | 2 | kuka-hammer-v3 | 43D | 2.125142 | 2.125118 | 2.156337 | 2.160816 | 1.885512 | 0.158375 | 0.70 | -0.30 |
| 1 | 2 | kuka-handle-press-side-v3 | 43D | 3.796306 | 3.810291 | 3.822129 | 3.841549 | 3.585412 | 0.108125 | 0.10 | -0.90 |
| 1 | 3 | kuka-hammer-v3 | 43D | 1.798961 | 1.799981 | 1.837713 | 1.840995 | 1.672330 | 0.122500 | 0.70 | -0.30 |
| 1 | 3 | kuka-handle-press-side-v3 | 43D | 3.115364 | 3.157128 | 3.291384 | 3.315299 | 3.006632 | 0.090893 | 0.10 | -0.90 |
| 1 | 3 | kuka-button-press-v3 | 43D | 2.550109 | 2.554099 | 2.720485 | 2.750401 | 2.447084 | 0.112917 | 0.35 | -0.65 |
| 2 | 1 | kuka-hammer-v3 | 22D | 1.327622 | 1.327622 | 1.327622 | 1.327622 | 1.234479 | 0.114812 | 0.95 | -0.05 |
| 2 | 2 | kuka-hammer-v3 | 22D | 1.215133 | 1.215133 | 1.215133 | 1.215133 | 1.164648 | 0.104625 | 0.95 | -0.05 |
| 2 | 2 | kuka-handle-press-side-v3 | 22D | 2.232909 | 2.232909 | 2.232909 | 2.232909 | 2.132035 | 0.108000 | 0.00 | -1.00 |
| 2 | 3 | kuka-hammer-v3 | 22D | 1.068661 | 1.068661 | 1.068661 | 1.068661 | 1.026962 | 0.100179 | 0.95 | -0.05 |
| 2 | 3 | kuka-handle-press-side-v3 | 22D | 2.173572 | 2.173572 | 2.173572 | 2.173572 | 2.115515 | 0.076964 | 0.00 | -1.00 |
| 2 | 3 | kuka-button-press-v3 | 22D | 1.560495 | 1.560495 | 1.560495 | 1.560495 | 1.540664 | 0.050000 | 0.95 | -0.05 |
| 2 | 1 | kuka-hammer-v3 | 43D | 1.733144 | 1.733130 | 1.670288 | 1.670321 | 1.557080 | 0.144937 | 0.45 | -0.55 |
| 2 | 2 | kuka-hammer-v3 | 43D | 2.094892 | 2.096048 | 2.106094 | 2.107419 | 1.835370 | 0.165000 | 0.45 | -0.55 |
| 2 | 2 | kuka-handle-press-side-v3 | 43D | 3.377271 | 3.363896 | 3.607991 | 3.593082 | 3.029867 | 0.177875 | 0.00 | -1.00 |
| 2 | 3 | kuka-hammer-v3 | 43D | 1.960319 | 1.961359 | 2.042428 | 2.044331 | 1.778765 | 0.138571 | 0.45 | -0.55 |
| 2 | 3 | kuka-handle-press-side-v3 | 43D | 3.278947 | 3.271749 | 3.589347 | 3.560680 | 3.023079 | 0.157500 | 0.00 | -1.00 |
| 2 | 3 | kuka-button-press-v3 | 43D | 2.857678 | 2.871761 | 2.969556 | 2.978995 | 2.725341 | 0.104167 | 0.70 | -0.30 |
| 3 | 1 | kuka-hammer-v3 | 22D | 1.400347 | 1.400347 | 1.400347 | 1.400347 | 1.267534 | 0.159312 | 1.00 | +0.05 |
| 3 | 2 | kuka-hammer-v3 | 22D | 1.379437 | 1.379437 | 1.379437 | 1.379437 | 1.241142 | 0.158000 | 1.00 | +0.05 |
| 3 | 2 | kuka-handle-press-side-v3 | 22D | 2.794500 | 2.794500 | 2.794500 | 2.794500 | 2.657201 | 0.116500 | 0.05 | -0.90 |
| 3 | 3 | kuka-hammer-v3 | 22D | 1.206028 | 1.206028 | 1.206028 | 1.206028 | 1.122065 | 0.137143 | 1.00 | +0.05 |
| 3 | 3 | kuka-handle-press-side-v3 | 22D | 2.680714 | 2.680714 | 2.680714 | 2.680714 | 2.588853 | 0.087321 | 0.05 | -0.90 |
| 3 | 3 | kuka-button-press-v3 | 22D | 1.932642 | 1.932642 | 1.932642 | 1.932642 | 1.894560 | 0.074167 | 0.90 | -0.10 |
| 3 | 1 | kuka-hammer-v3 | 43D | 1.859120 | 1.859106 | 1.854547 | 1.854528 | 1.634022 | 0.176187 | 0.95 | +0.00 |
| 3 | 2 | kuka-hammer-v3 | 43D | 2.150875 | 2.149946 | 2.183564 | 2.179818 | 1.799800 | 0.200750 | 0.95 | +0.00 |
| 3 | 2 | kuka-handle-press-side-v3 | 43D | 3.640856 | 3.621188 | 3.746155 | 3.710152 | 3.370341 | 0.151500 | 0.00 | -1.00 |
| 3 | 3 | kuka-hammer-v3 | 43D | 1.941808 | 1.940248 | 1.834473 | 1.835652 | 1.705976 | 0.171429 | 0.95 | +0.00 |
| 3 | 3 | kuka-handle-press-side-v3 | 43D | 3.387496 | 3.406363 | 3.462438 | 3.453051 | 3.232243 | 0.104464 | 0.00 | -1.00 |
| 3 | 3 | kuka-button-press-v3 | 43D | 3.209820 | 3.233204 | 3.265153 | 3.280565 | 3.019970 | 0.123958 | 1.00 | +0.00 |
| 4 | 1 | kuka-hammer-v3 | 22D | 1.447207 | 1.447207 | 1.447207 | 1.447207 | 1.291507 | 0.166187 | 1.00 | +0.00 |
| 4 | 2 | kuka-hammer-v3 | 22D | 1.332794 | 1.332794 | 1.332794 | 1.332794 | 1.205924 | 0.152375 | 1.00 | +0.00 |
| 4 | 2 | kuka-handle-press-side-v3 | 22D | 2.536128 | 2.536128 | 2.536128 | 2.536128 | 2.442260 | 0.103125 | 0.00 | -1.00 |
| 4 | 3 | kuka-hammer-v3 | 22D | 1.176736 | 1.176736 | 1.176736 | 1.176736 | 1.100179 | 0.132679 | 1.00 | +0.00 |
| 4 | 3 | kuka-handle-press-side-v3 | 22D | 2.325231 | 2.325231 | 2.325231 | 2.325231 | 2.248088 | 0.084821 | 0.00 | -1.00 |
| 4 | 3 | kuka-button-press-v3 | 22D | 1.397774 | 1.397774 | 1.397774 | 1.397774 | 1.378422 | 0.049583 | 0.25 | -0.75 |
| 4 | 1 | kuka-hammer-v3 | 43D | 1.660063 | 1.660075 | 1.677100 | 1.677169 | 1.360728 | 0.222812 | 0.90 | -0.10 |
| 4 | 2 | kuka-hammer-v3 | 43D | 2.015783 | 2.016645 | 2.124934 | 2.131128 | 1.520671 | 0.264750 | 0.90 | -0.10 |
| 4 | 2 | kuka-handle-press-side-v3 | 43D | 4.131835 | 4.108708 | 4.116104 | 4.085165 | 3.812948 | 0.138375 | 0.10 | -0.90 |
| 4 | 3 | kuka-hammer-v3 | 43D | 1.701971 | 1.697531 | 1.812531 | 1.813716 | 1.328624 | 0.261964 | 0.90 | -0.10 |
| 4 | 3 | kuka-handle-press-side-v3 | 43D | 3.735038 | 3.741237 | 3.883973 | 3.862716 | 3.540779 | 0.116786 | 0.10 | -0.90 |
| 4 | 3 | kuka-button-press-v3 | 43D | 3.158839 | 3.122461 | 3.016313 | 2.980757 | 2.943275 | 0.118750 | 0.70 | -0.30 |

## Distribution details

The machine-readable companion contains median, population SD, P95, maximum, per-action-dimension MSE, generated/expert action means and variances, timestep OOB, near-boundary fraction, and maximum bound violation for every row and diagnostic variant.

## Interpretation

- 22D: unweighted group mean raw=1.683554, both-corrected=1.683554, raw-observation/clipped-action=1.599721.
- 43D: unweighted group mean raw=2.668947, both-corrected=2.704345, raw-observation/clipped-action=2.410102.

These unweighted summaries do not replace task/stage matched comparisons. Structural correction changes only the expert's diagnostic input; no corrected trajectory was used for BC, so retention effects cannot be inferred.

Clipping lowers aggregate disagreement for the available representations; consult the values above for their ordering. Structural correction leaves 22D exactly unchanged, as expected, but its effect on generated representations is data dependent.

Task- and stage-specific rows contain both supporting examples and counterexamples for any relationship with retention. These comparisons keep task identity fixed but later replay stages use fewer trajectories per historical task.

With 5 seeds and correlated stage/task groups, associations with retention are descriptive. Counterexamples and task-specific trends should be prioritized over aggregate correlations.

## Limitations

- Expert agreement does not establish transition feasibility, reward quality, or closed-loop behavior.
- Generated observations may be off the expert's training distribution; an expert action is still produced but need not be meaningful there.
- Real-reference discrepancies include float32 serialization and SB3 inference reproducibility.
- Replay counts differ by task and stage; group means are not independent observations.
- No causal effect of correction, clipping, or representation was tested.

## Recommended experiments

1. Run a fixed-per-task replay control to separate representation effects from fixed-total replay dilution.
2. Repeat a different task order to test whether task-specific patterns generalize.
3. If disagreement remains predictive, prespecify a controlled replay filtering or weighting ablation; do not retrofit generated actions in the present artifacts.

Expert SHA-256 identities: `{"kuka-button-press-v3": "ee92f679f342a3609fdc56c4379641d7e60123fc3443e558203685577674f2f0", "kuka-hammer-v3": "1900fe4dd8d3150f7d484f7f8886f4e65616bf11fa681d5a29ae6b7a06696c8c", "kuka-handle-press-side-v3": "090824e59d5dcfe3a3be0700a020877b97e5545b3bf9e507a02d1b132dc47049", "kuka-reach-v3": "2be77e7884b1fdf0b29d364ec0a5dcfdae51b0abb34b56f7ecaded5e6be32792"}`.
