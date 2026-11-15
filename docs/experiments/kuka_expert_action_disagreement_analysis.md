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
| 0 | 40D | complete | validated |
| 0 | 43D | complete | validated |
| 1 | 22D | complete | validated |
| 1 | 40D | complete | validated |
| 1 | 43D | complete | validated |
| 2 | 22D | complete | validated |
| 2 | 40D | complete | validated |
| 2 | 43D | complete | validated |

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

## Generated replay: task-wise results

`Raw MSE` and `corrected MSE` use raw generated actions; `clipped MSE` clips only generated actions. Corrected means both goal and previous-state correction.

| Seed | Stage | Task | Rep | Raw MSE | Goal-only | Previous-only | Both corrected | Clipped MSE | Action OOB component | Final success | Δ acquisition→final |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | kuka-hammer-v3 | 22D | 0.587853 | 0.587853 | 0.587853 | 0.587853 | 0.585269 | 0.034063 | 0.95 | -0.05 |
| 0 | 2 | kuka-hammer-v3 | 22D | 0.841555 | 0.841555 | 0.841555 | 0.841555 | 0.649752 | 0.317625 | 0.95 | -0.05 |
| 0 | 2 | kuka-handle-press-side-v3 | 22D | 0.975507 | 0.975507 | 0.975507 | 0.975507 | 0.898133 | 0.187125 | 0.50 | -0.50 |
| 0 | 3 | kuka-hammer-v3 | 22D | 0.720911 | 0.720911 | 0.720911 | 0.720911 | 0.551291 | 0.304643 | 0.95 | -0.05 |
| 0 | 3 | kuka-handle-press-side-v3 | 22D | 1.292037 | 1.292037 | 1.292037 | 1.292037 | 1.163844 | 0.191250 | 0.50 | -0.50 |
| 0 | 3 | kuka-button-press-v3 | 22D | 1.148698 | 1.148698 | 1.148698 | 1.148698 | 1.129598 | 0.063750 | 1.00 | +0.00 |
| 0 | 1 | kuka-hammer-v3 | 40D | 0.193455 | 0.193455 | 0.207493 | 0.207493 | 0.185689 | 0.100812 | 1.00 | +0.00 |
| 0 | 2 | kuka-hammer-v3 | 40D | 0.299890 | 0.299890 | 0.308263 | 0.308263 | 0.250210 | 0.128250 | 1.00 | +0.00 |
| 0 | 2 | kuka-handle-press-side-v3 | 40D | 0.749195 | 0.749195 | 0.803119 | 0.803119 | 0.706375 | 0.147750 | 0.40 | -0.60 |
| 0 | 3 | kuka-hammer-v3 | 40D | 0.354236 | 0.354236 | 0.362582 | 0.362582 | 0.275811 | 0.158929 | 1.00 | +0.00 |
| 0 | 3 | kuka-handle-press-side-v3 | 40D | 0.886823 | 0.886823 | 0.945341 | 0.945341 | 0.813090 | 0.163750 | 0.40 | -0.60 |
| 0 | 3 | kuka-button-press-v3 | 40D | 1.235055 | 1.235055 | 1.212796 | 1.212796 | 1.184653 | 0.098542 | 1.00 | +0.00 |
| 0 | 1 | kuka-hammer-v3 | 43D | 0.329528 | 0.329502 | 0.341222 | 0.341196 | 0.325390 | 0.048188 | 1.00 | +0.00 |
| 0 | 2 | kuka-hammer-v3 | 43D | 0.316695 | 0.326748 | 0.346137 | 0.352881 | 0.298754 | 0.223500 | 1.00 | +0.00 |
| 0 | 2 | kuka-handle-press-side-v3 | 43D | 1.142918 | 1.156938 | 1.198414 | 1.223793 | 1.110343 | 0.103000 | 0.45 | -0.55 |
| 0 | 3 | kuka-hammer-v3 | 43D | 0.280379 | 0.287752 | 0.305317 | 0.310442 | 0.262046 | 0.220893 | 1.00 | +0.00 |
| 0 | 3 | kuka-handle-press-side-v3 | 43D | 0.856235 | 0.901716 | 0.895698 | 0.929052 | 0.829194 | 0.105536 | 0.45 | -0.55 |
| 0 | 3 | kuka-button-press-v3 | 43D | 1.303987 | 1.306496 | 1.208014 | 1.206423 | 1.273318 | 0.104583 | 1.00 | +0.00 |
| 1 | 1 | kuka-hammer-v3 | 22D | 0.354410 | 0.354410 | 0.354410 | 0.354410 | 0.328985 | 0.040000 | 1.00 | +0.00 |
| 1 | 2 | kuka-hammer-v3 | 22D | 0.461155 | 0.461155 | 0.461155 | 0.461155 | 0.424030 | 0.052750 | 1.00 | +0.00 |
| 1 | 2 | kuka-handle-press-side-v3 | 22D | 0.964177 | 0.964177 | 0.964177 | 0.964177 | 0.929396 | 0.106250 | 1.00 | +0.00 |
| 1 | 3 | kuka-hammer-v3 | 22D | 0.628729 | 0.628729 | 0.628729 | 0.628729 | 0.568423 | 0.067679 | 1.00 | +0.00 |
| 1 | 3 | kuka-handle-press-side-v3 | 22D | 0.922277 | 0.922277 | 0.922277 | 0.922277 | 0.879811 | 0.088929 | 1.00 | +0.00 |
| 1 | 3 | kuka-button-press-v3 | 22D | 0.460661 | 0.460661 | 0.460661 | 0.460661 | 0.440468 | 0.067917 | 0.80 | -0.20 |
| 1 | 1 | kuka-hammer-v3 | 40D | 0.420992 | 0.420992 | 0.440854 | 0.440854 | 0.392983 | 0.286813 | 0.75 | -0.25 |
| 1 | 2 | kuka-hammer-v3 | 40D | 0.680555 | 0.680555 | 0.709944 | 0.709944 | 0.643054 | 0.183125 | 0.75 | -0.25 |
| 1 | 2 | kuka-handle-press-side-v3 | 40D | 1.626462 | 1.626462 | 1.601140 | 1.601140 | 1.542795 | 0.128250 | 0.15 | -0.85 |
| 1 | 3 | kuka-hammer-v3 | 40D | 0.452305 | 0.452305 | 0.481053 | 0.481053 | 0.439118 | 0.083571 | 0.75 | -0.25 |
| 1 | 3 | kuka-handle-press-side-v3 | 40D | 1.290498 | 1.290498 | 1.341286 | 1.341286 | 1.241290 | 0.104464 | 0.15 | -0.85 |
| 1 | 3 | kuka-button-press-v3 | 40D | 0.555385 | 0.555385 | 0.558990 | 0.558990 | 0.504122 | 0.083958 | 0.85 | -0.15 |
| 1 | 1 | kuka-hammer-v3 | 43D | 0.394186 | 0.394199 | 0.478722 | 0.478735 | 0.380179 | 0.191562 | 0.80 | -0.20 |
| 1 | 2 | kuka-hammer-v3 | 43D | 0.611131 | 0.610975 | 0.617351 | 0.617525 | 0.608450 | 0.123500 | 0.80 | -0.20 |
| 1 | 2 | kuka-handle-press-side-v3 | 43D | 0.581894 | 0.631337 | 0.678428 | 0.736109 | 0.532584 | 0.149625 | 1.00 | +0.00 |
| 1 | 3 | kuka-hammer-v3 | 43D | 0.741119 | 0.742480 | 0.676217 | 0.677606 | 0.733739 | 0.164643 | 0.80 | -0.20 |
| 1 | 3 | kuka-handle-press-side-v3 | 43D | 0.792718 | 0.870125 | 0.941769 | 0.966669 | 0.716703 | 0.138750 | 1.00 | +0.00 |
| 1 | 3 | kuka-button-press-v3 | 43D | 0.396966 | 0.372952 | 0.388261 | 0.363795 | 0.376615 | 0.085417 | 1.00 | +0.00 |
| 2 | 1 | kuka-hammer-v3 | 22D | 0.269409 | 0.269409 | 0.269409 | 0.269409 | 0.264805 | 0.066062 | 0.00 | -1.00 |
| 2 | 2 | kuka-hammer-v3 | 22D | 0.384755 | 0.384755 | 0.384755 | 0.384755 | 0.382123 | 0.073000 | 0.00 | -1.00 |
| 2 | 2 | kuka-handle-press-side-v3 | 22D | 0.676130 | 0.676130 | 0.676130 | 0.676130 | 0.630455 | 0.118125 | 0.70 | -0.30 |
| 2 | 3 | kuka-hammer-v3 | 22D | 0.436176 | 0.436176 | 0.436176 | 0.436176 | 0.411163 | 0.189643 | 0.00 | -1.00 |
| 2 | 3 | kuka-handle-press-side-v3 | 22D | 0.873075 | 0.873075 | 0.873075 | 0.873075 | 0.769323 | 0.216607 | 0.70 | -0.30 |
| 2 | 3 | kuka-button-press-v3 | 22D | 1.373991 | 1.373991 | 1.373991 | 1.373991 | 1.244318 | 0.120208 | 1.00 | +0.00 |
| 2 | 1 | kuka-hammer-v3 | 40D | 0.345497 | 0.345497 | 0.378186 | 0.378186 | 0.345497 | 0.000000 | 0.90 | -0.10 |
| 2 | 2 | kuka-hammer-v3 | 40D | 0.290337 | 0.290337 | 0.326216 | 0.326216 | 0.290337 | 0.000000 | 0.90 | -0.10 |
| 2 | 2 | kuka-handle-press-side-v3 | 40D | 0.692097 | 0.692097 | 0.728757 | 0.728757 | 0.623035 | 0.148875 | 0.70 | -0.30 |
| 2 | 3 | kuka-hammer-v3 | 40D | 0.281694 | 0.281694 | 0.311766 | 0.311766 | 0.281694 | 0.000000 | 0.90 | -0.10 |
| 2 | 3 | kuka-handle-press-side-v3 | 40D | 1.182922 | 1.182922 | 1.186277 | 1.186277 | 0.925956 | 0.253929 | 0.70 | -0.30 |
| 2 | 3 | kuka-button-press-v3 | 40D | 0.940853 | 0.940853 | 0.995092 | 0.995092 | 0.876206 | 0.082292 | 1.00 | +0.00 |
| 2 | 1 | kuka-hammer-v3 | 43D | 0.253377 | 0.253424 | 0.353725 | 0.353773 | 0.252075 | 0.034875 | 0.95 | -0.05 |
| 2 | 2 | kuka-hammer-v3 | 43D | 0.211249 | 0.210452 | 0.314961 | 0.312740 | 0.207243 | 0.082250 | 0.95 | -0.05 |
| 2 | 2 | kuka-handle-press-side-v3 | 43D | 0.780698 | 0.694371 | 0.790700 | 0.730262 | 0.719449 | 0.113125 | 0.35 | -0.65 |
| 2 | 3 | kuka-hammer-v3 | 43D | 0.147592 | 0.146709 | 0.190819 | 0.188755 | 0.145503 | 0.066429 | 0.95 | -0.05 |
| 2 | 3 | kuka-handle-press-side-v3 | 43D | 1.194834 | 1.176041 | 1.263095 | 1.258635 | 0.986048 | 0.207857 | 0.35 | -0.65 |
| 2 | 3 | kuka-button-press-v3 | 43D | 0.838903 | 0.844324 | 0.829898 | 0.834701 | 0.790896 | 0.115833 | 0.90 | -0.10 |

## Distribution details

The machine-readable companion contains median, population SD, P95, maximum, per-action-dimension MSE, generated/expert action means and variances, timestep OOB, near-boundary fraction, and maximum bound violation for every row and diagnostic variant.

## Interpretation

- 22D: unweighted group mean raw=0.742861, both-corrected=0.742861, raw-observation/clipped-action=0.680621.
- 40D: unweighted group mean raw=0.693236, both-corrected=0.716620, raw-observation/clipped-action=0.640106.
- 43D: unweighted group mean raw=0.620801, both-corrected=0.660172, raw-observation/clipped-action=0.586029.

These unweighted summaries do not replace task/stage matched comparisons. Structural correction changes only the expert's diagnostic input; no corrected trajectory was used for BC, so retention effects cannot be inferred.

Across all saved replay groups, 43D has the lowest unweighted raw disagreement (0.620801), followed by 40D (0.693236) and 22D (0.742861). Clipping reduces these means to 0.586029, 0.640106, and 0.680621 respectively, without changing their ordering.

Both structural corrections leave 22D exactly unchanged, as expected. They do not improve the aggregate 40D or 43D diagnostic: both-corrected means rise to 0.716620 and 0.660172. Effects are task/group dependent, so inconsistency is not simply responsible for expert disagreement.

Handle provides partial descriptive alignment with retention: seed-1 40D has the largest Handle disagreement (1.626462 at stage 2; 1.290498 at stage 3) and the worst final Handle success (0.15), while seed-1 43D has lower disagreement (0.581894/0.792718) and retains 1.00. Counterexamples prevent a general rule: seed-2 43D Handle disagreement reaches 1.194834 yet final success is 0.35, and seed-0 22D reaches 1.292037 with final success 0.50.

Hammer is a strong counterexample to using disagreement alone: seed-2 22D has comparatively low disagreement (0.269409 to 0.436176) but final Hammer success collapses to 0.00, whereas seed-0 22D reaches 0.841555 and retains 0.95. Button likewise sometimes retains perfectly at disagreement above 1.0.

Hammer disagreement is not monotonically increasing in general. It rises across stages for seed-1 22D and seed-1 43D, falls for seed-2 43D, and is mixed elsewhere. Handle also has mixed stage-2→3 changes. These comparisons keep task identity fixed but use 10 versus 7 trajectories for Hammer/Handle at the later stage.

With three seeds and correlated stage/task groups, associations with retention are descriptive. Counterexamples and task-specific trends should be prioritized over aggregate correlations.

## Limitations

- Expert agreement does not establish transition feasibility, reward quality, or closed-loop behavior.
- Generated observations may be off the expert's training distribution; an expert action is still produced but need not be meaningful there.
- Real-reference discrepancies include float32 serialization and SB3 inference reproducibility.
- Replay counts differ by task and stage; group means are not independent observations.
- No causal effect of correction, clipping, or representation was tested.

## Recommended experiments

1. Complete paired 22D/43D seeds 3 and 4 before strengthening representation claims.
2. Run a fixed-per-task replay control to separate representation effects from fixed-total replay dilution.
3. If disagreement is predictive after replication, prespecify a controlled replay filtering or weighting ablation; do not retrofit generated actions in the present artifacts.

Expert SHA-256 identities: `{"kuka-button-press-v3": "ee92f679f342a3609fdc56c4379641d7e60123fc3443e558203685577674f2f0", "kuka-hammer-v3": "1900fe4dd8d3150f7d484f7f8886f4e65616bf11fa681d5a29ae6b7a06696c8c", "kuka-handle-press-side-v3": "090824e59d5dcfe3a3be0700a020877b97e5545b3bf9e507a02d1b132dc47049", "kuka-reach-v3": "2be77e7884b1fdf0b29d364ec0a5dcfdae51b0abb34b56f7ecaded5e6be32792"}`.
