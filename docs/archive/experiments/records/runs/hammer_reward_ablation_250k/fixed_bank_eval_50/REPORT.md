# Hammer reward-ablation: 50-task fixed-bank evaluation

## Protocol

- Deterministic SAC inference; no training was launched.
- One ordered bank of 50 `kuka-hammer-v3` task configurations, generated with task-set seed 10000.
- Reset/evaluation seeds were 10000 through 10049 for every checkpoint.
- The ordered task hashes and seeds are recorded in both JSON result files.
- Each checkpoint was evaluated with its matching reward configuration: original/0, nail-progress/2, or nail-progress/4.
- Returns are reported for completeness but are not compared across reward variants because their scales differ.

## Located runs and checkpoints

| Variant | Run directory | Actual final training step | Early stopped | Best-success step | Artifacts |
|---|---|---:|---|---:|---|
| Original | `original_seed_0` | 250,000 | No | 20,000 | `final_model.zip`; `best_success_model/best_success_model.zip`; `best_model/best_model.zip` |
| Progress w2 | `progress_w2_seed_0` | 250,000 | No | 120,000 | `final_model.zip`; `best_success_model/best_success_model.zip`; `best_model/best_model.zip` |
| Progress w4 | `progress_w4_seed_0` | 120,000 | Yes, at 120,000 | 100,000 | `final_model.zip`; `best_success_model/best_success_model.zip`; `best_model/best_model.zip` |

All requested artifacts were present; no fallback checkpoint was used. For w2, the first 100% evaluation was at 60k, but the saved best-success checkpoint is 120k because success tied at 100% and mean return was higher within that variant.

## Comparison A: final models

| Variant (checkpoint step) | Success | Mean return | Mean final qpos | Mean max qpos | Median max | Best max | >=.03 | >=.05 | >=.08 | >.09 | Contact frac. | Mean contact steps | Min proxy-goal | Min striking-goal | Proxy error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original (250k) | 6% | 635.87 | 0.0116 | 0.0117 | 0.0000 | 0.1010 | 12% | 12% | 6% | 6% | 10% | 0.10 | 0.0749 | 0.0867 | 0.0333 |
| Progress w2 (250k) | 54% | 380.31 | 0.0570 | 0.0573 | 0.1009 | 0.1025 | 58% | 58% | 54% | 54% | 0% | 0.00 | 0.2416 | 0.2492 | 0.0185 |
| Progress w4 (120k, early stop) | 96% | 1411.34 | 0.0969 | 0.1002 | 0.1030 | 0.1064 | 98% | 98% | 96% | 96% | 8% | 0.12 | 0.0920 | 0.1207 | 0.0460 |

## Comparison B: best-success models

| Variant (checkpoint step) | Success | Mean return | Mean final qpos | Mean max qpos | Median max | Best max | >=.03 | >=.05 | >=.08 | >.09 | Contact frac. | Mean contact steps | Min proxy-goal | Min striking-goal | Proxy error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original (20k) | 42% | 247.29 | 0.0452 | 0.0472 | 0.0204 | 0.1031 | 50% | 46% | 42% | 42% | 4% | 0.48 | 0.1758 | 0.1943 | 0.0271 |
| Progress w2 (120k) | 100% | 534.72 | 0.0998 | 0.1027 | 0.1028 | 0.1052 | 100% | 100% | 100% | 100% | 0% | 0.00 | 0.2505 | 0.2518 | 0.0141 |
| Progress w4 (100k) | 100% | 1510.45 | 0.0965 | 0.1029 | 0.1028 | 0.1068 | 100% | 100% | 100% | 100% | 8% | 0.12 | 0.1080 | 0.1300 | 0.0467 |

The explicit contact diagnostic is unexpectedly sparse (including zero detected contacts for successful w2 rollouts), so it should not override the unchanged success signal and directly observed nail displacement. This pattern indicates that the current contact detector does not capture every effective nail-driving interaction.

## Learning speed from saved evaluation history

| Variant | First >=50% | First >=80% | First >=95% | First 100% | Three consecutive >=95% | Early-stop step |
|---|---:|---:|---:|---:|---|---:|
| Original | Never | Never | Never | Never | No | — |
| Progress w2 | 40k | 60k | 60k | 60k | No | — |
| Progress w4 | 60k | 80k | 80k | 80k | Yes (80k, 100k, 120k) | 120k |

The history uses 20-task evaluations every 20k steps. w2 reached high success earlier but fluctuated; w4 reached it 20k later and satisfied the stability stopping rule.

## Final versus best-success selection

| Variant | Best-success result | Final result | Assessment |
|---|---:|---:|---|
| Original | 42% at 20k | 6% at 250k | Material regression: -36 percentage points |
| Progress w2 | 100% at 120k | 54% at 250k | Material regression: -46 percentage points |
| Progress w4 | 100% at 100k | 96% at 120k | No material regression: -4 percentage points (2/50 tasks) |

## Interpretation and next experiment

Direct nail-progress shaping is strongly supported: both shaped best-success checkpoints achieved 50/50 successes, versus 21/50 for original, and both learned much faster than the original variant.

The appropriate w2-versus-w4 conclusion is **D: seed 0 alone is insufficient to distinguish them globally**. Best-success selection ties them at 100%, while final-model selection strongly favors w4 (96% versus 54%) and shows better within-run stability/sample-efficiency for w4 under the early-stop rule. Thus w4 has the better seed-0 robustness evidence, but this does not establish that weight 4 is globally optimal.

Recommended next manual SAC experiment: **Option B**, w2 and w4 with seeds 0, 1, and 2, a 250k cap, the same early stopping, and the same 50-task fixed-bank evaluation for both final and best-success checkpoints. Preserve both checkpoint-selection analyses: best-success measures attainable policy quality, while final/early-stop performance measures training stability. The original baseline need not be rerun to decide between weights because its seed-0 evidence already establishes the shaping benefit; include it only if the final study requires matched multi-seed baseline uncertainty.
