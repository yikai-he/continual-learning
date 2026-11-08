# SAC Experiment Summary

- Task: kuka-hammer-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 250000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 203.0484 | 16.5671 |
| 20000 | 45.0% | 253.5099 | 105.5074 |
| 40000 | 0.0% | 300.6336 | 127.7386 |
| 60000 | 10.0% | 541.7353 | 68.4682 |
| 80000 | 25.0% | 503.4029 | 148.2564 |
| 100000 | 5.0% | 464.8836 | 148.8554 |
| 120000 | 0.0% | 456.5445 | 153.4629 |
| 140000 | 5.0% | 559.1995 | 188.4299 |
| 160000 | 5.0% | 690.8899 | 118.7463 |
| 180000 | 0.0% | 445.0509 | 116.2184 |
| 200000 | 5.0% | 661.0758 | 124.4897 |
| 220000 | 0.0% | 576.6467 | 174.0433 |
| 240000 | 5.0% | 722.1138 | 54.7569 |

## Best Checkpoints

- Best success: step 20000 (45.0%), `best_success_model/best_success_model.zip`
- Best return: step 240000 (722.1138), `best_model/best_model.zip`
- Final model: `final_model.zip`
