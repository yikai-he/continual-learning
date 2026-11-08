# SAC Experiment Summary

- Task: kuka-faucet-close-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 675.9466 | 77.4492 |
| 5000 | 0.0% | 675.4391 | 42.5602 |
| 10000 | 0.0% | 684.2504 | 58.9838 |
| 15000 | 20.0% | 1090.0232 | 396.9721 |
| 20000 | 70.0% | 1523.8179 | 589.1131 |
| 25000 | 15.0% | 747.9605 | 107.2784 |
| 30000 | 60.0% | 1395.6574 | 618.7896 |
| 35000 | 65.0% | 1476.2325 | 433.9249 |
| 40000 | 50.0% | 1506.1318 | 418.3176 |
| 45000 | 35.0% | 1382.0815 | 505.6629 |
| 50000 | 80.0% | 1663.3324 | 433.7347 |

## Best Checkpoints

- Best success: step 50000 (80.0%), `best_success_model/best_success_model.zip`
- Best return: step 50000 (1663.3324), `best_model/best_model.zip`
- Final model: `final_model.zip`
