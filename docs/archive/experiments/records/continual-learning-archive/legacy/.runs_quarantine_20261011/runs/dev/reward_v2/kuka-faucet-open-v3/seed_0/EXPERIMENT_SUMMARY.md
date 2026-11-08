# SAC Experiment Summary

- Task: kuka-faucet-open-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 500000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 670.6691 | 75.0649 |
| 20000 | 20.0% | 806.6915 | 94.3960 |
| 40000 | 0.0% | 634.7391 | 144.8602 |
| 60000 | 80.0% | 1759.8823 | 160.4462 |
| 80000 | 100.0% | 1694.6148 | 250.8271 |
| 100000 | 100.0% | 1838.1571 | 41.9504 |
| 120000 | 100.0% | 1870.3122 | 36.2353 |

## Best Checkpoints

- Best success: step 120000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 120000 (1870.3122), `best_model/best_model.zip`
- Final model: `final_model.zip`
