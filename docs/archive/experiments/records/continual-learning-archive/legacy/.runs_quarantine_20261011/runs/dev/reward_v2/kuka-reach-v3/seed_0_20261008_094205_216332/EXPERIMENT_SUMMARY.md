# SAC Experiment Summary

- Task: kuka-reach-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 1000000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 520000 | 100.0% | 1824.8037 | 86.8677 |
| 540000 | 80.0% | 1815.8196 | 142.8499 |
| 560000 | 90.0% | 1794.9287 | 72.8372 |
| 580000 | 85.0% | 1707.7493 | 285.0418 |
| 600000 | 95.0% | 1851.9701 | 25.1284 |
| 620000 | 95.0% | 1805.9198 | 179.4563 |
| 640000 | 95.0% | 1840.3900 | 40.1214 |

## Best Checkpoints

- Best success: step 520000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 600000 (1851.9701), `best_model/best_model.zip`
- Final model: `final_model.zip`
