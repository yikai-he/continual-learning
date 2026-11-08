# SAC Experiment Summary

- Task: kuka-window-open-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 76.8656 | 1.7217 |
| 5000 | 0.0% | 77.6082 | 1.0367 |
| 10000 | 0.0% | 77.6200 | 1.0854 |
| 15000 | 40.0% | 193.5748 | 89.1554 |
| 20000 | 0.0% | 83.7723 | 7.9340 |
| 25000 | 10.0% | 177.0276 | 190.1912 |
| 30000 | 65.0% | 380.3146 | 334.3313 |
| 35000 | 100.0% | 1094.3146 | 268.8535 |
| 40000 | 95.0% | 1042.1743 | 357.0849 |
| 45000 | 90.0% | 1254.5546 | 243.1764 |
| 50000 | 85.0% | 1177.0520 | 264.1352 |

## Best Checkpoints

- Best success: step 35000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 45000 (1254.5546), `best_model/best_model.zip`
- Final model: `final_model.zip`
