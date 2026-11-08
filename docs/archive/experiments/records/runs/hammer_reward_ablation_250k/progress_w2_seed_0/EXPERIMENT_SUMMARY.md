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
| 20000 | 0.0% | 176.4380 | 51.0645 |
| 40000 | 75.0% | 380.3962 | 187.7198 |
| 60000 | 100.0% | 534.9894 | 20.3480 |
| 80000 | 85.0% | 419.6336 | 142.4561 |
| 100000 | 95.0% | 459.9714 | 140.2031 |
| 120000 | 100.0% | 546.7524 | 18.4508 |
| 140000 | 90.0% | 489.1209 | 86.1948 |
| 160000 | 100.0% | 508.1033 | 60.5955 |
| 180000 | 80.0% | 422.5396 | 188.0206 |
| 200000 | 100.0% | 534.9929 | 18.1469 |
| 220000 | 100.0% | 518.4770 | 37.0593 |
| 240000 | 85.0% | 514.5983 | 39.2332 |

## Best Checkpoints

- Best success: step 120000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 120000 (546.7524), `best_model/best_model.zip`
- Final model: `final_model.zip`
