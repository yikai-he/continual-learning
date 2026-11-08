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
| 20000 | 0.0% | 183.6620 | 28.2241 |
| 40000 | 45.0% | 685.7138 | 649.2420 |
| 60000 | 60.0% | 1015.7889 | 639.7533 |
| 80000 | 100.0% | 1544.2719 | 200.0736 |
| 100000 | 100.0% | 1592.8052 | 59.3687 |
| 120000 | 100.0% | 1582.3602 | 192.7140 |

## Best Checkpoints

- Best success: step 100000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 100000 (1592.8052), `best_model/best_model.zip`
- Final model: `final_model.zip`
