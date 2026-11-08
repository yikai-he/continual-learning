# SAC Experiment Summary

- Task: kuka-push-cube-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 200000
- Reward: v6_rear_contact_push_progress
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 0.0047 | 0.0791 |
| 20000 | 0.0% | 0.0047 | 0.0791 |
| 40000 | 0.0% | 0.0128 | 0.0005 |
| 60000 | 0.0% | -0.0007 | 0.0012 |
| 80000 | 0.0% | 0.0090 | 0.0228 |
| 100000 | 0.0% | 0.1028 | 0.8537 |
| 120000 | 0.0% | 0.6429 | 0.2577 |
| 140000 | 14.0% | 1.8444 | 4.2841 |
| 160000 | 4.0% | 1.0534 | 2.1940 |
| 180000 | 22.0% | 3.3396 | 4.6459 |
| 200000 | 22.0% | 3.1628 | 4.4885 |

## Best Checkpoints

- Best success: step 180000 (22.0%), `best_success_model/best_success_model.zip`
- Best return: step 180000 (3.3396), `best_model/best_model.zip`
- Final model: `final_model.zip`
