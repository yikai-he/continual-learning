# SAC Experiment Summary

- Task: kuka-push-cube-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 400000
- Reward: v6_rear_contact_push_progress
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 220000 | 32.0% | 4.5334 | 5.1440 |
| 240000 | 22.0% | 3.1914 | 4.7925 |
| 260000 | 16.0% | 2.3974 | 4.1722 |
| 280000 | 52.0% | 6.4464 | 5.6615 |
| 300000 | 68.0% | 8.6453 | 5.4217 |
| 320000 | 84.0% | 10.3446 | 3.9096 |
| 340000 | 74.0% | 9.3700 | 4.6321 |
| 360000 | 84.0% | 10.2803 | 4.1995 |
| 380000 | 80.0% | 9.9537 | 4.5552 |
| 400000 | 82.0% | 10.1077 | 4.4437 |

## Best Checkpoints

- Best success: step 320000 (84.0%), `best_success_model/best_success_model.zip`
- Best return: step 320000 (10.3446), `best_model/best_model.zip`
- Final model: `final_model.zip`
