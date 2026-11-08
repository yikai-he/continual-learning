# SAC Experiment Summary

- Task: kuka-push-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 200000
- Reward: v3_2_retention
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 4.5697 | 0.8329 |
| 20000 | 0.0% | 11.0162 | 1.2789 |
| 40000 | 2.0% | 17.4088 | 36.9958 |
| 60000 | 0.0% | 5.2209 | 1.3895 |
| 80000 | 0.0% | 19.6970 | 18.4162 |
| 100000 | 0.0% | 5.8231 | 3.5257 |
| 120000 | 18.0% | 32.6246 | 36.3243 |
| 140000 | 8.0% | 45.0100 | 36.3584 |
| 160000 | 0.0% | 9.7405 | 13.2447 |
| 180000 | 0.0% | 7.7495 | 10.7508 |
| 200000 | 6.0% | 17.3707 | 41.9859 |

## Best Checkpoints

- Best success: step 120000 (18.0%), `best_success_model/best_success_model.zip`
- Best return: step 140000 (45.0100), `best_model/best_model.zip`
- Final model: `final_model.zip`
