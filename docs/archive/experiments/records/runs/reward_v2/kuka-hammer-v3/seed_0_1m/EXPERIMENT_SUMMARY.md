# SAC Experiment Summary

- Task: kuka-hammer-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 1000000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 194.4197 | 18.6759 |
| 20000 | 0.0% | 178.8953 | 36.4910 |
| 40000 | 30.0% | 366.5781 | 388.5250 |
| 60000 | 45.0% | 746.4342 | 605.0519 |
| 80000 | 95.0% | 1308.9026 | 414.0528 |
| 100000 | 100.0% | 1545.3282 | 215.4021 |
| 120000 | 95.0% | 1426.8023 | 403.6730 |

## Best Checkpoints

- Best success: step 100000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 100000 (1545.3282), `best_model/best_model.zip`
- Final model: `final_model.zip`
