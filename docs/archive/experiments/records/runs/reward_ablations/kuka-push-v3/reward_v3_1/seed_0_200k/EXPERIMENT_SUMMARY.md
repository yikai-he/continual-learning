# SAC Experiment Summary

- Task: kuka-push-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 200000
- Reward: v3_1
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 4.5697 | 0.8329 |
| 20000 | 0.0% | 4.6522 | 0.4439 |
| 40000 | 0.0% | 3.5946 | 0.8547 |
| 60000 | 0.0% | 4.4067 | 2.7543 |
| 80000 | 2.0% | 17.7432 | 27.8881 |
| 100000 | 12.0% | 46.3439 | 50.5027 |
| 120000 | 8.0% | 42.4006 | 46.6137 |
| 140000 | 22.0% | 63.3347 | 61.4679 |
| 160000 | 24.0% | 74.8616 | 65.9536 |
| 180000 | 10.0% | 28.3201 | 53.2888 |
| 200000 | 0.0% | 19.4352 | 22.2735 |

## Best Checkpoints

- Best success: step 160000 (24.0%), `best_success_model/best_success_model.zip`
- Best return: step 160000 (74.8616), `best_model/best_model.zip`
- Final model: `final_model.zip`
