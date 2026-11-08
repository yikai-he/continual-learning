# SAC Experiment Summary

- Task: kuka-push-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 11.3555 | 3.9510 |
| 5000 | 0.0% | 9.5173 | 2.3251 |
| 10000 | 0.0% | 9.6138 | 2.1377 |
| 15000 | 0.0% | 7.7432 | 2.1196 |
| 20000 | 0.0% | 10.5806 | 1.4763 |
| 25000 | 0.0% | 54.7743 | 107.9737 |
| 30000 | 0.0% | 52.6102 | 179.3322 |
| 35000 | 0.0% | 22.8969 | 25.7107 |
| 40000 | 0.0% | 10.7302 | 6.9734 |
| 45000 | 0.0% | 24.5973 | 23.8440 |
| 50000 | 5.0% | 23.5809 | 40.0465 |

## Best Checkpoints

- Best success: step 50000 (5.0%), `best_success_model/best_success_model.zip`
- Best return: step 25000 (54.7743), `best_model/best_model.zip`
- Final model: `final_model.zip`
