# SAC Experiment Summary

- Task: kuka-push-cube-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 200000
- Reward: sawyer_push_v2_adapted
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 9.1497 | 1.6730 |
| 20000 | 0.0% | 56.0075 | 64.5774 |
| 40000 | 0.0% | 14.1525 | 4.1580 |
| 60000 | 0.0% | 12.3760 | 10.6806 |
| 80000 | 8.0% | 88.5157 | 219.8259 |
| 100000 | 0.0% | 39.2930 | 92.2849 |
| 120000 | 0.0% | 40.8145 | 40.9376 |
| 140000 | 6.0% | 152.8948 | 371.0561 |
| 160000 | 0.0% | 52.6901 | 62.3768 |
| 180000 | 0.0% | 23.8453 | 13.4124 |
| 200000 | 18.0% | 155.4092 | 308.3080 |

## Best Checkpoints

- Best success: step 200000 (18.0%), `best_success_model/best_success_model.zip`
- Best return: step 200000 (155.4092), `best_model/best_model.zip`
- Final model: `final_model.zip`
