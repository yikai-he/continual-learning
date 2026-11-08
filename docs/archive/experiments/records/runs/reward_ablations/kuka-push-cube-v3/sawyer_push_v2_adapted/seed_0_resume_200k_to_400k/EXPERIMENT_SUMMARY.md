# SAC Experiment Summary

- Task: kuka-push-cube-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 400000
- Reward: sawyer_push_v2_adapted
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 220000 | 12.0% | 103.8040 | 287.4378 |
| 240000 | 6.0% | 69.7117 | 215.3894 |
| 260000 | 0.0% | 29.8623 | 31.4834 |
| 280000 | 4.0% | 37.1371 | 38.9284 |
| 300000 | 0.0% | 41.5270 | 78.7014 |
| 320000 | 2.0% | 59.2204 | 215.7757 |
| 340000 | 2.0% | 104.6181 | 241.2825 |
| 360000 | 10.0% | 152.3373 | 411.9600 |
| 380000 | 0.0% | 62.3908 | 107.2557 |
| 400000 | 8.0% | 75.4172 | 234.8366 |

## Best Checkpoints

- Best success: step 220000 (12.0%), `best_success_model/best_success_model.zip`
- Best return: step 360000 (152.3373), `best_model/best_model.zip`
- Final model: `final_model.zip`
