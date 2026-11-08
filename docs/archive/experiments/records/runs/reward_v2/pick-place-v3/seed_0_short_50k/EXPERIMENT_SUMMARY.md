# SAC Experiment Summary

- Task: pick-place-v3
- Backend: metaworld-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 2.0266 | 0.4658 |
| 5000 | 0.0% | 1.7966 | 0.3932 |
| 10000 | 0.0% | 1.8298 | 0.3753 |
| 15000 | 0.0% | 4.0040 | 0.5872 |
| 20000 | 0.0% | 2.5545 | 0.6646 |
| 25000 | 0.0% | 1.1704 | 0.2036 |
| 30000 | 0.0% | 2.3990 | 0.5021 |
| 35000 | 0.0% | 3.4796 | 3.3846 |
| 40000 | 0.0% | 2.8971 | 2.4683 |
| 45000 | 15.0% | 26.4118 | 88.6294 |
| 50000 | 0.0% | 4.4244 | 3.9013 |

## Best Checkpoints

- Best success: step 45000 (15.0%), `best_success_model/best_success_model.zip`
- Best return: step 45000 (26.4118), `best_model/best_model.zip`
- Final model: `final_model.zip`
