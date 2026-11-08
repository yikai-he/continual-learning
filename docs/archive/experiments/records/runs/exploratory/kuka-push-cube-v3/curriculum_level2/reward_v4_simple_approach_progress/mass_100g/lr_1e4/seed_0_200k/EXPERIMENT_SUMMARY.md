# SAC Experiment Summary

- Task: kuka-push-cube-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 200000
- Reward: v4_simple_approach_progress
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 0.1324 | 0.0043 |
| 20000 | 0.0% | 0.1324 | 0.0043 |
| 40000 | 0.0% | 0.1317 | 0.0002 |
| 60000 | 0.0% | 0.1253 | 0.0021 |
| 80000 | 0.0% | 0.3003 | 0.0418 |
| 100000 | 0.0% | 0.2200 | 0.0180 |
| 120000 | 2.0% | 0.3352 | 1.7011 |
| 140000 | 0.0% | 0.4519 | 0.3552 |
| 160000 | 0.0% | 0.5222 | 0.2218 |
| 180000 | 0.0% | 0.4140 | 0.5217 |
| 200000 | 0.0% | 0.3209 | 0.5197 |

## Best Checkpoints

- Best success: step 120000 (2.0%), `best_success_model/best_success_model.zip`
- Best return: step 160000 (0.5222), `best_model/best_model.zip`
- Final model: `final_model.zip`
