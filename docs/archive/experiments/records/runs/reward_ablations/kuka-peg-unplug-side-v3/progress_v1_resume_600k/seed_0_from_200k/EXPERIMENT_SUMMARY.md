# SAC Experiment Summary

- Task: kuka-peg-unplug-side-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 600000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 220000 | 98.0% | 998.1893 | 319.3618 |
| 240000 | 100.0% | 1272.1976 | 229.4907 |
| 260000 | 96.0% | 1228.7063 | 288.8652 |

## Best Checkpoints

- Best success: step 240000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 240000 (1272.1976), `best_model/best_model.zip`
- Final model: `final_model.zip`
