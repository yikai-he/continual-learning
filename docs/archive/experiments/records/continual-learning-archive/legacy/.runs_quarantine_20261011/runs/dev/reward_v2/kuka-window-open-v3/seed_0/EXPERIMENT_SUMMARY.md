# SAC Experiment Summary

- Task: kuka-window-open-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 500000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 84.9421 | 1.2414 |
| 20000 | 70.0% | 511.3104 | 93.6858 |
| 40000 | 100.0% | 1232.8065 | 253.7119 |
| 60000 | 90.0% | 1479.0299 | 74.5924 |
| 80000 | 100.0% | 1325.5113 | 243.4588 |
| 100000 | 100.0% | 1585.0747 | 152.7875 |
| 120000 | 100.0% | 1580.9146 | 122.3559 |

## Best Checkpoints

- Best success: step 100000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 100000 (1585.0747), `best_model/best_model.zip`
- Final model: `final_model.zip`
