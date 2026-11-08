# SAC Experiment Summary

- Task: kuka-drawer-close-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 500000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 0.0000 | 0.0000 |
| 20000 | 70.0% | 1141.3458 | 745.9801 |
| 40000 | 100.0% | 1460.7494 | 390.7561 |
| 60000 | 100.0% | 1624.5641 | 313.1163 |
| 80000 | 100.0% | 1746.0000 | 22.8910 |
| 100000 | 100.0% | 1608.5157 | 305.4436 |

## Best Checkpoints

- Best success: step 80000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 80000 (1746.0000), `best_model/best_model.zip`
- Final model: `final_model.zip`
