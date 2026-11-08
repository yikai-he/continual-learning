# SAC Experiment Summary

- Task: kuka-window-close-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 500000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 100.2789 | 2.0503 |
| 20000 | 50.0% | 675.2340 | 287.8578 |
| 40000 | 80.0% | 1189.0077 | 478.4178 |
| 60000 | 100.0% | 1512.1149 | 150.6668 |
| 80000 | 100.0% | 1552.6659 | 144.3860 |
| 100000 | 100.0% | 1571.7761 | 50.1831 |

## Best Checkpoints

- Best success: step 100000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 100000 (1571.7761), `best_model/best_model.zip`
- Final model: `final_model.zip`
