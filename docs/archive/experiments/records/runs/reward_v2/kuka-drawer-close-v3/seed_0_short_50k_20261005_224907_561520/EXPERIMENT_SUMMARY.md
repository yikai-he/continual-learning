# SAC Experiment Summary

- Task: kuka-drawer-close-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 0.0000 | 0.0000 |
| 5000 | 0.0% | 0.0000 | 0.0000 |
| 10000 | 0.0% | 0.0000 | 0.0000 |
| 15000 | 100.0% | 1764.3163 | 32.5436 |
| 20000 | 100.0% | 1751.6833 | 34.6693 |
| 25000 | 100.0% | 1683.1077 | 183.3943 |
| 30000 | 100.0% | 1664.7687 | 292.9818 |
| 35000 | 100.0% | 1745.2921 | 188.9927 |
| 40000 | 100.0% | 1775.6493 | 22.0627 |
| 45000 | 100.0% | 1786.0253 | 5.7665 |
| 50000 | 100.0% | 1789.6505 | 4.5678 |

## Best Checkpoints

- Best success: step 50000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 50000 (1789.6505), `best_model/best_model.zip`
- Final model: `final_model.zip`
