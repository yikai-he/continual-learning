# SAC Experiment Summary

- Task: kuka-button-press-topdown-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 1000000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 85.8218 | 5.8325 |
| 20000 | 70.0% | 311.2184 | 34.4996 |
| 40000 | 80.0% | 486.4017 | 198.2739 |
| 60000 | 100.0% | 1322.7950 | 164.3734 |
| 80000 | 90.0% | 1298.6353 | 203.5365 |
| 100000 | 95.0% | 1278.6528 | 146.3810 |
| 120000 | 100.0% | 1260.9364 | 80.0447 |
| 140000 | 95.0% | 1302.5079 | 79.9563 |

## Best Checkpoints

- Best success: step 60000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 60000 (1322.7950), `best_model/best_model.zip`
- Final model: `final_model.zip`
