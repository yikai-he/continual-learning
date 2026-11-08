# SAC Experiment Summary

- Task: kuka-button-press-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 57.3674 | 9.0010 |
| 5000 | 0.0% | 54.5130 | 6.5516 |
| 10000 | 0.0% | 56.4530 | 7.6100 |
| 15000 | 65.0% | 144.2462 | 56.6244 |
| 20000 | 95.0% | 270.9372 | 103.6724 |
| 25000 | 100.0% | 676.3226 | 448.5204 |
| 30000 | 100.0% | 1304.4091 | 483.5964 |
| 35000 | 95.0% | 1285.0984 | 521.4093 |
| 40000 | 100.0% | 1585.1387 | 399.9710 |
| 45000 | 100.0% | 1612.5161 | 340.5882 |
| 50000 | 100.0% | 1693.0003 | 357.2042 |

## Best Checkpoints

- Best success: step 50000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 50000 (1693.0003), `best_model/best_model.zip`
- Final model: `final_model.zip`
