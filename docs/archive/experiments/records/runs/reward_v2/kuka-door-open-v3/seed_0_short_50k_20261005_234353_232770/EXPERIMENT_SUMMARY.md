# SAC Experiment Summary

- Task: kuka-door-open-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 139.0634 | 33.3765 |
| 5000 | 0.0% | 130.4779 | 29.1761 |
| 10000 | 0.0% | 132.9241 | 22.5671 |
| 15000 | 10.0% | 389.2417 | 267.6847 |
| 20000 | 35.0% | 886.0753 | 566.1202 |
| 25000 | 65.0% | 1278.2177 | 360.3200 |
| 30000 | 15.0% | 449.2407 | 291.2123 |
| 35000 | 15.0% | 645.2049 | 399.4189 |
| 40000 | 90.0% | 1519.3248 | 344.7760 |
| 45000 | 90.0% | 1340.0408 | 333.3710 |
| 50000 | 95.0% | 1537.0485 | 211.8539 |

## Best Checkpoints

- Best success: step 50000 (95.0%), `best_success_model/best_success_model.zip`
- Best return: step 50000 (1537.0485), `best_model/best_model.zip`
- Final model: `final_model.zip`
