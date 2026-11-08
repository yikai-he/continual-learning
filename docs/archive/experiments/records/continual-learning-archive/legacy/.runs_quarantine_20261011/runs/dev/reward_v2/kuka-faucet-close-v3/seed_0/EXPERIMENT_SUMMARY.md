# SAC Experiment Summary

- Task: kuka-faucet-close-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 1000000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 707.4701 | 57.3793 |
| 20000 | 90.0% | 1362.2202 | 686.9790 |
| 40000 | 100.0% | 1848.1259 | 42.5720 |
| 60000 | 50.0% | 1639.7430 | 269.7285 |
| 80000 | 100.0% | 1782.3245 | 118.0619 |
| 100000 | 100.0% | 1807.6962 | 242.2644 |
| 120000 | 90.0% | 1754.6630 | 218.9821 |
| 140000 | 100.0% | 1857.9618 | 48.9866 |
| 160000 | 100.0% | 1762.1929 | 216.8175 |
| 180000 | 100.0% | 1880.8749 | 46.2769 |

## Best Checkpoints

- Best success: step 180000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 180000 (1880.8749), `best_model/best_model.zip`
- Final model: `final_model.zip`
