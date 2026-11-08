# SAC Experiment Summary

- Task: kuka-faucet-open-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 563.4643 | 82.6429 |
| 5000 | 0.0% | 538.4744 | 82.9372 |
| 10000 | 0.0% | 527.3648 | 75.6836 |
| 15000 | 0.0% | 828.8376 | 10.2356 |
| 20000 | 0.0% | 799.5757 | 38.5162 |
| 25000 | 0.0% | 863.9234 | 69.2530 |
| 30000 | 40.0% | 940.1376 | 591.2639 |
| 35000 | 100.0% | 1682.3991 | 298.3824 |
| 40000 | 45.0% | 622.5760 | 206.3277 |
| 45000 | 95.0% | 1577.0048 | 311.7962 |
| 50000 | 80.0% | 1660.7535 | 385.1961 |

## Best Checkpoints

- Best success: step 35000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 35000 (1682.3991), `best_model/best_model.zip`
- Final model: `final_model.zip`
