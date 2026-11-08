# SAC Experiment Summary

- Task: kuka-hammer-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 194.1849 | 18.0891 |
| 5000 | 0.0% | 197.0347 | 15.3697 |
| 10000 | 0.0% | 194.0735 | 15.2095 |
| 15000 | 0.0% | 201.0935 | 20.8097 |
| 20000 | 0.0% | 88.2579 | 26.4865 |
| 25000 | 0.0% | 202.8457 | 19.9345 |
| 30000 | 0.0% | 198.1824 | 16.6703 |
| 35000 | 40.0% | 371.2236 | 171.7717 |
| 40000 | 0.0% | 170.3294 | 87.8317 |
| 45000 | 50.0% | 452.6301 | 193.4090 |
| 50000 | 50.0% | 235.4521 | 182.1350 |

## Best Checkpoints

- Best success: step 45000 (50.0%), `best_success_model/best_success_model.zip`
- Best return: step 45000 (452.6301), `best_model/best_model.zip`
- Final model: `final_model.zip`
