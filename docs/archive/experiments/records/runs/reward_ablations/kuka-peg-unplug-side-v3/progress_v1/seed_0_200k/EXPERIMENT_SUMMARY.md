# SAC Experiment Summary

- Task: kuka-peg-unplug-side-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 200000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 11.0039 | 2.9241 |
| 20000 | 0.0% | 11.0460 | 3.8059 |
| 40000 | 22.0% | 30.2020 | 23.1021 |
| 60000 | 20.0% | 76.9438 | 58.0367 |
| 80000 | 40.0% | 160.4171 | 117.8895 |
| 100000 | 14.0% | 268.9258 | 124.0324 |
| 120000 | 54.0% | 380.9030 | 141.2627 |
| 140000 | 58.0% | 133.1016 | 107.5519 |
| 160000 | 60.0% | 444.8609 | 239.0990 |
| 180000 | 74.0% | 729.9166 | 409.6069 |
| 200000 | 86.0% | 406.3469 | 260.3628 |

## Best Checkpoints

- Best success: step 200000 (86.0%), `best_success_model/best_success_model.zip`
- Best return: step 180000 (729.9166), `best_model/best_model.zip`
- Final model: `final_model.zip`
