# SAC Experiment Summary

- Task: kuka-window-close-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 77.4611 | 5.1345 |
| 5000 | 0.0% | 78.3064 | 4.2984 |
| 10000 | 0.0% | 77.7635 | 3.8458 |
| 15000 | 10.0% | 208.0781 | 80.0374 |
| 20000 | 20.0% | 171.2703 | 47.0792 |
| 25000 | 20.0% | 299.0103 | 166.2803 |
| 30000 | 0.0% | 179.1243 | 23.3473 |
| 35000 | 85.0% | 488.4807 | 200.7739 |
| 40000 | 100.0% | 1087.7152 | 356.2241 |
| 45000 | 90.0% | 1084.3711 | 346.9711 |
| 50000 | 90.0% | 1065.1857 | 314.5744 |

## Best Checkpoints

- Best success: step 40000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 40000 (1087.7152), `best_model/best_model.zip`
- Final model: `final_model.zip`
