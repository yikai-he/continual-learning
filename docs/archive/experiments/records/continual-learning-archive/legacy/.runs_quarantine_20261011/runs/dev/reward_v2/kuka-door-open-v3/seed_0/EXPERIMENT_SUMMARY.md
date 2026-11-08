# SAC Experiment Summary

- Task: kuka-door-open-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 500000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 136.4118 | 27.3700 |
| 20000 | 0.0% | 490.9388 | 84.1596 |
| 40000 | 80.0% | 991.7070 | 238.8487 |
| 60000 | 100.0% | 1365.1583 | 309.8014 |
| 80000 | 100.0% | 1519.8288 | 45.0648 |
| 100000 | 80.0% | 1448.6666 | 257.3299 |
| 120000 | 100.0% | 1598.9933 | 26.9980 |
| 140000 | 100.0% | 1594.6921 | 38.6330 |
| 160000 | 100.0% | 1637.7160 | 25.2128 |

## Best Checkpoints

- Best success: step 160000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 160000 (1637.7160), `best_model/best_model.zip`
- Final model: `final_model.zip`
