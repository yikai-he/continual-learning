# SAC Experiment Summary

- Task: kuka-push-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 200000
- Reward: v3_1
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 4.5697 | 0.8329 |
| 20000 | 0.0% | 11.0162 | 1.2789 |
| 40000 | 2.0% | 17.3968 | 36.9205 |
| 60000 | 0.0% | 5.6963 | 1.3548 |
| 80000 | 0.0% | 9.9005 | 5.3490 |
| 100000 | 2.0% | 32.2971 | 37.9659 |
| 120000 | 18.0% | 73.0686 | 63.3550 |
| 140000 | 24.0% | 77.1132 | 60.6862 |
| 160000 | 12.0% | 82.6650 | 48.0423 |
| 180000 | 6.0% | 32.8414 | 44.2471 |
| 200000 | 2.0% | 30.1891 | 30.9450 |

## Best Checkpoints

- Best success: step 140000 (24.0%), `best_success_model/best_success_model.zip`
- Best return: step 160000 (82.6650), `best_model/best_model.zip`
- Final model: `final_model.zip`
