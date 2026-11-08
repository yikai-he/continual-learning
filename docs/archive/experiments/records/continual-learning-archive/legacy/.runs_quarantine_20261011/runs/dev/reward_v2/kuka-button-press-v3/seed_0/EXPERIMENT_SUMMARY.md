# SAC Experiment Summary

- Task: kuka-button-press-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 500000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 73.1704 | 9.1587 |
| 20000 | 30.0% | 457.7506 | 132.1178 |
| 40000 | 60.0% | 1269.6747 | 183.8170 |
| 60000 | 60.0% | 1111.1135 | 57.0372 |
| 80000 | 100.0% | 1386.4079 | 173.6318 |
| 100000 | 70.0% | 1322.4618 | 168.6795 |
| 120000 | 100.0% | 1365.0288 | 116.4318 |
| 140000 | 90.0% | 1450.8136 | 81.5047 |
| 160000 | 100.0% | 1446.0069 | 67.2340 |
| 180000 | 100.0% | 1452.3361 | 10.5745 |
| 200000 | 100.0% | 1485.6740 | 16.9977 |

## Best Checkpoints

- Best success: step 200000 (100.0%), `best_success_model/best_success_model.zip`
- Best return: step 200000 (1485.6740), `best_model/best_model.zip`
- Final model: `final_model.zip`
