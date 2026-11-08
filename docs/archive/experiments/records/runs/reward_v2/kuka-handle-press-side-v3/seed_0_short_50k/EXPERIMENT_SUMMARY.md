# SAC Experiment Summary

- Task: kuka-handle-press-side-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 71.5491 | 10.7613 |
| 5000 | 0.0% | 64.4773 | 7.8515 |
| 10000 | 0.0% | 62.8277 | 11.5443 |
| 15000 | 0.0% | 55.0601 | 27.8042 |
| 20000 | 5.0% | 62.0721 | 19.4128 |
| 25000 | 0.0% | 169.3631 | 6.8737 |
| 30000 | 0.0% | 250.7652 | 16.6639 |
| 35000 | 0.0% | 164.1339 | 33.5398 |
| 40000 | 0.0% | 240.7637 | 7.4538 |
| 45000 | 0.0% | 198.4840 | 37.4375 |
| 50000 | 60.0% | 387.4194 | 165.2568 |

## Best Checkpoints

- Best success: step 50000 (60.0%), `best_success_model/best_success_model.zip`
- Best return: step 50000 (387.4194), `best_model/best_model.zip`
- Final model: `final_model.zip`
