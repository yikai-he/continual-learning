# SAC Experiment Summary

- Task: kuka-push-cube-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 200000
- Reward: v3_1
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 4.5748 | 0.8365 |
| 20000 | 0.0% | 12.8070 | 2.3787 |
| 40000 | 0.0% | 5.5674 | 3.6626 |
| 60000 | 4.0% | 22.8618 | 40.8853 |
| 80000 | 2.0% | 13.1290 | 16.2880 |
| 100000 | 0.0% | 10.8280 | 7.3042 |
| 120000 | 0.0% | 10.6379 | 11.1108 |
| 140000 | 0.0% | 9.2147 | 27.5915 |
| 160000 | 0.0% | 7.6313 | 5.3267 |
| 180000 | 0.0% | 10.1407 | 11.6560 |
| 200000 | 0.0% | 8.4681 | 11.6753 |

## Best Checkpoints

- Best success: step 60000 (4.0%), `best_success_model/best_success_model.zip`
- Best return: step 60000 (22.8618), `best_model/best_model.zip`
- Final model: `final_model.zip`
