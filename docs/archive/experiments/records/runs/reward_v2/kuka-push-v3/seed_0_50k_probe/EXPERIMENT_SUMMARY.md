# SAC Experiment Summary

- Task: kuka-push-v3
- Backend: kuka-v3
- Seed: 0
- Steps: 50000
- Reward: v2
- Horizon: 200

## Results

| Step | Success | Mean Return | Std |
|---:|---:|---:|---:|
| 0 | 0.0% | 9.2840 | 1.7343 |
| 5000 | 0.0% | 9.3449 | 1.7844 |
| 10000 | 0.0% | 9.1853 | 1.5638 |
| 15000 | 0.0% | 7.2854 | 1.5992 |
| 20000 | 0.0% | 11.1149 | 0.6671 |
| 25000 | 0.0% | 9.7093 | 2.5416 |
| 30000 | 5.0% | 26.3497 | 28.7528 |
| 35000 | 0.0% | 35.1776 | 27.6188 |
| 40000 | 0.0% | 28.0026 | 34.0057 |
| 45000 | 15.0% | 113.2136 | 124.1324 |
| 50000 | 0.0% | 64.4750 | 50.0971 |

## Best Checkpoints

- Best success: step 45000 (15.0%), `best_success_model/best_success_model.zip`
- Best return: step 45000 (113.2136), `best_model/best_model.zip`
- Final model: `final_model.zip`
