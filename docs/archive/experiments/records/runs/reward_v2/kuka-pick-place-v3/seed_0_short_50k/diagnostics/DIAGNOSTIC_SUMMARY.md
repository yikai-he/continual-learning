# KUKA Pick Place Diagnostic Summary

Deterministic evaluation of the saved policies only: seeds 10000–10019, 200-step horizon, reward v2, and no termination on success.

## Stage counts

| Furthest stage | best_model | final_model |
|---|---:|---:|
| no meaningful approach | 20 | 20 |
| reached object | 0 | 0 |
| valid caging/contact | 0 | 0 |
| grasped | 0 | 0 |
| lifted | 0 | 0 |
| transported toward goal | 0 | 0 |
| success | 0 | 0 |

## Key metrics

### best_model

- Median/minimum TCP-to-object distance: 0.0972 / 0.0425 m
- Near object: 0/20
- Bilateral contact: 0/20
- Grasp success: 0/20
- Lift over 1 cm: 0/20
- Meaningful transport: 0/20
- Success: 0/20

### final_model

- Median/minimum TCP-to-object distance: 0.1498 / 0.0462 m
- Near object: 0/20
- Bilateral contact: 0/20
- Grasp success: 0/20
- Lift over 1 cm: 0/20
- Meaningful transport: 0/20
- Success: 0/20

## Conclusion

Main failure: **reaching/exploration failure**.

Ranked likely causes:

1. **Exploration difficulty.** The deterministic saved policies rarely reach the 3 cm near-object threshold and do not reliably discover the sequential contact, grasp, lift, and transport behavior.
2. **Reward shaping issue.** There is no standalone TCP-reaching term: the base reward couples caging with object-to-goal progress, while the stronger lift reward is available only after a close, lifted configuration. The low grasp rewards and absence of near-object episodes show that this signal did not teach reliable reach.
3. **Insufficient training steps.** This run has 50k environment steps and only 40k steps after learning starts, while neither saved policy establishes the first contact-dependent stages.

Thresholds: reached object uses the task's `near_object` (TCP distance ≤3 cm); contact is simultaneous left/right pad contact; grasp uses the task's `grasp_success`; lift is object rise >1 cm, matching the v2 lift-reward gate; meaningful transport is a ≥5 cm object-to-goal reduction after lift; success is the unmodified task `success` field.
