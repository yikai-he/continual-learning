# KUKA Pick Place Scripted Reaching Diagnostic

Deterministic scripted control on seeds 10000–10019, horizon 200, reward v2, and `terminate_on_success=False`. The controller uses only normal environment actions, keeps the gripper open, and stops at the official 3 cm near-object threshold.

## Result

- Success reaching ≤3 cm: **20/20**
- Median initial distance: **0.1578 m**
- Median minimum distance: **0.0267 m**
- Median steps to ≤3 cm: **55.0**
- Median reward increase during approach: **0.004804**
- Median per-episode reward/distance Spearman correlation: **-0.5172** (negative means reward rises as distance falls)
- Median fraction of closer steps with non-decreasing reward: **94.3%**
- Maximum grasp reward across episodes: **0.044039**
- Maximum in-place reward range within an episode: **0.000171166**
- Unexpected contact episodes: **0/20**
- Unexpected object-motion episodes (>1 mm): **0/20**

## Interpretation

The reward **does not provide a useful reaching gradient** by the diagnostic criterion (median increase ≥0.05 and median per-episode correlation ≤−0.5). Its relationship to distance is directionally consistent during approach.
Workspace/action scaling **does not prevent reliable reaching**.
`in_place_reward` tracks object-to-target distance, so it is effectively independent of TCP approach while the object remains stationary.
`grasp_reward` is the only pre-grasp component that changes with the TCP; the total reward is its Hamacher product with `in_place_reward`.

## Saved-policy comparison

- best_model: near object in 0/20 episodes; median closest distance 0.0972 m.
- final_model: near object in 0/20 episodes; median closest distance 0.1498 m.

## Failed seeds

None.

## Classification

**weak reaching reward signal likely**
