---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# KUKA Peg Unplug progress reward ablation

This controlled ablation retains the original MetaWorld V2 Peg Unplug reward
and adds one dense +X extraction-progress term. It does not change task physics,
observations, actions, resets, the 200-step horizon, or the success condition
`distance(pegEnd, goal) <= 0.07 m`.

The original reward is the existing caging/grasp reward, with its in-place branch
and completion reward of 10 unchanged. It remains the default under
`peg_unplug_reward_variant: original` with zero progress weight.

For `peg_unplug_progress_v1`, define

```
Phi(s) = clip((pegEnd_x - initial_pegEnd_x)
              / (goal_x - initial_pegEnd_x), 0, 1)
F(s_t, s_t+1) = 2.0 * (0.99 * Phi(s_t+1) - Phi(s_t))
reward = original_v2_reward + F
```

The implementation records the true potential at reset and updates it from the
post-simulation `pegEnd` position once per step. A degenerate X denominator maps
to zero. Reset replaces the saved potential, preventing cross-episode leakage.
Stationary nonzero progress receives the expected small discount penalty rather
than a persistent positive bonus, and backward progress is negative.

Gymnasium's outer `TimeLimit` truncates the registered environment at 200 steps;
the task itself does not terminate on success in SAC. Stable-Baselines3 treats
that truncation as a timeout and bootstraps from its terminal observation, so the
final potential is not forced to zero. This matches the continuing/bootstrap
semantics used by the experiment, but exact policy invariance is not claimed for
the finite training horizon, function approximation, or imperfect value learning.

The experiment uses the unchanged baseline SAC settings, 200,000 steps, 50 fixed
evaluation tasks (task-bank seed 10000), evaluation every 20,000 steps, and
checkpoints every 50,000 steps. Its collision-safe run base is
`runs/reward_ablations/kuka-peg-unplug-side-v3/progress_v1/seed_0_200k`;
an existing directory receives a timestamped sibling rather than being resumed.
Raw returns should not be compared directly across reward variants.

Run manually from the training repository (this is not run by validation):

```bash
conda run --no-capture-output -n metaworld \
  python train_sac.py \
  --config configs/sac/kuka_peg_unplug_side_v3_progress_200k.yaml
```
