---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# KUKA Cube Push reward v3.3

## Status and scope

`v3_3_contact_push_progress` is a Cube-only experimental reward for
`kuka-push-cube-v3`. It is not enabled for Cylinder Push and does not change
task physics, success, termination, observations, actions, or the 200-step
horizon. The coefficients are hypotheses to be tested by one controlled SAC
ablation; they are not claimed to be optimal.

## Reward definition

For transition `(s, a, s')`, the implemented reward is exactly

```
r = r_approach + r_first_contact + r_push
    + r_goal_progress + r_first_success
```

The goal-aware approach point is

```
u = (goal_xy - object_xy) / max(||goal_xy - object_xy||, eps)
p_approach = [object_x - 0.060 u_x, object_y - 0.060 u_y, 0.042]
Phi(s) = -||tcp - p_approach||
r_approach = 1.0 * (0.99 Phi(s') - Phi(s))
```

The remaining terms are

```
r_first_contact = 0.5 on the first detected physical gripper-cube contact,
                  otherwise 0
r_push = 100 * min(||object_xy(s') - object_xy(s)||, 0.005)
         when physical contact is detected, otherwise 0
r_goal_progress = 1000 * (||object(s) - goal||_2 - ||object(s') - goal||_2)
r_first_success = 100 on the first successful step, otherwise 0
```

Goal progress is signed, uses the same 3D center distance as success, is not
clipped, and is not gated by a particular contact geom. Rotation and tilt are
not penalized. Translation caused by rolling or tipping can therefore earn
progress. Success remains `object_goal_distance <= 0.05 m`.

## Contact and latches

The physical predicate reads MuJoCo contacts involving `objGeom` and any
collidable geom on the Robotiq `base` body or its descendants. This includes
the left/right pad geoms, left/right followers, drivers, couplers, spring
links, and the Robotiq base collision geom. Cube-floor/world contacts and the
non-colliding goal visualization are excluded. `named_pad_contact` and
`follower_contact` are logged separately.

MuJoCo advances five 2.5 ms physics frames per 80 Hz control step. Detection
currently reads the post-step contact set; a contact that begins and ends
inside those five frames can be missed. Replacing the stepping loop merely to
instrument substeps could change behavior and was not done.

First-contact and first-success latches reset in `reset_model()`. Losing and
reacquiring contact does not earn another contact bonus. Leaving and reentering
the goal does not earn another success bonus. Episodes do not terminate on
success.

## Diagnostics

Each step exposes:

- `reward_approach`, `reward_first_contact`, `reward_push`,
  `reward_goal_progress`, `reward_first_success`, `reward_total`
- `physical_robot_object_contact`, `named_pad_contact`, `follower_contact`
- `object_displacement`, `signed_goal_progress`, `object_goal_distance`
- `first_contact_step`, `first_success_step` (`-1` until absent)

Tests verify that the five components sum to `reward_total` and remain finite.
The existing v2, v3, v3_1, and v3_2_retention implementations are unchanged.

## Scale and offline measurements

The following is an exploratory offline counterfactual replay, not evidence
that SAC will learn a better policy. Deterministic policies at 60k, 100k,
180k, and 200k were replayed on fixed-bank tasks 4, 19, and 35 (seeds 10004,
10019, and 10035). Tasks 4 and 35 were selected because the archived 60k
evaluation marked them successful; Task 19 was requested. The unchanged
scripted controller was evaluated on the same tasks. Values are mean
undiscounted component sums over three episodes:

| controller | success | approach | first contact | push | goal progress | first success |
|---|---:|---:|---:|---:|---:|---:|
| 60k | 2/3 | 0.259 | 0.333 | 30.267 | 61.840 | 66.667 |
| 100k | 0/3 | 0.256 | 0.500 | 2.867 | 17.382 | 0 |
| 180k | 0/3 | 0.400 | 0.167 | 0 | -0.040 | 0 |
| 200k | 0/3 | 0.361 | 0 | 0 | 0 | 0 |
| scripted | 3/3 | 0.211 | 0.500 | 15.387 | 166.287 | 100 |

Across this subset, approach per-step ranges were approximately -0.012 to
+0.021, goal-progress per-step ranges were -11.545 to +10.024, push was capped
at 0.5, and success was a single +100 event. Mean discounted (`gamma=0.99`)
sums were:

| controller | approach | first contact | push | goal progress | first success |
|---|---:|---:|---:|---:|---:|
| 60k | 0.173 | 0.273 | 13.245 | 58.092 | 39.393 |
| 100k | 0.161 | 0.309 | 1.470 | 7.895 | 0 |
| 180k | 0.158 | 0.128 | 0 | -0.062 | 0 |
| 200k | 0.171 | 0 | 0 | 0 | 0 |
| scripted | 0.181 | 0.249 | 4.018 | 47.156 | 16.693 |

Thus the nonproductive 200k sample earned only 0.30--0.41 total, all from
approach shaping. The productive scripted episodes earned 167.30--200.77
before the success bonus and 267.30--300.77 including it. The 100k
contact-but-unsuccessful episodes
earned 0.29--41.81 depending on actual net object progress. These are only
three selected tasks, not full-bank estimates.

Actual Task 19 scripted contact inspection observed both left and right named
finger pads and the left follower. Across 200 controls it reported 103
physical-contact steps, including 7 follower-only steps. Right-follower
membership is verified by the same body-hierarchy classifier but was not
physically exercised by this trajectory.

## Known risks and interpretation

- Push reward is path-length based. Alternating two 2 mm contacted motions
  earns 0.4 per cycle even with zero net progress; 100 such cycles could earn
  40. Goal progress cancels over an undiscounted closed oscillation, but its
  discounted sum need not cancel exactly. This is a material exploitation risk
  to inspect during training; the coefficient was retained as specified.
- With `Phi=-distance` and gamma 0.99, a stationary nonterminal state receives
  `(1-gamma)*distance`, a small positive approach reward. In observed failed
  episodes its 200-step sum was about 0.3--0.4, far below productive progress,
  but it is not literally zero. A safer follow-up would use terminal-potential
  correction or a carefully bounded transition bonus, not an undocumented
  coefficient change.
- A post-step contact flag establishes association, not exact physical
  causation, for same-step object displacement.
- Potential-based policy invariance is not claimed: the environment is finite
  horizon, success is nonterminal, and no terminal potential correction is
  applied.
- Offline reward ranking cannot predict a newly trained policy.

## Controlled SAC experiment

Configuration:
`configs/sac/kuka_push_cube_v3_reward_v3_3_200k.yaml`.
It matches the v3_1 Cube baseline: seed 0, 200k steps, learning rate 3e-4,
batch 128, `[256,256,256,256]`, gamma 0.99, tau 0.005, 1M replay capacity,
10k learning start, train/gradient frequency 1, automatic entropy, 20k
evaluation/checkpoint interval, and early stopping at 95% for three evaluations
after 100k. Evaluation uses 50 fixed tasks from task-set seed 10000 and reset
seeds 10000--10049 with deterministic inference.

Expected isolated output:
`runs/reward_ablations/kuka-push-cube-v3/reward_v3_3_contact_push_progress/seed_0_200k/`.
The preflight verified this path does not exist. The existing trainer creates
periodic model/replay-buffer checkpoints, best-return and best-success models,
the final model, TensorBoard events, and evaluation diagnostics.

Manual training command (not executed):

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
conda run --no-capture-output -n metaworld \
  python -u train_sac.py \
  --config configs/sac/kuka_push_cube_v3_reward_v3_3_200k.yaml
```

TensorBoard command (not executed):

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
conda run --no-capture-output -n metaworld \
  tensorboard --logdir runs/reward_ablations/kuka-push-cube-v3/reward_v3_3_contact_push_progress/seed_0_200k/tensorboard
```
