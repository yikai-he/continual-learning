---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# KUKA Push v3.2 retention experiment

## Purpose

This controlled ablation tests one question: does post-success reward feedback improve final and sustained goal retention? It does not change initial task acquisition, contact detection, progress shaping, success radius, dynamics, observations, actions, horizon, or SAC hyperparameters. No performance improvement is assumed before training.

## Reward

Let `d_t` be puck-to-goal distance, `R=0.05`, `K_t` physical pad contact with positive gripper opening, `C_t` caging, and

`delta_t = clip(d_(t-1)-d_t, -0.005, 0.005)`.

Before historical success, v3.2 is exactly v3.1:

- if `d_t <= R` for the first time: `r_t=100` and latch success;
- otherwise, if `K_t`: `r_t=0.05 + 1000 delta_t`;
- otherwise: `r_t=C_t`.

After the latch:

- `r_t=+0.2` while `d_t<=R`;
- `r_t=-1.0` while `d_t>R`.

The latch is cleared only by environment reset. Exit and re-entry therefore pay `-1.0` then `+0.2`; the 100 bonus cannot repeat. Official `info["success"]` remains the current, non-latched distance test. Episodes still truncate at 200 steps and do not terminate on success.

## Scale and offline ranking

The existing contact-progress term remains `[-5,+5]` per contact step and first success remains 100. If success occurs on the first of 200 steps, the maximum additional retention return is `199*0.2=39.8`; at `gamma=.99` it is 17.1204. Thus retention is material but cannot exceed the first-success bonus in one episode.

For an illustrative common prefix of 100 caging steps at 0.4 followed by first success, all 200-step returns are:

| Suffix after identical first success | v3.1 undiscounted / discounted | v3.2 undiscounted / discounted |
|---|---:|---:|
| stays inside for 99 steps | 140.0 / 61.9619 | 159.8 / 66.5298 |
| exits for all 99 steps | 140.0 / 61.9619 | 41.0 / 39.1227 |
| exits 10 steps, then re-enters for 89 | 140.0 / 61.9619 | 147.8 / 62.3719 |

Meaningful partial progress, stationary contact, wrong-direction pushes, and no-contact approach receive exactly their v3.1 returns because they never activate the latch. Stable success is guaranteed to outrank transient success only when their histories through first success are identical; different pre-success progress can change the overall ordering.

The main scale concern is the asymmetric post-success signal: each outside step is five times the magnitude of each retained step. This deliberately emphasizes exit avoidance, but may encourage conservative behavior near the boundary. The ablation should be judged by physical metrics rather than return alone.

## SAC configuration

Configuration: `configs/sac/kuka_push_v3_reward_v3_2_retention_200k.yaml`.

- Fresh SAC run, seed 0, 200,000 steps, learning rate `3e-4`.
- Batch 128; replay buffer 1,000,000; learning starts 10,000.
- Gamma `.99`; tau `.005`; train frequency and gradient steps 1.
- Entropy coefficient `auto`; network `[256,256,256,256]`.
- Fixed 50-task evaluation bank, task-set seed 10000, deterministic policy.
- Evaluation reset seeds 10000--10049 every 20,000 steps.
- Model and matching replay buffer checkpoint every 20,000 steps.
- Early stopping after at least 100,000 steps, only after three consecutive evaluations at any-step success >=.95.
- Best checkpoint remains lexicographic any-step success first, mean return only as tie-breaker.

The expected run directory is
`runs/reward_ablations/kuka-push-v3/reward_v3_2_retention/seed_0_200k/`.
Run creation is collision-safe: an existing directory causes a timestamped sibling rather than overwrite. The configuration contains no resume checkpoint.

## Evaluation outputs

Existing evaluation diagnostics record any-step success, final-step success, sustained success for at least 10 consecutive steps, first-success step, longest successful run, retained-success steps/fraction, episode returns, and return standard deviation. They remain observational and do not alter checkpoint selection or early stopping. Reward diagnostics additionally expose contact, signed/clipped progress, caging, latch, first success, current goal satisfaction, retention reward, and total reward.

## Manual launch

Review the diff and tests first. Then launch manually from the training repository:

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
conda run --no-capture-output -n metaworld \
  python -u train_sac.py \
  --config configs/sac/kuka_push_v3_reward_v3_2_retention_200k.yaml
```

This command has not been executed by Codex.

## Limitations

The reward latch is hidden bookkeeping, so two physically identical post-success states can have different rewards depending on history. Retention shaping may improve overshoot without improving first acquisition. One seed is a causal screen, not evidence of robustness. Any-step checkpoint selection may still prefer a policy with weaker final retention; the auxiliary metrics must therefore be inspected explicitly.
