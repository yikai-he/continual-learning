---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# KUKA Push reward-v3.1 SAC experiment

## Purpose

This is a controlled, from-scratch SAC experiment for `kuka-push-v3`. It tests whether reward `v3_1` improves learning under the repository's established SAC settings. Preparing this experiment does **not** establish an improvement; that claim requires comparison of completed runs under the same evaluation protocol.

The experiment config is `configs/sac/kuka_push_v3_reward_v3_1_200k.yaml`. Do not pass `--resume-from`: this run is intentionally initialized from scratch with seed 0.

## Reward under test

Let `d_t` be object-to-goal distance and `p_t = clip(d_(t-1) - d_t, -0.005, 0.005)`. Reward `v3_1` is phase-based:

- After success has already been latched: `0`.
- On the first step satisfying `d_t <= 0.05 m`: `100`, then latch success until reset.
- While both gripper-pad contact and an open TCP are detected: `0.05 + 1000 p_t`.
- Before contact: the existing high-density gripper-caging reward.

Unlike `v3`, `v3_1` does not retain `0.1 * caging` during contact. It uses the full caging signal only before contact, then switches to the contact/progress term. Both versions use the same clipped signed progress and one-time success bonus.

## Controlled settings

| Setting | Value |
|---|---:|
| Environment backend / task | `kuka-v3` / `kuka-push-v3` |
| Reward | `v3_1` |
| Maximum steps | 200,000 |
| Seed | 0 |
| Learning rate / batch size | 0.001 / 128 |
| Replay capacity / learning starts | 1,000,000 / 10,000 |
| Network | `[256, 256, 256, 256]` |
| Gamma / tau | 0.99 / 0.005 |
| Train frequency / gradient steps | 1 / 1 |
| Entropy coefficient | `auto` |
| Evaluation | 50 fixed tasks every 20,000 steps |
| Evaluation task-bank seed | 10,000 |
| Checkpoint interval | 50,000 steps |
| Early stopping | at least 100,000 steps, success >= 0.95 for 3 consecutive evaluations |

Evaluation uses deterministic policy actions. With 50 episodes, `success >= 0.95` means at least 48 successes: 47/50 is 0.94 and 48/50 is 0.96. Evaluation has its own environment and fixed ordered task bank; disabling sampling there does not disable the training environment's randomized task sampling.

## Output and artifact behavior

The preferred run directory is:

`runs/reward_ablations/kuka-push-v3/reward_v3_1/seed_0_200k`

If it already exists, `train_sac.py` creates a timestamped sibling rather than overwriting it. Use the actual path printed by the launcher for all later commands.

Training writes deterministic evaluation history, best-return and best-success models, TensorBoard events, periodic checkpoints, and a final model. Periodic checkpoints at 50,000-step global boundaries are paired with replay buffers (`checkpoints/sac_<step>_steps.zip` and `checkpoints/sac_replay_buffer_<step>_steps.pkl`). The final model and evaluation-selected best models do **not** have replay-buffer sidecars. Consequently, a normal 200,000-step completion has a resumable 200,000-step pair, but an early-stopped run may end between checkpoint boundaries and its exact final state is not resumable. No training-code change is included in this preparation.

## Manual launch (not run during preparation)

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
conda run --no-capture-output -n metaworld \
  python -u train_sac.py \
  --config configs/sac/kuka_push_v3_reward_v3_1_200k.yaml
```

Do not add `--resume-from` for this experiment.

## Monitor and assess

For the preferred directory, start TensorBoard with:

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
conda run --no-capture-output -n metaworld \
  tensorboard --logdir runs/reward_ablations/kuka-push-v3/reward_v3_1/seed_0_200k/tensorboard
```

Inspect `evaluations/evaluations.npz`, `evaluation_history.csv`, `run_summary.json`, and `EXPERIMENT_SUMMARY.md` in the actual run directory. The in-training evaluator already measures every saved best checkpoint candidate on the requested fixed 50-task bank and records task hashes and deterministic results.

The repository's current `scripts/evaluation/evaluate_sac.py` CLI cannot faithfully re-evaluate this experiment: its backend choices omit `kuka-v3` and its reward choices omit `v3_1`. Do not substitute its defaults, because that would evaluate a different environment/reward contract. A standalone re-evaluation command requires a separate evaluator compatibility change; that is deliberately outside this config-only preparation.

Compare learning curves and fixed-bank success rates against controls with identical seed, budget, SAC parameters, and task bank. Report both success rate and return distribution, and use multiple training seeds before claiming that `v3_1` improves performance.
