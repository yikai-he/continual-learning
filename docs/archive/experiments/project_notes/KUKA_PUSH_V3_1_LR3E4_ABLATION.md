---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# KUKA Push v3.1 learning-rate ablation

This is a learning-rate ablation, not a new reward experiment. It keeps the completed reward-v3.1 experiment contract and changes SAC learning rate from `0.001` to `0.0003`. It starts from scratch with training seed 0; do not pass `--resume-from`.

## Configuration

The config is `configs/sac/kuka_push_v3_reward_v3_1_lr3e4_200k.yaml`. It retains the 200,000-step budget, batch size 128, `[256, 256, 256, 256]` network, 1,000,000 replay capacity, 10,000 learning starts, gamma 0.99, tau 0.005, train frequency 1, one gradient step, and automatic entropy tuning.

Evaluation remains deterministic on the same ordered 50-task bank (`fixed-tasks`, task-set seed 10000, reset seeds 10000–10049) every 20,000 steps. Training continues to sample randomized tasks independently. The official any-step environment success remains the checkpoint-selection and early-stopping metric; mean return only breaks success-rate ties.

Early stopping remains enabled after 100,000 steps at success rate >=0.95 for three consecutive evaluations. With 50 episodes, this requires at least 48 successes.

Periodic checkpoints now use the existing 20,000-step global schedule. Each periodic model is paired with a replay buffer and retains the serialized SAC optimizer and entropy state required by existing resume support. Best-success, best-return, and final models remain separate model-only artifacts.

The evaluator additionally records observational metrics without changing stepping or learning: final-step success, at least 10 consecutive successful steps, longest success run, and the count/fraction of successful steps after first success. These do not influence checkpoint selection or early stopping.

## Renderer and comparison videos

Select reward semantics explicitly with `--reward-version`; omission remains backward-compatible and defaults to `v2`:

```bash
python scripts/render_kuka_push_policy.py \
  --model PATH/TO/MODEL.zip \
  --env-name kuka-push-v3 \
  --reward-version v3_1 \
  --task-set-seed 10000 --task-index 19 --seeds 10019
```

The 160k-versus-200k comparison videos are stored under `videos/kuka-push-v3/v3_1_diagnostics/`.

- `kuka-push-v3_160k_task{0,6,7,19}.mp4`
- `kuka-push-v3_200k_task{0,6,7,19}.mp4`

The completed control used reward `v3_1`, learning rate `0.001`, seed 0, and the same 200,000-step budget and fixed evaluation bank. Its isolated output remains `runs/reward_ablations/kuka-push-v3/reward_v3_1/seed_0_200k`.

## Output and manual launch

The preferred output is `runs/reward_ablations/kuka-push-v3/reward_v3_1_lr3e4/seed_0_200k`. Existing collision-safe handling creates a timestamped sibling if that directory exists.

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
conda run --no-capture-output -n metaworld \
  python -u train_sac.py \
  --config configs/sac/kuka_push_v3_reward_v3_1_lr3e4_200k.yaml
```

Monitor the preferred run with:

```bash
conda run --no-capture-output -n metaworld \
  tensorboard --logdir runs/reward_ablations/kuka-push-v3/reward_v3_1_lr3e4/seed_0_200k/tensorboard
```
