# Three-task four-layer SAC training

This experiment independently trains fresh SAC policies for `reach-v3`,
`hammer-v3`, and `handle-press-side-v3`. Each run uses seed 0 and exactly
1,000,000 environment steps. Early stopping is disabled, so validation success
never shortens training.

The actor and both critics use `[256, 256, 256, 256]`. The actor has LayerNorm
after its first Linear layer. All tasks share the same SAC hyperparameters from
`configs/sac_4layer_3task_1m.yaml`, use CPU with one PyTorch thread, and start
from random initialization without policy transfer.

Validation runs every 20,000 training steps using 50 deterministic episodes
from fixed task bank seed 10000. Best-success selection compares
`(success_rate, mean_return)` lexicographically: success rate first, then mean
return; an exact tie retains the earliest checkpoint. Best-return selection
uses mean return alone for metric reporting, but does not create a separate
best-return checkpoint.

The final test loads only the best-success checkpoint and evaluates two
separate fixed banks (seeds 20000 and 20001), 50 episodes each. Training,
validation, and final-test task banks are checked for pairwise disjointness.
Test results never participate in checkpoint selection.

To minimize disk usage, each task/seed directory retains only two model
archives:

- `best_model.zip`: atomically replaced only when the validation
  `(success_rate, mean_return)` score strictly improves.
- `latest_model.zip`: atomically replaced every 50,000 steps and once more at
  the exact 1,000,000-step boundary.

No periodic history, best-return archive, final-model duplicate, or replay
buffer is written. The in-memory replay capacity remains 1,000,000 transitions.
Every evaluation remains in `evaluations/evaluations.npz`,
`evaluations/evaluation_history.csv`, TensorBoard, and the final summary.

## Run all tasks

```bash
conda run -n metaworld python -m scripts.run_sac_architecture_benchmark \
  --config configs/sac_4layer_3task_1m.yaml --all --seed 0
```

## Run one task

```bash
conda run -n metaworld python -m scripts.run_sac_architecture_benchmark \
  --config configs/sac_4layer_3task_1m.yaml --task reach-v3 --seed 0
```

Replace `reach-v3` with `hammer-v3` or `handle-press-side-v3` as needed.

## Interruption and inference

`latest_model.zip` is a valid inference checkpoint, but it does not include the
in-memory replay buffer and therefore is not a complete SAC training snapshot.
The runner rejects `--resume` for this experiment instead of silently
continuing with an empty replay buffer. An interrupted training run may need to
restart from scratch in a new or deliberately cleared output directory.

## TensorBoard

```bash
tensorboard --logdir runs/sac_4layer_3task_1m --port 6006
```

## Aggregate completed runs

```bash
conda run -n metaworld python -m scripts.aggregate_sac_architecture_benchmark \
  --root runs/sac_4layer_3task_1m \
  --tasks reach-v3 hammer-v3 handle-press-side-v3 \
  --seeds 0 \
  --json runs/sac_4layer_3task_1m_summary.json \
  --csv runs/sac_4layer_3task_1m_summary.csv
```

New runs refuse to overwrite an existing task/seed directory. Existing
incomplete directories are preserved for inspection and must be handled
manually before restarting.
