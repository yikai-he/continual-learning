# Immediate SAC Forward Transfer Experiment

This standalone experiment compares random and GeneralPolicy-transferred
initialization of the same LayerNorm SAC actor on Hammer. It does not modify or
invoke DiffCRL. The source checkpoint is the formal GeneralPolicy after Faucet,
before Hammer.

## Conditions

- Random: construct `TransferableSACPolicy` and retain its random actor.
- Transferred: construct the identical SAC object, then copy only the actor mean
  pathway from the validated GeneralPolicy.

Both conditions retain fresh critics, targets, replay, entropy state, log-std
head, and optimizer states. Each uses 100,000 steps, evaluations at step zero
and every 10,000 steps, 50 deterministic episodes from task-set seed 10000,
and paired training seeds 0, 1, and 2.

## Run one condition

```bash
conda run -n metaworld python -m scripts.run_forward_transfer \
  --config configs/forward_transfer/hammer_random.yaml \
  --seed 0
```

```bash
conda run -n metaworld python -m scripts.run_forward_transfer \
  --config configs/forward_transfer/hammer_transferred.yaml \
  --seed 0
```

## Run all paired seeds sequentially

```bash
for seed in 0 1 2; do
  conda run -n metaworld python -m scripts.run_forward_transfer \
    --config configs/forward_transfer/hammer_random.yaml \
    --seed "$seed"
  conda run -n metaworld python -m scripts.run_forward_transfer \
    --config configs/forward_transfer/hammer_transferred.yaml \
    --seed "$seed"
done
```

Every target directory must be absent before a new run. The runner refuses to
overwrite it.

## Resume an interrupted run

Use the same condition, seed, and configuration with `--resume`:

```bash
conda run -n metaworld python -m scripts.run_forward_transfer \
  --config configs/forward_transfer/hammer_transferred.yaml \
  --seed 0 \
  --resume
```

Resume requires `RUNNING`, rejects `RUN_COMPLETE`, validates the saved config
and source provenance, and loads the latest model/replay-buffer pair. Evaluation
history is restored before training continues.

## Aggregate completed results

```bash
conda run -n metaworld python -m scripts.aggregate_forward_transfer \
  --root runs/forward_transfer \
  --seeds 0 1 2 \
  --output runs/forward_transfer/hammer-v3/aggregate.json
```

The output path must not already exist. The aggregator requires all six
`RUN_COMPLETE` markers and reads saved summaries only; it does not evaluate or
train policies.

Normalized success AUC is trapezoidal area divided by the step range. Threshold
time is the first measured evaluation step at or above 80%; it remains null if
never observed. Forward transfer is transferred AUC minus random AUC, paired by
training seed.
