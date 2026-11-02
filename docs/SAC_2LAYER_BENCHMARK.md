# Ten-task two-layer SAC benchmark

This benchmark trains independent, randomly initialized SAC agents on ten
MetaWorld v3 tasks. It is isolated from expert, DiffCRL, and forward-transfer
runs under `runs/sac_2layer_benchmark/<task>/seed_<seed>/`.

The resolved configuration is in `configs/sac_2layer_benchmark.yaml`. The actor
is `TransferableSACPolicy` with a 39-256-LayerNorm-ReLU-256-ReLU trunk and
independent mean/log-standard-deviation heads. No GeneralPolicy or expert
parameters are loaded. SB3 supplies ordinary twin 256-256 critics.

## Protocol

- Training samples the task variation exposed by the existing MetaWorld v3
  factory. Its MT1 bank is seeded by the training seed.
- Validation uses one fixed 50-configuration bank (bank seed 10000) and reset
  seeds 10000-10049 every 20,000 environment steps.
- Training stops at three consecutive validation success rates of at least
  0.95, but never before step 100,000. The hard limit is step 1,000,000.
- Final testing evaluates the selected best-success model once on two disjoint
  50-configuration banks (bank seeds 20000 and 20001), with reset seeds
  30000-30099. This forms a 100-episode test without exceeding the MT1 bank
  size.
- The runner verifies that training, validation, and both final-test banks are
  pairwise disjoint and records their ordered hashes in `metadata.json`.
- Best-success (success first, mean-return tie break), best-return, periodic,
  final-model, and replay-buffer artifacts are retained. The qualifying best
  checkpoint is saved before an early-stop decision takes effect.

Fresh runs refuse any existing run directory. Resume accepts only an incomplete
directory containing `RUNNING`, an identical resolved configuration and
metadata, and a paired periodic model/replay-buffer checkpoint. Evaluation
history is truncated to that checkpoint before continuing, so an evaluation
newer than the latest durable checkpoint is not duplicated.

## Commands

Run from the repository root in the `metaworld` Conda environment.

Single task and seed:

```bash
conda run -n metaworld python -m scripts.run_sac_2layer_benchmark \
  --config configs/sac_2layer_benchmark.yaml \
  --task hammer-v3 --seed 0
```

All ten tasks sequentially for one seed:

```bash
conda run -n metaworld python -m scripts.run_sac_2layer_benchmark \
  --config configs/sac_2layer_benchmark.yaml --all --seeds 0
```

The full 30-run matrix sequentially:

```bash
conda run -n metaworld python -m scripts.run_sac_2layer_benchmark \
  --config configs/sac_2layer_benchmark.yaml --all --seeds 0 1 2
```

Start the full matrix in tmux:

```bash
tmux new -s sac-2layer
conda activate metaworld
python -m scripts.run_sac_2layer_benchmark \
  --config configs/sac_2layer_benchmark.yaml --all --seeds 0 1 2
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t sac-2layer`.

Monitor TensorBoard:

```bash
conda run -n metaworld tensorboard \
  --logdir runs/sac_2layer_benchmark --port 6006
```

Resume exactly one interrupted run from its latest paired periodic checkpoint:

```bash
conda run -n metaworld python -m scripts.run_sac_2layer_benchmark \
  --config configs/sac_2layer_benchmark.yaml \
  --task hammer-v3 --seed 0 --resume
```

Aggregate only after every requested task/seed has `RUN_COMPLETE`:

```bash
conda run -n metaworld python -m scripts.aggregate_sac_2layer_benchmark \
  --root runs/sac_2layer_benchmark \
  --tasks reach-v3 push-v3 hammer-v3 handle-press-side-v3 \
    button-press-topdown-v3 faucet-open-v3 drawer-close-v3 window-close-v3 \
    door-open-v3 peg-unplug-side-v3 \
  --seeds 0 1 2 \
  --json runs/sac_2layer_benchmark_summary.json \
  --csv runs/sac_2layer_benchmark_summary.csv
```

Aggregation is read-only with respect to run directories and refuses to replace
either summary output.

## Limitations

- A resume point exists only after the first 50,000-step periodic checkpoint.
  An interruption earlier than that requires preserving the incomplete run for
  diagnosis and choosing a new output root; the runner will not overwrite it.
- The final test is intentionally performed only after training. A run stopped
  during finalization can resume without another training step only when the
  durable paired checkpoint is exactly at the one-million-step budget;
  otherwise it resumes from that earlier checkpoint.
- The infrastructure and mocked tests do not establish that a task is learnable
  or that 0.95 success will be reached. Those are benchmark outcomes.
