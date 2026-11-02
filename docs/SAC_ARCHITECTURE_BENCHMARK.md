# Ten-task four-layer SAC benchmark

The active SAC benchmark uses four hidden layers of width 256 in the actor and
both critics. The actor applies LayerNorm after its first Linear layer. No
alternative policy-depth compatibility path is supported.

The protocol is fixed at a maximum of 1,000,000 steps, validation every 20,000
steps over 50 episodes, and early stopping after three consecutive validation
success rates of at least 95%, no earlier than 100,000 steps. The final test is
an independent 100 episodes.

```bash
conda run -n metaworld python -m scripts.run_sac_architecture_benchmark \
  --config configs/sac_4layer_benchmark.yaml --task reach-v3 --seed 0
```

Results are written beneath `runs/sac_4layer_benchmark/`. Existing historical
benchmark output directories are not read, converted, overwritten, or deleted.
