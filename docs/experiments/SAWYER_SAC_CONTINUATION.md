---
status: prepared-not-started
last_verified: 2026-11-16
applies_to: continual-learning
---

# Sawyer SAC continuation experiments

These are four independent single-task continuations, ordered by current
standardized success as an approximate easy-to-hard execution schedule. This is
not an intrinsic task-difficulty claim and is not a continual-learning sequence.

| Order | Task | Source success | Maximum additional steps | Source |
|---:|---|---:|---:|---|
| 1 | Peg Unplug Side | 82% | 300,000 | `experts/sawyer/peg-unplug-side-v3/seed_0/best_model.zip` |
| 2 | Stick Pull | 50% | 500,000 | `experts/sawyer/stick-pull-v3/seed_0/best_model.zip` |
| 3 | Push | 30% | 1,000,000 | `experts/sawyer/push-v3/seed_0/best_model.zip` |
| 4 | Push Wall | 28% | 1,000,000 | `experts/sawyer/push-wall-v3/seed_0/best_model.zip` |

Shelf Place is excluded because its selected archive is the untrained
`num_timesteps=0` policy and both its success and return are zero. It needs a
separate environment/reward investigation.

## Protocol

Each run restores its own full SB3 model archive and starts with an independent
empty replay buffer. A 10,000-step stochastic-policy refill populates that
buffer without optimizer updates. Refill interactions count toward both the
additional-step hard budget and the 50,000-step early-stopping minimum.

Validation uses 50 deterministic episodes from fixed task bank 10000 every
20,000 global environment steps. Three consecutive validations at 100% success
stop training after the minimum; a failure resets the counter. Plateau stopping
is disabled. Selection remains validation success first and mean return second,
with the initial source policy evaluated before updates. Independent final
testing uses 100 episodes from task bank 20000 and cannot affect selection.

Model-state continuation restores network, optimizer, entropy, and counter
state. Ordinary SB3 model archives contain no replay buffer or live RNG stream;
the buffer is refilled and the RNG is reseeded with run seed 0. The manifest,
checkpoint hash, restored/reset components, refill result, source baseline,
final policy, selected best policy, additional steps and updates, stopping
reason, and final-bank result are recorded. Improved experts are not registered
automatically.

Outputs will be created only by training:

```text
runs/sac_continuation/sawyer/<task>/seed_0/
```

Inspect `results.json` for source/final/selected distinctions,
`evaluations/evaluations.npz` for validation history, `run_manifest.json` and
`RUN_COMPLETE` for lifecycle state, and `console.log` for the captured command
output.

## Commands

Activate the repository's `metaworld` Conda environment and run the safe
preflight:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  bash tools/experiments/train_sawyer_continuation.sh --dry-run
```

Training is deliberately guarded behind an explicit command:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  bash tools/experiments/train_sawyer_continuation.sh --run
```

The launcher runs sequentially, stops on the first failure, refuses existing
output paths, and captures one console log per run.

**The experiment configurations and launcher have been prepared, but the actual
training runs have not been started.**
