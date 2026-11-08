---
status: current
last_verified: 2026-11-07
applies_to: continual-learning
---

# Canonical SAC Training

Use `train_sac.py` for new single-task MetaWorld-v3 and KUKA-v3 SAC runs.

## Default checkpoint semantics

The default `sac.checkpoint_mode: low_disk` retains two inference checkpoints:

```text
best_model.zip    # highest validation (success_rate, mean_return)
latest_model.zip  # most recently persisted policy state
```

Success rate is primary. Mean return breaks equal-success ties, and exact ties
keep the earlier checkpoint. Metrics must be finite. Files are serialized to a
temporary path and atomically replaced; replay buffers are not saved. Therefore
`latest_model.zip` is not an exact-resume artifact.

Set `sac.checkpoint_mode: full` for the legacy replay-backed layout, periodic
model/replay pairs, and `--resume-from`. Existing checkpoints remain loadable.

## Evaluation separation

- Optimization: `runtime.seed`
- Validation/selection: `evaluation.task_set_seed` (default `10000`)
- Final test: `evaluation.final_task_set_seeds` (default `[20000, 20001]`)

Serialized configuration identities are checked pairwise for overlap. Final
tests run only after selection and cannot change checkpoint ranking.

## Migration

Existing YAML files remain valid because omitted fields receive defaults. To
preserve the former standard trainer behavior, add:

```yaml
sac:
  checkpoint_mode: full
```

Historical `best_model.zip` files may be return-best rather than success-first.

## Manual engineering smoke test

The smoke configuration exercises environment creation, training, four periodic
validation events, atomic best/latest replacement, independent final testing,
and metadata serialization. It is an infrastructure check, not a performance
benchmark, and does not require the policy to solve the task.

From the repository root, with the `metaworld` Conda environment active, run:

```bash
python train_sac.py \
  --config configs/sac/smoke_low_disk.yaml \
  --run-name manual_YYYYMMDD_HHMMSS
```

Replace the timestamp placeholder so the run has a new path under
`runs/smoke/sac_low_disk/`. The trainer also refuses to overwrite an existing
run directory. A successful run contains `best_model.zip`, `latest_model.zip`,
`config.yaml`, `config.json`, `run_manifest.json`, evaluation records, and
checkpoint manifests. `config.json` must report success-first selection and
final-test results. No replay-buffer file or periodic checkpoint directory is
expected.

Stop safely with `Ctrl-C`. An interrupted run is intentionally not marked
complete; use a fresh run name for the next attempt because low-disk checkpoints
are inference-only and cannot provide exact training resumption.
