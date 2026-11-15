---
status: current
last_verified: 2026-11-11
applies_to: continual-learning
---

# Test suite

Run the default, simulator-free suite from the repository root:

```bash
conda run --no-capture-output -n metaworld \
  env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/ -q
```

Disabling plugin autoload prevents unrelated system or ROS pytest plugins from
interfering with collection.
The `metaworld` environment supplies the project runtime dependencies; disabling
autoload also avoids ROS 2 Jazzy pytest-plugin conflicts from the host environment.
The default configuration excludes tests marked `integration` because repeatedly
creating MuJoCo environments in one Python process can retain substantial native
memory even after each environment is closed.

Run a required integration test explicitly and in isolation so the operating
system reclaims simulator memory when that process exits:

```bash
python -m pytest -q -m integration \
  tests/test_kuka_v3_sac.py::test_kuka_v3_factory_contract
```

The suite covers configuration schemas, environment contracts, SAC checkpoints
and persistence, trajectory collection, GeneralPolicy behavior cloning, stage data,
DDPM and Flow Matching replay, Real Replay, replay allocation, evaluation and
reporting, DiffCRL artifacts, and continual-learning metrics.
Test filenames follow those responsibilities and should be treated as the
authoritative inventory rather than maintaining a duplicate list here.

Optional KUKA, MetaWorld, CUDA, or artifact-dependent integration tests require
their local resources. Do not run all simulator integrations in one process.

Verified direct-43D test counts and end-to-end DDPM/Flow smoke commands are
recorded once in [Direct 43D migration](../docs/DIRECT_43D_MIGRATION.md).
