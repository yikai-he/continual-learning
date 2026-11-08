---
status: current
last_verified: 2026-11-07
applies_to: continual-learning
---

# Test suite

Run the full suite from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Disabling plugin autoload prevents unrelated system or ROS pytest plugins from
interfering with collection.

The suite covers configuration schemas, environment contracts, SAC checkpoints
and resume, trajectory collection, GeneralPolicy behavior cloning, stage data,
DDPM and Flow Matching replay, Real Replay, replay allocation, evaluation and
reporting, DiffCRL artifacts/resume, sequential workflows, and forward transfer.
Test filenames follow those responsibilities and should be treated as the
authoritative inventory rather than maintaining a duplicate list here.

Optional KUKA, CUDA, or other resource-dependent integration tests may be
skipped when the required local resources are unavailable.
