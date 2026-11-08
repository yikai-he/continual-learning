---
status: current
last_verified: 2026-11-07
applies_to: continual-learning
---

# Agent guidelines

This is focused bachelor’s-thesis research software, not a general-purpose RL framework. Apply priorities in this order:

1. Scientific correctness.
2. Experimental reproducibility.
3. Code clarity.
4. Minimalism.
5. Appropriate modularity.
6. Focused testing.
7. Maintainability.

Before changing anything, inspect Git status and preserve unrelated work. Do not modify or delete runs, experiments, datasets, checkpoints, trained models, or archived evidence unless the user explicitly scopes that operation.

For implementation work:

- Preserve reward, success, termination, sampling, task-order, seed, evaluation, checkpoint, and resume semantics unless the requested scientific change requires otherwise.
- Prefer the smallest explicit solution inside the current architecture. Do not build speculative frameworks, plugin systems, compatibility layers, or abstractions for hypothetical reuse.
- Avoid new diagnostics, configuration options, tests, and documents unless each has a concrete current use that existing facilities cannot cover.
- Keep scientific configuration strict and recorded. Never silently change defaults or reinterpret historical results.
- Test the changed contract and important failure modes. Do not duplicate assertions already covered at a lower level or weaken unrelated tests.
- Run validation proportional to risk and report exactly what ran, passed, skipped, or could not run. Never claim unexecuted checks passed.
- Update canonical documentation when behavior changes. Maintained Markdown needs verified metadata; preserve archived records verbatim.

Use [Research Code Standards](docs/RESEARCH_CODE_STANDARDS.md) for detailed acceptance criteria.
