---
status: current
last_verified: 2026-11-07
applies_to: continual-learning
---

# Research Code Standards

## Purpose and scope

These project-specific standards guide work on the thesis implementation of SAC, sequential SAC, DiffCRL, behavior cloning, DDPM replay, Flow Matching replay, and continual robotic manipulation. The goal is trustworthy thesis evidence from a small, understandable codebase. The repository is not intended to provide a general reinforcement-learning framework or anticipate unrelated future projects.

The policy draws on established research-software-engineering ideas, including traceable computational results, explicit dependencies, versioned inputs, reviewable code, and proportionate testing. Relevant references include [Good Enough Practices in Scientific Computing](https://doi.org/10.1371/journal.pcbi.1005510), the [FAIR Principles for Research Software](https://doi.org/10.15497/RDA00068), and guidance from the [Software Sustainability Institute](https://www.software.ac.uk/). These sources inform this repository; the rules below are local engineering choices, not universal standards.

## Priority order

When priorities conflict, use this order:

1. Scientific correctness.
2. Experimental reproducibility.
3. Code clarity.
4. Minimalism.
5. Appropriate modularity.
6. Focused testing.
7. Maintainability.

For example, a small amount of explicit duplication can be preferable to an abstraction that obscures experimental differences. Conversely, short code is not acceptable if it weakens validation or provenance.

## Quality criteria

### Scientific correctness

- Treat rewards, success definitions, termination, task sampling, horizons, trajectory transformations, replay allocation, training objectives, evaluation banks, and metric formulas as scientific behavior.
- State assumptions and units where ambiguity could change an interpretation.
- Reject invalid shapes, nonfinite values, incompatible checkpoints, unknown configuration keys, and inconsistent provenance at boundaries.
- Do not change a scientific default as a cleanup. A deliberate change needs an explicit rationale, focused validation, and documentation of its effect on comparability.
- Never make a result look consistent by rewriting historical measurements or weakening acceptance criteria.

### Experimental reproducibility

- Record resolved configuration, seeds, task order, environment/reward version, model identity, checkpoint provenance, and evaluation protocol when the workflow supports them.
- Keep output directories collision-safe and checkpoint/resume behavior explicit.
- Preserve immutable historical artifacts and associate results with the implementation and configuration that produced them.
- Distinguish deterministic seeding from guaranteed numerical identity across devices and dependency versions.
- When comparing methods, match budgets, task order, experts, seeds, and evaluation protocol unless the difference is the controlled variable.

### Clarity and minimalism

- Prefer direct names, typed data structures, explicit control flow, and short module-level contracts over clever indirection.
- Add a module or abstraction only when it removes demonstrated duplication or protects a meaningful invariant across active workflows.
- Keep public APIs limited to currently supported use cases.
- Remove dead branches and obsolete compatibility plumbing after verifying that active workflows do not use them.
- Avoid speculative abstractions, generic registries, extension systems, and configuration switches without a current experiment requiring them.

### Diagnostics and observability

- Keep diagnostics that are required to validate scientific contracts, provenance, failure recovery, or core evaluation.
- Prefer existing reports, assertions, and tests before adding a new diagnostic pipeline.
- Do not add persistent telemetry, task-specific analysis modules, plots, or output files merely because they might be useful later.
- Temporary investigation code should remain local or be removed after it answers the question unless it becomes an essential reproducibility check.

### Configuration discipline

- Reuse a maintained configuration when an override expresses the intended run clearly.
- Add a YAML file only for a named, reproducible workflow or controlled experimental condition.
- Avoid matrices of near-identical configurations, redundant aliases, and options that expose a value intended to remain fixed.
- Archive completed study configurations with their results when they are no longer active defaults; do not present them as current guidance.

### Focused testing

- Test observable contracts, numerical transformations, deterministic behavior where promised, validation failures, checkpoint compatibility, and resume boundaries.
- Add regression tests for plausible failures introduced by the change, not for every implementation line.
- Prefer a small pure unit test over an environment-heavy integration test when both verify the same contract.
- Use integration tests for wiring that cannot be established at a lower level, including environment adapters and persisted artifact boundaries.
- Avoid redundant tests, duplicated fixtures, broad characterization snapshots without a stated invariant, and tests tied to incidental implementation details.
- Do not delete or weaken unrelated coverage to make a change pass.

### Documentation

- Update existing canonical documents before creating another Markdown file.
- Document current contracts and commands; archive dated experiments and superseded designs without silently rewriting them.
- Every maintained Markdown document must contain `status`, `last_verified`, and `applies_to` metadata. Refresh verification dates only after checking the claims.
- Keep explanations near the canonical source and link to them instead of duplicating README or architecture material.
- Avoid excessive documentation for self-evident helpers, temporary investigations, or removed features.

## Proportional acceptance checks

Choose checks according to the risk of the change. The examples below are expectations, not claims about the current worktree.

| Change | Minimum relevant checks |
|---|---|
| Documentation only | Resolve maintained Markdown links; check referenced commands or paths; run `git diff --check` |
| Pure helper or validation | Focused unit tests; affected import or compilation check; `git diff --check` |
| Configuration schema/default | Schema and loading tests; representative active configurations; unknown-key and invalid-value cases |
| Environment adapter or reward semantics | Focused contract tests; deterministic state/action comparison where practical; environment reset and step for available backends |
| Replay transformation or generator sampling | Shape/finiteness and round-trip tests; seeded determinism where promised; provenance/checkpoint compatibility; affected replay tests |
| Training, evaluation, checkpoint, or resume | Focused unit and integration tests; configuration load; small smoke path when feasible; artifact and resume invariants |

Do not launch expensive training solely as a routine quality check. If a dependency, device, checkpoint, or environment is unavailable, record the limitation and run the strongest safe alternative.

## Change acceptance checklist

Before declaring work complete, confirm only the items relevant to the change and report their actual outcomes:

- The requested scientific behavior is explicit and unchanged behavior remains unchanged.
- Seeds, configurations, provenance, checkpoint compatibility, and historical evidence remain coherent.
- The solution is no more general or configurable than the active research question requires.
- New modules, abstractions, diagnostics, tests, configurations, and documents each have a concrete current purpose.
- Focused tests cover the changed contract and meaningful failure modes without duplicating unrelated coverage.
- Canonical documentation and metadata reflect any user-visible behavior change.
- Unrelated worktree changes and protected experiment artifacts are preserved.
- Commands actually executed, along with failures and skips, are reported accurately.
- `git diff --check` is clean.

Passing this checklist does not establish scientific validity by itself. Thesis claims still require an appropriate experimental design, qualified inputs, matched comparisons, and honest reporting of uncertainty.
