---
status: current
last_verified: 2026-11-11
applies_to: continual-learning
---

# Documentation archive

This directory retains dated design notes, audits, cleanup records, and experiment reports that remain useful for provenance but do not describe the current repository reliably.

Archived commands, paths, configuration fields, and numerical assumptions may no longer work. Use [the project README](../../README.md), [current architecture](../ARCHITECTURE.md), [development status](../DEVELOPMENT_STATUS.md), and [workflow index](../WORKFLOW_INDEX.md) for maintained guidance.

Archive metadata uses these meanings:

- `original_date` is supported by the document itself, version history, or experiment metadata; otherwise it is `unknown`.
- `archived_on` records when the document was designated or moved into this archive.
- `last_verified: unknown` means the historical claims were intentionally preserved rather than rewritten against current code.

Files below `experiments/records/` are preserved experiment outputs and metadata. They are protected research records and are not maintained documentation; their original contents are left unchanged.

The removed generator data contract is documented in
[Legacy 22D Representation](LEGACY_22D_REPRESENTATION.md). It is historical and
must not be treated as a supported configuration or checkpoint-loading guide.

The removed GeneralPolicy-to-SAC initialization experiment is documented in
[SAC Policy Transfer Implementation](SAC_POLICY_TRANSFER_IMPLEMENTATION.md).
Its experiment-specific AUC comparison is distinct from the maintained
ContinualManipulation Forward Transfer metric.
