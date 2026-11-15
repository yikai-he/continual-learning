---
status: historical
last_verified: 2026-11-12
applies_to: continual-learning before the stochastic GeneralPolicy migration
---

# Historical deterministic GeneralPolicy

This document preserves the deterministic `GeneralPolicy` contract that was
active before 2026-11-12. It describes repository implementation facts first
and historical evidence separately. The active implementation is now a
Squashed-Gaussian policy; historical `general_policy_v2` checkpoints and
completed experiment artifacts were not rewritten.

## Architecture

The policy consumed the raw 39-dimensional environment observation. With
`bc.task_conditioning: one_hot`, it appended one one-hot task vector ordered by
the complete configured task sequence; otherwise its input stayed 39D. It used
four fully connected hidden layers of width 256. The first linear layer was
followed by `LayerNorm(256)` and ReLU; each remaining hidden linear layer was
followed by ReLU. There was no observation z-score normalization.

A final `Linear(256, 4)` deterministic action head and `tanh` produced a
normalized action. The normalized action was mapped to physical bounds as

```text
low + (tanh(z) + 1) / 2 * (high - low).
```

Consequently, inference was deterministic and bounded. `act(...,
deterministic=False)` intentionally returned the same value as deterministic
inference; there was no learned variance or sampling distribution.

## Training

Two configured MSE objectives existed:

- `post-tanh-mse` minimized MSE between physical bounded predictions and the
  physical action targets.
- `pre-tanh-mse` normalized physical targets to `[-1, 1]`, clipped them to
  `[-1 + 1e-4, 1 - 1e-4]`, applied inverse tanh, and minimized MSE against the
  unsquashed output of the deterministic head.

The default was `post-tanh-mse`. Dataset actions outside environment bounds
were rejected unless the caller explicitly enabled the generated-replay
diagnostic path. In that path, post-tanh targets remained unchanged even
though the bounded policy could not reproduce them; pre-tanh targets used the
same inverse-tanh clipping rule.

Training flattened already split whole episodes into observation/action
timesteps. Optional task IDs were carried as a third tensor. A seeded
`DataLoader` controlled minibatch order. Adam was used, gradients were not
clipped, and each epoch recorded objective loss plus physical-action MSE on
the complete training and validation tensors. Checkpoint selection used the
lowest validation physical-action MSE, even for the pre-tanh objective, and
restored both model and optimizer states from that epoch.

Policy parameters persisted across DiffCRL stages. The active trainer created
a fresh Adam optimizer for each stage; optimizer continuity existed only
within a stage and in an individual saved continuation payload. The loader
could restore an explicitly saved Adam state and verified that it owned every
policy parameter exactly once.

The later, now-removed stochastic GeneralPolicy-to-SAC transfer experiment is
documented separately in
`docs/archive/SAC_POLICY_TRANSFER_IMPLEMENTATION.md`.

## Checkpoints and experiments

The deterministic checkpoint identity was `general_policy_v2`. Its payload
stored explicit architecture metadata, hidden sizes, LayerNorm use, action
bounds in the state dictionary, all model parameters, optional one-hot task
ordering, optional Adam state with parameter names, and caller-provided stage
metadata. The loader accepted only four 256-wide hidden layers with first-layer
LayerNorm and rejected older implicit architectures.

The stochastic active loader rejects `general_policy_v2` with an explicit
historical-format error. No deterministic checkpoint is automatically
converted because a variance head and its scientific semantics cannot be
recovered from deterministic weights.

Archived configurations and completed reports record their original
`post-tanh-mse` or `pre-tanh-mse` interpretation. They remain historical
provenance and are not valid active stochastic configurations without an
explicit new experiment design. Archived validation and pilot results are
evidence about those particular deterministic runs; this document makes no
additional performance claim.

## Relationship to DISTR

The repository's deterministic expert-driven pipeline was inspired by
[DISTR](https://arxiv.org/abs/2411.10809)'s
trajectory-replay and general-policy separation, but it was not an exact DISTR
implementation. In particular, frozen independently trained experts supplied
current-task data, online Immediate SAC training was not part of DiffCRL, and
the deterministic MSE GeneralPolicy did not express a stochastic policy
likelihood. The current stochastic migration narrows the last difference only;
it does not remove the others.
