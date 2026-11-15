---
status: current
last_verified: 2026-11-11
applies_to: continual-learning
---

# Development Status

This document classifies repository capabilities by implementation and evidence
status. “Implemented” means an executable code path exists. “Validated” means
the repository also contains focused automated tests or retained provenance;
it does not imply scientific superiority or real-world validation.

## Implemented and routinely tested

- Strict YAML configuration loading and experiment-specific validation.
- MetaWorld-v3 environment construction, 39D observation/4D action contracts,
  task switching, episode success accumulation, and fixed task banks.
- Single-task SAC training, fixed-bank evaluation, checkpoint manifests, and
  success-based checkpoint qualification.
- Expert trajectory collection with episode, task, goal, and provenance checks.
- Persistent four-layer Squashed-Gaussian `GeneralPolicy` trained by NLL behavior cloning.
- DiffCRL modes: No Replay, Real Replay, DDPM Diffusion Replay, and conditional
  Flow Matching Replay.
- Direct 43D DDPM and Flow Matching generation with external task/goal
  conditioning, 43-feature normalization, structural diagnostics, and strict
  rejection of legacy generator checkpoints.
- Whole-episode splitting, per-feature generator normalization, replay
  allocation, checkpoint persistence, authenticated completion markers, and
  validated stage synchronization.
- Continual evaluation matrices, Average Performance, and end-of-task
  forgetting.

## Implemented with backend or resource constraints

- KUKA-v3 single-task SAC, expert collection, DiffCRL, and fixed-task
  evaluation require the editable sibling `metaworld-kuka-v3` checkout.
- Device-native DDPM sampling is CPU-equivalence-tested. CUDA execution
  and performance have not been validated on a CUDA-capable test host.

## Experimental workflows

- Flow Matching replay and fixed-total replay-budget comparisons.

Experimental means the code/configuration exists; consult the relevant protocol
and retained artifacts before treating a result as thesis evidence.

## Known limitations

- The training/evaluation pipeline is simulation-only; there is no real-robot
  deployment interface in this repository.
- DiffCRL uses pretrained frozen SAC experts and BC. It does not reproduce
  DISTR’s online Immediate Policy training process.
- Forward transfer is computed only from explicitly aligned learning/reference
  curves. Endpoint matrices serialize it as `null`.
- Formal trajectory packing requires full-horizon episodes.
- Generator optimizers are recreated at each stage; model weights persist.
- Deterministic seeds improve repeatability, but exact equivalence is not
  promised across devices, CUDA versions, or GPU models.
- Expert provenance is weaker for legacy checkpoints without manifest sidecars.

## Open research questions

- How do DDPM, Flow Matching, Real Replay, and No Replay compare under matched
  multi-seed budgets and task orders?
- Which replay allocation strategy best preserves difficult or dissimilar
  tasks as the curriculum grows?
- How sensitive are retention and plasticity to expert quality, trajectory
  count, generator capacity, and task ordering?
- How much of the approach transfers from simulation to a physical KUKA setup,
  including observation calibration, action safety, and success detection?

## Prioritized future work

1. Complete matched multi-seed thesis experiments and report uncertainty without
   combining smoke or incomplete runs.
2. Qualify every formal expert with manifests and fixed-task evidence.
3. Validate Flow Matching and device-native DDPM sampling on the intended compute
   platform.
4. Define and test the real-robot observation/action/safety boundary in the
   separate deployment project before claiming real-world readiness.

For implementation details see [Architecture](ARCHITECTURE.md), for migration
and acceptance evidence see [Direct 43D migration](DIRECT_43D_MIGRATION.md), and
for workflow classification see [Workflow index](WORKFLOW_INDEX.md).
