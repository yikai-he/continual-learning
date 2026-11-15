---
status: prepared
last_verified: 2026-11-10
---

# KUKA four-stage 200+200 DDPM versus Flow Matching protocol

This preparation defines one seed-0 comparison over Hammer → Handle Press Side
→ Button Press → Reach. DDPM uses its existing 100 reverse steps. Flow Matching
uses the existing Euler solver with 10 fixed steps. These step counts are not
computationally or algorithmically equivalent.

All arms use the same KUKA-v3 SAC experts, reward version v2, raw actions, no
generated-action projection, unconditioned GeneralPolicy, 100 generator epochs,
100 BC epochs, fixed evaluation bank seed 10000, and 20 evaluation episodes.
The deterministic collector uses episode reset seeds 0–199 and the same task
bank in every arm. Whole-episode split seeds are also identical, giving 160
current training and 40 validation trajectories. Historical replay is
training-only, producing 160/360/360/360 training trajectories and 40 current
validation trajectories at Stages 0/1/2/3.

The unchanged fixed-total allocator gives Stage 1 `200`, Stage 2 `100+100`, and
Stage 3 `67+67+66` in historical task order. No reusable, formally validated
200-episode KUKA collection exists for all four requested tasks. Existing
20-episode stage artifacts are not reused or duplicated. On manual execution,
the production deterministic collector will collect 200 episodes per current
task independently in each arm; prior paired runs confirm this seed-based path
produces identical current tensors, and the analyzer verifies hashes afterward.

## Resource considerations

A physical 200-trajectory group `(200,200,43)` is about 6.9 MB in float32.
Stage-1–3 packed training data `(360,200,43)` is about 12.4 MB. Encoded 22D
and 43D generator tensors are about 6.3 MB and 12.4 MB respectively. BC uses
32,000 samples at Stage 0 and 72,000 later; its flattened float32 observation
plus action tensors are about 5.5 MB and 12.4 MB. Normalization creates
additional tensors of comparable encoded size.

Replay sampling currently materializes each task allocation in one batch. The
largest single decoded batch is 200 trajectories (6.9 MB); temporal-network
activations with width 64 are materially larger but sampling is under
`no_grad`. This should be manageable on ordinary CPU RAM and typical GPUs, but
43D generation and a 200-sample batch can create transient activation peaks.
No unnecessary all-stage trajectory archive is held in memory. Checkpoints are
architecture-dependent rather than dataset-dependent and historical examples
are approximately 0.69 MB.

The number of generator minibatches per epoch is roughly ten times the 20+20
condition because batch size remains eight. Collection and BC examples also
increase tenfold. Wall time need not scale exactly tenfold because environment,
checkpoint, evaluation, and Python overhead are fixed. DDPM samples ten times
as many trajectories with the same 100 steps. FM combines ten times the replay
count with one tenth the Euler steps, so neural-work scaling differs sharply
from DDPM; measured end-to-end time must be taken from stage reports.

## Commands

Validate and print all four commands without creating outputs:

```bash
bash scripts/experiments/run_kuka_generator_200plus200.sh --dry-run
```

Run all four manually:

```bash
bash scripts/experiments/run_kuka_generator_200plus200.sh
```

Run or resume a single arm:

```bash
bash scripts/experiments/run_kuka_generator_200plus200.sh --method fm_structured_22d
bash scripts/experiments/run_kuka_generator_200plus200.sh --method fm_structured_22d --resume
```

After completion, analyze without mutation:

```bash
conda run -n metaworld python scripts/evaluation/analyze_kuka_generator_200plus200.py
```

Outputs are collision-protected under
`runs/generator_comparison/four_stage_200plus200/<method>_seed0/`. Resume uses
the production hash-validated stage-boundary and configuration checks. The
analyzer reports matrices, AP, forgetting, returns, task retention, structural
replay diagnostics, stage timings, four requested contrasts, and a explicitly
single-seed descriptive representation-by-generator interaction. Expert-action
disagreement is not inferred; it requires a separately saved diagnostic.
