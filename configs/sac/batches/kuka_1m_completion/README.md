---
status: current
last_verified: 2026-11-07
applies_to: continual-learning
---

# KUKA 1M-step completion batch

These configurations run the twelve KUKA-v3 tasks with validated retained SAC
experts for a full 1,000,000 environment steps. They preserve the established
single-task SAC settings, use 50 deterministic fixed-task evaluations, and
disable early stopping so completion means an actual one million steps.

Every run starts from scratch with seed 0. The retained canonical `policy.zip`
files include policy and optimizer state but no replay buffer, while
`train_sac.py --resume-from` deliberately requires a matching periodic replay
buffer. Treating a retained policy as an exact resume would therefore be
incorrect.

Outputs are isolated below
`runs/training_batches/kuka_1m_completion/<task>/seed_0_fresh_1m`. Existing
directories are never overwritten; the trainer creates a timestamped sibling
on collision.

The batch intentionally excludes `kuka-push-v3`, `kuka-push-cube-v3`, and
`kuka-pick-place-v3`. Push and Push-Cube now have separate Sawyer-inspired
pilot and canonical 1M configurations, while Pick-Place has no verified
successful policy. Keep those distinct protocols outside this completion batch.
