# Diffusion Replay for Continual RL

This repository contains the continual-learning and DiffCRL experiments for
modern MetaWorld v3. Default Sawyer tasks use the `metaworld` Conda
environment.

## Environment setup

Run from the repository root:

```bash
conda activate metaworld
python -m pip install -r requirements.txt
```

The expected package version is `metaworld==3.1.1`.

## KUKA experiments

KUKA experiments require a separate custom MetaWorld fork, `metaworld-kuka`.

Clone or obtain that repository separately, then install it in the dedicated
`metaworld-kuka` Conda environment:

```bash
conda activate metaworld-kuka
cd metaworld-kuka
python -m pip install -e .
```

## Experiment types

- **Single-task SAC expert training** trains an independent expert checkpoint
  for one task with `train_sac.py`; DiffCRL uses these experts to collect data.
- **Sequential SAC** is a continual-learning baseline that trains one SAC
  learner across tasks without pretrained experts, behavioral cloning, or
  diffusion replay.
- **DiffCRL with diffusion replay** trains the shared behavior-cloning policy
  from current expert trajectories and generated trajectories for prior tasks.
- **DiffCRL without replay** is the corresponding ablation: it retains expert
  supervision and behavioral cloning but disables prior-task diffusion replay.

## Maintained configurations

- `configs/sac/sac.yaml`
- `configs/diffcrl/diffcrl.yaml`
- `configs/diffcrl/diffcrl_no_replay.yaml`
- `configs/continual_sac/continual_sac.yaml`

The corresponding `*_smoke.yaml` files are small developer/testing presets.

## Single-task experts and DiffCRL

First, train the SAC experts required by the selected DiffCRL configuration.
For each task, copy `configs/sac/sac.yaml`, update `continual.tasks[0]`, and run:

```bash
python -u train_sac.py --config path/to/expert-config.yaml --seed 0
```

Then update the expert checkpoint paths under `continual.experts` in
`configs/diffcrl/diffcrl.yaml` and start the continual-learning run:

```bash
python -u train_diffcrl.py \
  --config configs/diffcrl/diffcrl.yaml \
  --seed 0 \
  --run-name example-seed0
```

Results and checkpoints are written under `runs/`. Use a new `--run-name` for
each run because existing run directories are not overwritten.

For a fast pipeline check, use `configs/diffcrl/diffcrl_smoke.yaml`.

Run the no-replay ablation with:

```bash
python -u train_diffcrl.py \
  --config configs/diffcrl/diffcrl_no_replay.yaml \
  --seed 0 \
  --run-name no-replay-seed0
```

## Sequential SAC baseline

Sequential SAC maintains one SAC learner across the task sequence. The learner
state persists across task boundaries, while task-local replay is reset at each
boundary. It does not load pretrained experts or use behavioral cloning or
diffusion replay.

Smoke run:

```bash
python -u train_continual_sac.py \
  --config configs/continual_sac/continual_sac_smoke.yaml
```

Formal five-task run:

```bash
python -u train_continual_sac.py \
  --config configs/continual_sac/continual_sac.yaml
```

These configurations write to `runs/continual_sac_smoke/` and
`runs/formal/continual_sac_5task/`, respectively. Existing run directories are
not overwritten; use `--run-name NAME` to place a run in a distinct named
subdirectory.

## Utilities

Evaluate a trained SAC policy:

```bash
python scripts/evaluation/evaluate_sac.py \
  --task reach-v3 \
  --model runs/path/to/model.zip \
  --episodes 50
```

Render a saved SAC or DiffCRL policy to MP4 (requires FFmpeg):

```bash
python scripts/visualization/render_policy_video.py \
  --task reach-v3 \
  --model path/to/checkpoint \
  --policy-type sac \
  --output videos/reach.mp4
```

Use `--policy-type diffcrl` for a saved `GeneralPolicy` checkpoint. Run any
entry point with `--help` to see its remaining options.
