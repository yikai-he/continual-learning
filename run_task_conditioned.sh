#!/usr/bin/env bash
set -euo pipefail

# Resolve all relative configuration, expert, output, and log paths from the
# repository containing this script, regardless of the caller's directory.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO_ROOT"

CONDA_ENV="metaworld"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs/task_conditioned_${RUN_TIMESTAMP}"

CONFIGS=(
  "configs/diffcrl/diffcrl_task_conditioned.yaml"
  "configs/diffcrl/diffcrl_no_replay_task_conditioned.yaml"
)
LABELS=(
  "diffcrl_task_conditioned"
  "no_replay_task_conditioned"
)
OUTPUT_BASES=(
  "runs/formal/diffcrl_5task_task_conditioned"
  "runs/formal/no_replay_5task_task_conditioned"
)
SEEDS=(0 1 2)

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is not available on PATH." >&2
  exit 1
}

echo "Preflight: validating configurations, CPU selection, experts, and outputs..."
conda run -n "$CONDA_ENV" python -c '
import sys
from dataclasses import replace
from pathlib import Path

from src.config import load_config, output_directory

expected_modes = ("diffusion", "none")
expected_tasks = (
    "button-press-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "window-close-v3",
)

for config_path, replay_mode in zip(sys.argv[1:], expected_modes):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    config = load_config(path, expected_experiment="diffcrl")
    if config.bc.task_conditioning != "one_hot":
        raise ValueError(f"{path}: bc.task_conditioning must be one_hot")
    if config.runtime.device != "cpu":
        raise ValueError(f"{path}: runtime.device must remain cpu")
    if config.continual.replay_mode != replay_mode:
        raise ValueError(
            f"{path}: expected replay mode {replay_mode!r}, "
            f"found {config.continual.replay_mode!r}"
        )
    if tuple(config.continual.tasks) != expected_tasks:
        raise ValueError(f"{path}: unexpected formal five-task ordering")
    for task in expected_tasks:
        expert = Path(config.continual.experts[task])
        if not expert.is_file():
            raise FileNotFoundError(f"Expert checkpoint not found: {expert}")
    for seed in (0, 1, 2):
        target = output_directory(
            replace(
                config,
                runtime=replace(config.runtime, seed=seed, run_name=f"seed_{seed}"),
            )
        )
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
        print(f"  available: {target}")
' "${CONFIGS[@]}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight completed successfully; no experiments were launched."
  exit 0
fi

# mkdir without -p deliberately refuses a timestamp collision. The parent is
# not an experiment-result directory and is safe to create if absent.
mkdir -p "$REPO_ROOT/logs"
mkdir "$LOG_DIR"

for config_index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$config_index]}"
  label="${LABELS[$config_index]}"
  output_base="${OUTPUT_BASES[$config_index]}"

  for seed in "${SEEDS[@]}"; do
    run_name="seed_${seed}"
    output_dir="$REPO_ROOT/$output_base/$run_name"
    log_file="$LOG_DIR/${label}_seed_${seed}_${RUN_TIMESTAMP}.log"

    echo
    echo "START: $label (seed $seed)"
    echo "  config: $REPO_ROOT/$config"
    echo "  output: $output_dir"
    echo "  log:    $log_file"

    conda run --no-capture-output -n "$CONDA_ENV" \
      python train_diffcrl.py \
      --config "$config" \
      --seed "$seed" \
      --run-name "$run_name" \
      2>&1 | tee "$log_file"

    echo "COMPLETE: $label (seed $seed)"
    echo "  output: $output_dir"
    echo "  log:    $log_file"
  done
done

echo
echo "All six formal Task-Conditioned BC experiments completed successfully."
echo "Logs: $LOG_DIR"
echo "Outputs:"
for output_base in "${OUTPUT_BASES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "  $REPO_ROOT/$output_base/seed_${seed}"
  done
done
