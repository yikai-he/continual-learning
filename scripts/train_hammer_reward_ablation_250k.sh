#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repository_root"

run_experiment() {
  local label="$1"
  local config="$2"

  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] START: ${label} (seed 0)"
  python -u train_sac.py \
    --config "$config" \
    --seed 0
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] FINISH: ${label} (seed 0)"
}

run_experiment \
  "Hammer reward ablation: original" \
  "configs/sac/kuka_hammer_v3_ablation_original_250k.yaml"

run_experiment \
  "Hammer reward ablation: nail-progress weight 2" \
  "configs/sac/kuka_hammer_v3_ablation_progress_w2_250k.yaml"

run_experiment \
  "Hammer reward ablation: nail-progress weight 4" \
  "configs/sac/kuka_hammer_v3_ablation_progress_w4_250k.yaml"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] All Hammer 250k screening runs finished."
