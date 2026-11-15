#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
seed=0
tasks=(stick-pull-v3)
configs=(stick_pull_v3.yaml)
output_root="$repo_root/runs/cw5_experts/legacy_sac/reward_v2"
log_root="$repo_root/logs/cw5_experts/legacy_sac"

command -v "$python_bin" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $python_bin" >&2
  exit 127
}
[[ -f "$repo_root/train_sac.py" ]] || {
  echo "Missing training entry point: $repo_root/train_sac.py" >&2
  exit 1
}
"$python_bin" -c "import stable_baselines3, continual_manipulation" || {
  echo "Required SAC dependencies are unavailable via $python_bin" >&2
  exit 1
}

for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  config="$repo_root/configs/sac/cw5/${configs[$index]}"
  run_dir="$output_root/$task/seed_$seed"
  log_file="$log_root/${task}_seed${seed}.log"
  [[ -f "$config" ]] || { echo "Missing configuration: $config" >&2; exit 1; }
  "$python_bin" -c \
    "import sys; from src.config import load_config; load_config(sys.argv[1], expected_experiment='sac')" \
    "$config"
  [[ ! -e "$run_dir" ]] || { echo "Refusing existing run: $run_dir" >&2; exit 1; }
  [[ ! -e "$log_file" ]] || { echo "Refusing existing log: $log_file" >&2; exit 1; }
done

mkdir -p "$log_root"
for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  config="configs/sac/cw5/${configs[$index]}"
  log_file="$log_root/${task}_seed${seed}.log"
  echo "[$(date --iso-8601=seconds)] START task=$task seed=$seed"
  if "$python_bin" -u train_sac.py --config "$config" 2>&1 | tee "$log_file"; then
    echo "[$(date --iso-8601=seconds)] COMPLETE task=$task seed=$seed"
  else
    status="${PIPESTATUS[0]}"
    echo "[$(date --iso-8601=seconds)] FAILED task=$task seed=$seed status=$status" >&2
    exit "$status"
  fi
done
