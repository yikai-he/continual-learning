#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {run|resume|diagnostics}" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${FLOW_PILOT_PYTHON:-/home/yikai/anaconda3/envs/metaworld/bin/python3.10}"
metaworld_root="${METAWORLD_ROOT:-/home/yikai/Documents/Code/Thesis/metaworld-kuka-v3}"
run_dir="$repo_root/runs/pilots/flow_matching_5task_fixed_budget/flow_matching_5task_fixed_budget_seed0"
export PYTHONPATH="${metaworld_root}:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$repo_root"
case "$1" in
  run)
    exec "$python_bin" train_diffcrl.py --config configs/diffcrl/flow_matching_5task_pilot.yaml
    ;;
  resume)
    exec "$python_bin" train_diffcrl.py --config configs/diffcrl/flow_matching_5task_pilot.yaml --resume-from "$run_dir"
    ;;
  diagnostics)
    exec "$python_bin" scripts/flow_matching/analyze_continual_pilot.py --run-dir "$run_dir"
    ;;
  *) echo "Unknown operation: $1" >&2; exit 2 ;;
esac
