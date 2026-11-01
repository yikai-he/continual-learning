#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 EXPERIMENT_ID [generate|bc|all]" >&2
  exit 2
fi

experiment_id="$1"
stage="${2:-generate}"
case "$experiment_id" in
  integration_steps_10|integration_steps_20|integration_steps_50|integration_steps_100|training_epochs_50|training_epochs_100|training_epochs_200) ;;
  *) echo "Unknown experiment: $experiment_id" >&2; exit 2 ;;
esac
case "$stage" in generate|bc|all) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/flow_matching/run_flow_matching_ablation.py"
config="$repo_root/configs/flow_matching_ablation/button_press_v3.yaml"
if [[ ! -f "$runner" || ! -f "$config" ]]; then
  echo "Flow Matching Phase 7B cannot be launched from this checkout." >&2
  echo "The original runner and configuration are absent from all repository history; restore them from an external source before running this workflow." >&2
  exit 2
fi
python_bin="${FLOW_ABLATION_PYTHON:-/home/yikai/anaconda3/envs/metaworld/bin/python3.10}"
metaworld_root="${METAWORLD_ROOT:-/home/yikai/Documents/Code/Thesis/metaworld-kuka-v3}"
export PYTHONPATH="${metaworld_root}:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$repo_root"
progress_flag="--progress"
if [[ "${FLOW_ABLATION_PROGRESS:-1}" == "0" ]]; then
  progress_flag="--no-progress"
fi
exec "$python_bin" "$runner" \
  --config "$config" \
  --experiment "$experiment_id" \
  --stage "$stage" \
  "$progress_flag"
