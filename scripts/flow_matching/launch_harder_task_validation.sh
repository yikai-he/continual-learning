#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {all|train-flow|train-diffusion|generate-flow-10|generate-flow-50|generate-diffusion|bc-real|bc-flow-10|bc-flow-50|bc-diffusion}" >&2
  exit 2
fi
job="$1"
case "$job" in
  all|train-flow|train-diffusion|generate-flow-10|generate-flow-50|generate-diffusion|bc-real|bc-flow-10|bc-flow-50|bc-diffusion) ;;
  *) echo "Unknown job: $job" >&2; exit 2 ;;
esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${FLOW_VALIDATION_PYTHON:-/home/yikai/anaconda3/envs/metaworld/bin/python3.10}"
metaworld_root="${METAWORLD_ROOT:-/home/yikai/Documents/Code/Thesis/metaworld-kuka-v3}"
progress_flag="--progress"
[[ "${FLOW_VALIDATION_PROGRESS:-1}" == "0" ]] && progress_flag="--no-progress"
export PYTHONPATH="${metaworld_root}:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$repo_root"
exec "$python_bin" scripts/flow_matching/run_harder_task_validation.py \
  --config configs/flow_matching_harder_task/hammer_v3.yaml \
  --job "$job" "$progress_flag"
