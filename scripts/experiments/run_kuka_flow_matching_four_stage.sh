#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

seeds="0,1,2,3,4"
methods="fm_structured_22d,fm_full_43d"
dry_run=false
selected_methods=()

usage() {
  cat <<'EOF'
Usage: run_kuka_flow_matching_four_stage.sh [options]

  --seeds LIST    Comma-separated subset of seeds 0,1,2,3,4 (default: all).
  --method NAME   Select one method; may be repeated.
  --dry-run       Validate inputs and print exact commands without creating runs.
  -h, --help      Show this help.

Methods: fm_structured_22d, fm_full_43d
EOF
}

while (($#)); do
  case "$1" in
    --seeds) [[ $# -ge 2 ]] || { echo "--seeds requires a value" >&2; exit 2; }; seeds="$2"; shift 2 ;;
    --method) [[ $# -ge 2 ]] || { echo "--method requires a value" >&2; exit 2; }; selected_methods+=("$2"); shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ((${#selected_methods[@]})); then methods="$(IFS=,; echo "${selected_methods[*]}")"; fi
IFS=',' read -r -a seed_list <<< "$seeds"
IFS=',' read -r -a method_list <<< "$methods"
((${#seed_list[@]})) || { echo "At least one seed is required." >&2; exit 2; }
((${#method_list[@]})) || { echo "At least one method is required." >&2; exit 2; }

config_root="configs/experiments/kuka_trajectory_representations/four_stage_20plus20_flow_matching"
output_root="runs/representation_ablation/four_stage_20plus20_flow_matching"
declare -A configs=(
  [fm_structured_22d]="$config_root/fm_structured_22d.yaml"
  [fm_full_43d]="$config_root/fm_full_43d.yaml"
)

for seed in "${seed_list[@]}"; do
  [[ "$seed" =~ ^[0-4]$ ]] || { echo "Invalid seed '$seed'; allowed seeds are 0,1,2,3,4." >&2; exit 2; }
done
for method in "${method_list[@]}"; do
  [[ -n "${configs[$method]+set}" ]] || { echo "Unknown method '$method'." >&2; exit 2; }
done

# Typed loading validates the full scientific configuration. DiffCRL then
# manifest-validates every expert checkpoint before reserving a new run output.
for method in "${method_list[@]}"; do
  conda run -n metaworld python -c \
    "from src.config import load_config; load_config('${configs[$method]}', expected_experiment='diffcrl')"
done

for seed in "${seed_list[@]}"; do
  for method in "${method_list[@]}"; do
    run_name="${method}_seed${seed}"
    run_dir="$output_root/$run_name"
    command=(conda run -n metaworld python train_diffcrl.py --config "${configs[$method]}" --seed "$seed" --run-name "$run_name")
    if [[ -e "$run_dir" ]]; then
      echo "Refusing existing output: $run_dir" >&2; exit 1
    fi
    echo "seed=$seed method=$method output=$run_dir"
    if [[ "$dry_run" == true ]]; then
      printf '  '; printf '%q ' "${command[@]}"; printf '\n'; continue
    fi
    if "${command[@]}"; then
      echo "COMPLETED seed=$seed method=$method output=$run_dir"
    else
      status=$?; echo "FAILED seed=$seed method=$method output=$run_dir status=$status" >&2; exit "$status"
    fi
  done
done
