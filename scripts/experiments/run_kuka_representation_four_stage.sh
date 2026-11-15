#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

seeds="0,1,2"
methods="ddpm_structured_22d,ddpm_goal_conditioned_40d,ddpm_full_43d"
dry_run=false
selected_methods=()

usage() {
  cat <<'EOF'
Usage: run_kuka_representation_four_stage.sh [options]

  --seeds LIST    Comma-separated subset of seeds 0,1,2,3,4 (default: 0,1,2).
  --method NAME   Select one method; may be repeated.
  --dry-run       Print exact commands without creating runs.
  -h, --help      Show this help.

Methods: ddpm_structured_22d, ddpm_goal_conditioned_40d, ddpm_full_43d
EOF
}

while (($#)); do
  case "$1" in
    --seeds)
      [[ $# -ge 2 ]] || { echo "--seeds requires a value" >&2; exit 2; }
      seeds="$2"
      shift 2
      ;;
    --method)
      [[ $# -ge 2 ]] || { echo "--method requires a value" >&2; exit 2; }
      selected_methods+=("$2")
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ((${#selected_methods[@]})); then
  methods="$(IFS=,; echo "${selected_methods[*]}")"
fi

IFS=',' read -r -a seed_list <<< "$seeds"
IFS=',' read -r -a method_list <<< "$methods"
((${#seed_list[@]})) || { echo "At least one seed is required." >&2; exit 2; }
((${#method_list[@]})) || { echo "At least one method is required." >&2; exit 2; }

config_root="configs/experiments/kuka_trajectory_representations/four_stage_20plus20"
output_root="runs/representation_ablation/four_stage_20plus20"
declare -A configs=(
  [ddpm_structured_22d]="$config_root/ddpm_structured_22d.yaml"
  [ddpm_goal_conditioned_40d]="$config_root/ddpm_goal_conditioned_40d.yaml"
  [ddpm_full_43d]="$config_root/ddpm_full_43d.yaml"
)

for seed in "${seed_list[@]}"; do
  [[ "$seed" =~ ^[0-4]$ ]] || {
    echo "Invalid seed '$seed'; allowed seeds are 0,1,2,3,4." >&2
    exit 2
  }
done
for method in "${method_list[@]}"; do
  [[ -n "${configs[$method]+set}" ]] || {
    echo "Unknown method '$method'." >&2
    exit 2
  }
done

for seed in "${seed_list[@]}"; do
  for method in "${method_list[@]}"; do
    run_name="${method}_seed${seed}"
    run_dir="$output_root/$run_name"
    command=(
      conda run -n metaworld python train_diffcrl.py
      --config "${configs[$method]}"
      --seed "$seed"
      --run-name "$run_name"
    )

    if [[ -e "$run_dir" ]]; then
      echo "Refusing existing output: $run_dir" >&2
      exit 1
    fi

    echo "seed=$seed method=$method output=$run_dir"
    if [[ "$dry_run" == true ]]; then
      printf '  '
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi

    if "${command[@]}"; then
      echo "COMPLETED seed=$seed method=$method output=$run_dir"
    else
      status=$?
      echo "FAILED seed=$seed method=$method output=$run_dir status=$status" >&2
      exit "$status"
    fi
  done
done
