#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

seeds="1,2,3,4"
methods="ddpm_structured_22d,ddpm_goal_conditioned_40d,ddpm_full_43d,real_replay,no_replay"
dry_run=false

usage() {
  cat <<'EOF'
Usage: run_kuka_representation_multiseed.sh [options]

Options:
  --seeds LIST    Comma-separated subset of seeds 1,2,3,4 (default: all).
  --method NAME   Run one method (may be repeated).
  --dry-run       Print commands without creating or training anything.
  -h, --help      Show this help.

Methods: ddpm_structured_22d, ddpm_goal_conditioned_40d, ddpm_full_43d,
         real_replay, no_replay
EOF
}

selected_methods=()
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

declare -A configs=(
  [ddpm_structured_22d]="configs/experiments/kuka_trajectory_representations/two_task_ddpm_structured_22d.yaml"
  [ddpm_goal_conditioned_40d]="configs/experiments/kuka_trajectory_representations/two_task_ddpm_goal_conditioned_40d.yaml"
  [ddpm_full_43d]="configs/experiments/kuka_trajectory_representations/two_task_ddpm_full_43d.yaml"
  [real_replay]="configs/experiments/kuka_trajectory_representations/two_task_real_replay.yaml"
  [no_replay]="configs/experiments/kuka_trajectory_representations/two_task_no_replay.yaml"
)

declare -A run_prefixes=(
  [ddpm_structured_22d]="ddpm_structured_22d"
  [ddpm_goal_conditioned_40d]="ddpm_goal_conditioned_40d"
  [ddpm_full_43d]="ddpm_full_43d"
  [real_replay]="real_replay"
  [no_replay]="no_replay"
)

for seed in "${seed_list[@]}"; do
  [[ "$seed" =~ ^[1-4]$ ]] || {
    echo "Invalid seed '$seed'; allowed seeds are 1,2,3,4." >&2
    exit 2
  }
done

for method in "${method_list[@]}"; do
  [[ -n "${configs[$method]+set}" ]] || {
    echo "Unknown method '$method'." >&2
    exit 2
  }
done

output_root="runs/representation_ablation/two_task"

for seed in "${seed_list[@]}"; do
  for method in "${method_list[@]}"; do
    run_name="${run_prefixes[$method]}_seed${seed}"
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

    echo "seed=$seed method=$method run=$run_name"
    if [[ "$dry_run" == true ]]; then
      printf '  '
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi

    if "${command[@]}"; then
      echo "COMPLETED seed=$seed method=$method run=$run_name"
    else
      status=$?
      echo "FAILED seed=$seed method=$method run=$run_name status=$status" >&2
      exit "$status"
    fi
  done
done
