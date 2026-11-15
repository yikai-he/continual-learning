#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

config_root="configs/experiments/kuka_generator_comparison/four_stage_200plus200_seed0"
output_root="runs/generator_comparison/four_stage_200plus200"
all_methods=(ddpm_structured_22d ddpm_full_43d fm_structured_22d fm_full_43d)
methods=()
dry_run=false

usage() {
  cat <<'EOF'
Usage: run_kuka_generator_200plus200.sh [options]
  --method NAME  Select one method; may be repeated (default: all four).
  --dry-run      Validate configs and experts, then print commands only.
  -h, --help     Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --method) [[ $# -ge 2 ]] || { echo "--method requires a value" >&2; exit 2; }; methods+=("$2"); shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if ((${#methods[@]} == 0)); then methods=("${all_methods[@]}"); fi

for method in "${methods[@]}"; do
  valid=false
  for candidate in "${all_methods[@]}"; do [[ "$method" == "$candidate" ]] && valid=true; done
  [[ "$valid" == true ]] || { echo "Unknown method: $method" >&2; exit 2; }
done

for method in "${methods[@]}"; do
  config="$config_root/$method.yaml"
  # Typed config plus strict KUKA expert-manifest/hash/shape/horizon validation.
  conda run -n metaworld python -c "from src.config import load_config; from continual_manipulation.environments import OBSERVATION_SHAPE,ACTION_SHAPE,horizon_for_backend; from src.support.expert_manifest import validate_expert_checkpoint; c=load_config('$config',expected_experiment='diffcrl'); [validate_expert_checkpoint(p,expected_task=t,expected_backend=c.environment.backend,expected_reward_function_version=c.environment.reward_function_version,expected_observation_shape=OBSERVATION_SHAPE,expected_action_shape=ACTION_SHAPE,expected_horizon=horizon_for_backend(c.environment.backend),allow_legacy=False) for t,p in c.continual.experts.items()]" >/dev/null

  run_name="${method}_seed0"
  run_dir="$output_root/$run_name"
  command=(conda run -n metaworld python train_diffcrl.py --config "$config" --run-name "$run_name")
  if [[ -e "$run_dir" ]]; then
    echo "Refusing existing output: $run_dir" >&2; exit 1
  fi
  printf 'method=%s output=%s\n  ' "$method" "$run_dir"
  printf '%q ' "${command[@]}"; printf '\n'
  if [[ "$dry_run" != true ]]; then "${command[@]}"; fi
done
