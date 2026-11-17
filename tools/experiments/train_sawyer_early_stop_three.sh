#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

case "${1:-}" in
  --preflight-only|--dry-run) execute=false ;;
  --run) execute=true ;;
  *) echo "Usage: $0 --preflight-only|--dry-run|--run" >&2; exit 2 ;;
esac
[[ $# -eq 1 ]] || { echo "Usage: $0 --preflight-only|--dry-run|--run" >&2; exit 2; }

python_bin="${PYTHON_BIN:-python}"
configs=(
  configs/sac/sawyer_push_back_from_faucet_close_early_stop.yaml
  configs/sac/sawyer_stick_pull_continuation_early_stop.yaml
  configs/sac/sawyer_push_continuation_early_stop.yaml
)
tasks=(push-back-v3 stick-pull-v3 push-v3)
outputs=(
  runs/sac_early_stop_20261117/push-back-v3/seed_0
  runs/sac_early_stop_20261117/stick-pull-v3/seed_0
  runs/sac_early_stop_20261117/push-v3/seed_0
)

command -v "$python_bin" >/dev/null 2>&1 || { echo "Python not found: $python_bin" >&2; exit 127; }
"$python_bin" -c "import stable_baselines3, continual_manipulation" || {
  echo "Activate the project's MetaWorld environment or set PYTHON_BIN." >&2
  exit 1
}

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  task="${tasks[$index]}"
  output="${outputs[$index]}"
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
  [[ ! -e "$output" ]] || { echo "Refusing existing run: $output" >&2; exit 1; }
  "$python_bin" - "$config" "$task" "$output" <<'PY'
import sys
from pathlib import Path
from src.config import load_config, output_directory
from src.support.expert_manifest import load_expert_manifest, validate_expert_checkpoint
from src.support.reproducibility import file_hash

config_path, task, expected_output = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
cfg = load_config(config_path, expected_experiment="sac")
assert cfg.continual.tasks == [task]
assert cfg.environment.backend == "metaworld-v3"
assert cfg.environment.reward_function_version == "v2"
assert cfg.runtime.seed == 0
assert cfg.sac.early_stopping.enabled
assert cfg.evaluation.mode == "fixed-tasks"
assert cfg.evaluation.episodes == 100
assert cfg.evaluation.frequency == 40_000
assert cfg.evaluation.task_set_seed == 10_000
assert cfg.evaluation.final_task_set_seeds == [20_000, 20_001]
assert output_directory(cfg) == expected_output

source_name = cfg.resume.checkpoint if cfg.resume.enabled else cfg.actor_init.checkpoint
source = Path(source_name).resolve()
manifest_path = source.parent / "manifest.json"
manifest = load_expert_manifest(manifest_path)
assert source.is_file()
assert file_hash(source) == manifest["checkpoint_sha256"]
assert manifest["backend"] == "metaworld-v3"
assert manifest["reward_function_version"] == "v2"
assert manifest["observation_shape"] == [39]
assert manifest["action_shape"] == [4]
assert manifest["horizon"] == 200
if cfg.resume.enabled:
    validate_expert_checkpoint(
        source,
        expected_task=task,
        expected_backend="metaworld-v3",
        expected_reward_function_version="v2",
        expected_observation_shape=(39,),
        expected_action_shape=(4,),
        expected_horizon=200,
        manifest_path=manifest_path,
        allow_legacy=False,
    )
else:
    assert cfg.actor_init.mode == "sac_actor"
    assert manifest["task_name"] == "faucet-close-v3"
PY
  echo "READY order=$((index + 1)) task=$task config=$config output=$output"
done

if [[ "$execute" == false ]]; then
  echo "Preflight complete; no training was launched."
  exit 0
fi

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  task="${tasks[$index]}"
  output="${outputs[$index]}"
  temporary="$(mktemp "${TMPDIR:-/tmp}/sawyer-early-stop-${index}.XXXXXX")"
  status=0
  echo "[$(date --iso-8601=seconds)] START task=$task"
  "$python_bin" -u train_sac.py --config "$config" 2>&1 | tee "$temporary" || status="${PIPESTATUS[0]}"
  if [[ -d "$output" && ! -e "$output/console.log" ]]; then
    mv "$temporary" "$output/console.log"
  else
    echo "Console retained at $temporary" >&2
  fi
  [[ "$status" -eq 0 ]] || { echo "FAILED task=$task status=$status" >&2; exit "$status"; }
  echo "[$(date --iso-8601=seconds)] COMPLETE task=$task"
done
