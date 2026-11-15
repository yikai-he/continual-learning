"""Read-only DDPM versus Flow Matching four-stage representation comparison."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from continual_manipulation.evaluation import EvaluationMatrix  # noqa: E402
from continual_manipulation.metrics import average_performance, forgetting  # noqa: E402
from src.continual.diffcrl_artifacts import completed_stage_markers  # noqa: E402
from src.support.reproducibility import file_hash  # noqa: E402

TASKS = ("kuka-hammer-v3", "kuka-handle-press-side-v3", "kuka-button-press-v3", "kuka-reach-v3")


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    generator: str
    representation: str
    root: Path


def arms(ddpm_root: Path, fm_root: Path) -> tuple[Arm, ...]:
    return (
        Arm("ddpm_structured_22d", "DDPM 22D", "diffusion", "structured_22d", ddpm_root),
        Arm("ddpm_full_43d", "DDPM 43D", "diffusion", "full_43d", ddpm_root),
        Arm("fm_structured_22d", "FM 22D", "flow-matching", "structured_22d", fm_root),
        Arm("fm_full_43d", "FM 43D", "flow-matching", "full_43d", fm_root),
    )


def inspect(arm: Arm, seed: int) -> dict:
    directory = arm.root / f"{arm.key}_seed{seed}"
    result = {"arm": arm.label, "seed": seed, "status": "missing", "run_dir": str(directory)}
    if not directory.exists():
        return result
    result["status"] = "incomplete"
    if not (directory / "RUN_COMPLETE").is_file():
        result["issue"] = "RUN_COMPLETE absent"
        return result
    try:
        config = load_config(directory / "config.yaml", expected_experiment="diffcrl")
        representation = config.diffusion.trajectory_representation if arm.generator == "diffusion" else config.flow_matching.trajectory_representation
        if config.continual.replay_mode != arm.generator or representation != arm.representation:
            raise ValueError("generator or representation differs from requested arm")
        completed_stage_markers(directory, TASKS)
        matrix = EvaluationMatrix.load(directory / "evaluation_matrix.json").R
        reports = [json.loads((directory / f"stage_{i}_{task}" / "report.json").read_text()) for i, task in enumerate(TASKS)]
        manifest = json.loads((directory / "run_manifest.json").read_text())
        evaluation = json.loads((directory / "evaluation_matrix.json").read_text())
        result.update(
            status="complete",
            ap=[average_performance(matrix, i) for i in range(4)],
            forgetting=[forgetting(matrix, i) for i in range(4)],
            final_tasks=matrix[3].tolist(),
            generator_training_seconds=[r["timing"]["generator_training_seconds"] for r in reports],
            sampling_seconds=[r["timing"]["replay_generation_seconds"] for r in reports],
            pairing={
                "expert_hashes": {task: manifest["experts"][task]["checkpoint_sha256"] for task in TASKS},
                "current_data_hashes": [file_hash(directory / f"stage_{i}_{task}" / f"data_task_{i}.npy") for i, task in enumerate(TASKS)],
                "splits": [reports[i]["splits"][task] for i, task in enumerate(TASKS)],
                "evaluation_task_hashes": evaluation["cells"][0][0]["ordered_task_hashes"],
                "protocol": {
                    "tasks": config.continual.tasks,
                    "trajectories_per_task": config.continual.trajectories_per_task,
                    "replay_allocation": config.continual.replay_allocation,
                    "replay_budget": config.continual.replay_budget,
                    "historical_replay_training_only": config.continual.historical_replay_training_only,
                    "bc": config.bc.__dict__,
                    "evaluation": config.evaluation.__dict__,
                    "action_space": config.diffusion.action_space,
                    "generated_action_projection": config.diffusion.generated_action_projection,
                },
            },
        )
    except Exception as error:
        result.update(status="invalid", issue=str(error))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddpm-root", type=Path, default=ROOT / "runs/representation_ablation/four_stage_20plus20")
    parser.add_argument("--fm-root", type=Path, default=ROOT / "runs/representation_ablation/four_stage_20plus20_flow_matching")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expert-disagreement-json",
        type=Path,
        action="append",
        default=[],
        help="Optional DDPM/FM disagreement JSON; may be repeated.",
    )
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        parser.error("--seeds must contain unique non-negative integers")
    results = [inspect(arm, seed) for seed in seeds for arm in arms(args.ddpm_root, args.fm_root)]
    lookup = {(item["arm"], item["seed"]): item for item in results}
    disagreement: dict[tuple[str, int], list[float]] = {}
    for path in args.expert_disagreement_json:
        data = json.loads(path.read_text(encoding="utf-8"))
        status_methods = {item.get("method") for item in data.get("statuses", [])}
        for record in data.get("generated_replay", []):
            method = record.get("method")
            if method is None and not any(str(value).startswith("fm_") for value in status_methods):
                method = {
                    "22D": "ddpm_structured_22d",
                    "43D": "ddpm_full_43d",
                }.get(record.get("representation"))
            if method not in {arm.key for arm in arms(args.ddpm_root, args.fm_root)}:
                continue
            value = record["metrics"]["raw_observation"]["raw_action"]["mean_squared_l2"]
            disagreement.setdefault((method, int(record["seed"])), []).append(float(value))
    labels = {arm.key: arm.label for arm in arms(args.ddpm_root, args.fm_root)}
    for (method, seed), values in disagreement.items():
        lookup[(labels[method], seed)]["expert_disagreement_equal_group_mean"] = float(np.mean(values))

    pairing_issues = []
    for seed in seeds:
        complete = [lookup[(arm.label, seed)] for arm in arms(args.ddpm_root, args.fm_root) if lookup[(arm.label, seed)]["status"] == "complete"]
        if len(complete) > 1:
            reference = complete[0]["pairing"]
            for item in complete[1:]:
                if item["pairing"] != reference:
                    pairing_issues.append(f"seed {seed}: {item['arm']} is not exactly paired with {complete[0]['arm']}")

    comparison_specs = (
        ("ddpm_representation_43d_minus_22d", "DDPM 43D", "DDPM 22D"),
        ("fm_representation_43d_minus_22d", "FM 43D", "FM 22D"),
        ("generator_fm_minus_ddpm_22d", "FM 22D", "DDPM 22D"),
        ("generator_fm_minus_ddpm_43d", "FM 43D", "DDPM 43D"),
    )
    fields = {
        "final_ap": lambda item: item["ap"][3],
        "final_forgetting": lambda item: item["forgetting"][3],
        **{f"final_task_{index}": (lambda item, i=index: item["final_tasks"][i]) for index in range(4)},
        **{f"stage_ap_{index}": (lambda item, i=index: item["ap"][i]) for index in range(4)},
        **{f"stage_forgetting_{index}": (lambda item, i=index: item["forgetting"][i]) for index in range(4)},
    }
    paired_effects = {}
    for name, left, right in comparison_specs:
        paired_effects[name] = {}
        for field, getter in fields.items():
            values = [
                {"seed": seed, "difference": float(getter(lookup[(left, seed)]) - getter(lookup[(right, seed)]))}
                for seed in seeds
                if lookup[(left, seed)]["status"] == lookup[(right, seed)]["status"] == "complete"
            ]
            array = np.asarray([item["difference"] for item in values])
            paired_effects[name][field] = {
                "per_seed": values,
                "mean": float(array.mean()) if len(array) else None,
                "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else None,
                "matched_n": len(array),
            }

    interactions = []
    for seed in seeds:
        matched = [lookup[(label, seed)] for label in ("DDPM 22D", "DDPM 43D", "FM 22D", "FM 43D")]
        if all(item["status"] == "complete" for item in matched):
            item = {"seed": seed}
            for field, getter in fields.items():
                item[field] = float((getter(matched[3]) - getter(matched[2])) - (getter(matched[1]) - getter(matched[0])))
            interactions.append(item)
    payload = {"schema_version": 1, "results": results, "pairing_issues": pairing_issues, "paired_effects": paired_effects, "interactions": interactions}
    if interactions:
        payload["interaction_summary"] = {}
        for field in fields:
            values = np.asarray([item[field] for item in interactions])
            payload["interaction_summary"][field] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None, "matched_n": len(values)}
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
