import importlib.util
from pathlib import Path

import pytest

from scripts.flow_matching.artifact_markers import (
    canonical_sha,
    require_completion_marker,
    write_completion_marker,
)


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/flow_matching/run_harder_task_validation.py"
SPEC = importlib.util.spec_from_file_location("harder_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)
CONFIG = ROOT / "configs/flow_matching_harder_task/hammer_v3.yaml"


def test_full_configuration_is_matched_and_raw():
    config = MODULE.load_config(CONFIG)
    assert config["task"] == "hammer-v3"
    assert config["flow"]["epochs"] == 200
    assert config["flow"]["integration_steps"] == [10, 50]
    assert config["diffusion"]["action_space"] == "raw"
    assert config["generation"] == {"trajectories": 200, "seed": 100000, "action_projection": "none"}
    assert config["bc"]["epochs"] == 100
    assert config["evaluation"]["episodes"] == 50


def test_debug_configuration_is_short_and_separate():
    config = MODULE.load_config(CONFIG, True)
    assert config["flow"]["epochs"] == 2
    assert config["diffusion"]["epochs"] == 2
    assert config["generation"]["trajectories"] == 8
    assert config["output_dir"].endswith("_debug")


def test_unknown_field_and_step_grid_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"; path.write_text(CONFIG.read_text() + "unknown: true\n")
    with pytest.raises(ValueError): MODULE.load_config(path)
    path.write_text(CONFIG.read_text().replace("[10, 50]", "[10, 100]"))
    with pytest.raises(ValueError, match="10 and 50"): MODULE.load_config(path)


def test_required_assets_exist_and_have_formal_shape():
    config = MODULE.load_config(CONFIG)
    assets = {key: ROOT / value for key, value in config["assets"].items()}
    assert all(path.is_file() for path in assets.values())
    real, dynamic, goals, train_ids, validation_ids = MODULE.dataset(config, assets)
    assert real.shape == (200, 200, 43)
    assert dynamic.shape == (200, 200, 22)
    assert goals.shape == (200, 3)
    assert (len(train_ids), len(validation_ids)) == (160, 40)


def test_completion_markers_are_canonical_and_hash_validated(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"value": 1}\n', encoding="utf-8")
    first = canonical_sha({"b": 2, "a": 1})
    assert first == canonical_sha({"a": 1, "b": 2})

    marker = write_completion_marker(tmp_path, "train-flow", first, {"data": artifact})
    assert require_completion_marker(tmp_path, "train-flow", first) == marker

    artifact.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed or is missing"):
        require_completion_marker(tmp_path, "train-flow", first)
