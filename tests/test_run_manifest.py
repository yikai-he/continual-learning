import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from src.config import default_config, save_resolved_config
from src.support.run_manifest import complete_run, create_run_manifest


def test_manifest_required_fields_and_completion_markers(tmp_path):
    config = default_config("diffcrl")
    config = replace(config, runtime=replace(config.runtime, output=str(tmp_path)))
    save_resolved_config(config, tmp_path)
    create_run_manifest(
        tmp_path,
        config,
        experts={"reach-v3": {"checkpoint_sha256": "abc"}},
        task_banks={"reach-v3": {"disjoint": True}},
    )
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "running"
    assert manifest["repository"]["commit"]
    assert manifest["environment"]["python_version"]
    assert manifest["experiment"]["config_path"] == "config.yaml"
    assert manifest["experiment"]["config_sha256"]
    assert "resolved_config" not in manifest["experiment"]
    assert manifest["experts"]["reach-v3"]["checkpoint_sha256"] == "abc"
    assert (tmp_path / "RUNNING").is_file()
    assert not (tmp_path / "RUN_COMPLETE").exists()

    complete_run(tmp_path)
    completed = json.loads((tmp_path / "run_manifest.json").read_text())
    assert completed["status"] == "complete"
    assert completed["run_completed_at"]
    assert not (tmp_path / "RUNNING").exists()
    assert (tmp_path / "RUN_COMPLETE").is_file()


def test_manifest_creation_is_exclusive_and_strict(tmp_path):
    config = default_config("diffcrl")
    config = replace(config, runtime=replace(config.runtime, output=str(tmp_path)))
    save_resolved_config(config, tmp_path)
    create_run_manifest(tmp_path, config)
    original = (tmp_path / "run_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_run_manifest(tmp_path, config)
    assert (tmp_path / "run_manifest.json").read_bytes() == original

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    save_resolved_config(config, invalid)
    with pytest.raises(ValueError, match="Out of range float values"):
        create_run_manifest(
            invalid,
            config,
            experts={"reach-v3": {"success_rate": float("nan")}},
        )
    assert not (invalid / "run_manifest.json").exists()
    assert not (invalid / "RUNNING").exists()


def test_completion_marker_waits_for_manifest_replacement(tmp_path):
    config = default_config("diffcrl")
    config = replace(config, runtime=replace(config.runtime, output=str(tmp_path)))
    save_resolved_config(config, tmp_path)
    create_run_manifest(tmp_path, config)

    with patch(
        "src.support.run_manifest.atomic_replace_json",
        side_effect=OSError("publication failed"),
    ), pytest.raises(OSError, match="publication failed"):
        complete_run(tmp_path)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "running"
    assert (tmp_path / "RUNNING").is_file()
    assert not (tmp_path / "RUN_COMPLETE").exists()
