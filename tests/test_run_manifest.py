import json
from dataclasses import replace

from src.config import default_config
from src.support.run_manifest import complete_run, create_run_manifest


def test_manifest_required_fields_and_completion_markers(tmp_path):
    config = default_config("diffcrl")
    config = replace(config, runtime=replace(config.runtime, output=str(tmp_path)))
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
    assert manifest["experiment"]["canonical_config_sha256"]
    assert manifest["experts"]["reach-v3"]["checkpoint_sha256"] == "abc"
    assert (tmp_path / "RUNNING").is_file()
    assert not (tmp_path / "RUN_COMPLETE").exists()

    complete_run(tmp_path)
    completed = json.loads((tmp_path / "run_manifest.json").read_text())
    assert completed["status"] == "complete"
    assert completed["run_completed_at"]
    assert not (tmp_path / "RUNNING").exists()
    assert (tmp_path / "RUN_COMPLETE").is_file()
