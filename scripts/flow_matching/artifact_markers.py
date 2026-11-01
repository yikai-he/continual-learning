"""Hash-validated completion markers shared by Flow Matching study scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from src.support.io import write_json


def canonical_sha(value: object) -> str:
    """Hash a JSON-compatible value using one stable canonical encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path: str | Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_completion_marker(
    output: str | Path,
    job: str,
    config_sha256: str,
    artifacts: Mapping[str, str | Path],
) -> dict:
    """Write an exclusive completion marker after validating every artifact."""
    output = Path(output)
    records = {}
    for name, value in artifacts.items():
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Completed artifact is missing: {path}")
        records[name] = {"path": str(path), "sha256": file_sha(path)}
    marker = {
        "kind": "flow_matching_study_completion_v1",
        "job": job,
        "config_sha256": config_sha256,
        "artifacts": records,
    }
    write_json(output / f"{job}.complete.json", marker)
    return marker


def require_completion_marker(
    output: str | Path, job: str, config_sha256: str
) -> dict:
    """Load a matching completion marker and re-hash all recorded artifacts."""
    path = Path(output) / f"{job}.complete.json"
    marker = json.loads(path.read_text(encoding="utf-8"))
    if (
        marker.get("kind") != "flow_matching_study_completion_v1"
        or marker.get("job") != job
        or marker.get("config_sha256") != config_sha256
        or not isinstance(marker.get("artifacts"), dict)
    ):
        raise ValueError(f"Invalid completion marker: {path}")
    for artifact in marker["artifacts"].values():
        artifact_path = Path(artifact["path"])
        if (
            not artifact_path.is_file()
            or file_sha(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError(f"Completed artifact changed or is missing: {artifact_path}")
    return marker
