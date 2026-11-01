#!/usr/bin/env python3
"""Render the ten verified KUKA expert runs through render_kuka_push_policy.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts/render_kuka_push_policy.py"
OUTPUT_DIR = ROOT / "rendered_videos/kuka_10_tasks"
TASK_INDEX = 0

MODELS = {
    task: f"runs/experts/kuka-v3/{task.removeprefix('kuka-')}/policy.zip"
    for task in (
        "kuka-button-press-v3",
        "kuka-door-open-v3",
        "kuka-drawer-close-v3",
        "kuka-faucet-close-v3",
        "kuka-faucet-open-v3",
        "kuka-hammer-v3",
        "kuka-handle-press-side-v3",
        "kuka-reach-v3",
        "kuka-window-close-v3",
        "kuka-window-open-v3",
    )
}


def main() -> None:
    for task, model_path in MODELS.items():
        model = ROOT / model_path
        output = OUTPUT_DIR / f"{task}.mp4"
        if not model.is_file():
            raise FileNotFoundError(f"verified checkpoint disappeared: {model}")
        subprocess.run(
            [
                sys.executable,
                str(RENDERER),
                "--env-name", task,
                "--model", str(model),
                "--task-index", str(TASK_INDEX),
                "--no-loop",
                "--output-video", str(output),
                "--overwrite-video",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
