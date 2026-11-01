import hashlib
import subprocess
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(path, *args):
    """Return one read-only Git command's stripped output, or ``None``."""
    try:
        return subprocess.check_output(
            ("git", "-C", str(path), *args),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def state_hash(model):
    """Hash ordered state-dict names and raw CPU tensor bytes for integrity checks."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
