import json
import errno
import os
from pathlib import Path

import torch


def _publish_exclusive(temporary: Path, path: Path) -> None:
    """Publish without overwrite; tolerate filesystems without hard links."""
    try:
        os.link(temporary, path)
        temporary.unlink()
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV):
            raise
        # Google Drive FUSE may not support hard links. Filenames are uniquely
        # selected by the trainer; retain the explicit collision check there and
        # immediately before rename rather than making Drive persistence fail.
        if path.exists():
            raise FileExistsError(path) from error
        os.rename(temporary, path)


def write_json(path, value):
    """Write strict JSON to a new file, refusing overwrite and non-finite values."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def atomic_write_json(path, value):
    """Atomically create strict JSON without replacing an existing artifact."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists():
        raise FileExistsError(path)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_exclusive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path, value):
    """Atomically create a torch checkpoint without replacing an existing file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists():
        raise FileExistsError(path)
    try:
        with temporary.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_exclusive(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
