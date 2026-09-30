import json
from pathlib import Path


def write_json(path, value):
    """Write strict JSON to a new file, refusing overwrite and non-finite values."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
