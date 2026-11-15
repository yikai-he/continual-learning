"""Strict YAML I/O and command-line overrides for experiment configurations."""

from __future__ import annotations

import argparse
import math
import types
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

import yaml
from continual_manipulation.sequences import resolve_named_sequence

from .schema import ExperimentConfig, default_config
from .validation import validate_config


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("YAML configuration keys must be strings.")
        if key in result:
            raise ValueError(f"Duplicate YAML field: {key}.")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def _typed(value, annotation, path):
    origin, args = get_origin(annotation), get_args(annotation)
    union_type = getattr(types, "UnionType", None)
    if origin is Union or union_type is not None and origin is union_type:
        if value is None and type(None) in args:
            return None
        allowed = [item for item in args if item is not type(None)]
        if len(allowed) != 1:
            raise ValueError(f"Unsupported type annotation for {path}.")
        return _typed(value, allowed[0], path)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list.")
        return [
            _typed(item, args[0], f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping.")
        return {
            _typed(key, args[0], path): _typed(item, args[1], f"{path}.{key}")
            for key, item in value.items()
        }
    if annotation is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number.")
        return float(value)
    if type(value) is not annotation:
        raise ValueError(
            f"{path} must be {annotation.__name__}, got {type(value).__name__}."
        )
    return value


def _merge(default, values, path=""):
    if not isinstance(values, dict):
        raise ValueError(f"{path or 'config'} must be a mapping.")
    hints = get_type_hints(type(default))
    known = {field.name for field in fields(default)}
    unknown = set(values) - known
    if unknown:
        name = sorted(unknown, key=str)[0]
        raise ValueError(f"Unknown config field: {path + '.' if path else ''}{name}.")
    updates = {}
    for name, value in values.items():
        location = f"{path}.{name}" if path else name
        prior = getattr(default, name)
        updates[name] = (
            _merge(prior, value, location)
            if is_dataclass(prior)
            else _typed(value, hints[name], location)
        )
    return replace(default, **updates)


def _resolve_sequence(values: dict) -> dict:
    continual = values.get("continual")
    if not isinstance(continual, dict) or "sequence" not in continual:
        return values
    if "tasks" in continual:
        raise ValueError(
            "continual.sequence and continual.tasks are mutually exclusive."
        )
    sequence_name = continual["sequence"]
    tasks = resolve_named_sequence(sequence_name)
    normalized = dict(values)
    normalized_continual = dict(continual)
    normalized_continual.pop("sequence")
    normalized_continual["tasks"] = list(tasks)
    normalized["continual"] = normalized_continual
    return normalized


def config_from_dict(
    values: dict,
    *,
    expected_experiment: str | None = None,
    overrides: dict | None = None,
) -> ExperimentConfig:
    if not isinstance(values, dict) or "experiment" not in values:
        raise ValueError("Missing required field: experiment.")
    experiment = values["experiment"]
    if type(experiment) is not str:
        raise ValueError("experiment must be a string.")
    if expected_experiment is not None and experiment != expected_experiment:
        raise ValueError(
            f"experiment must be {expected_experiment} for this entry point."
        )
    config = _merge(default_config(experiment), _resolve_sequence(values))
    if overrides:
        config = replace(config, runtime=_merge(config.runtime, overrides, "runtime"))
    validate_config(config)
    return config


def load_config(path, *, expected_experiment=None, overrides=None) -> ExperimentConfig:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot load config {path}: {error}") from error
    return config_from_dict(
        values,
        expected_experiment=expected_experiment,
        overrides=overrides,
    )


def config_to_dict(config: ExperimentConfig) -> dict:
    validate_config(config)
    values = asdict(config)
    if values["runtime"]["run_dir"] is None:
        values["runtime"].pop("run_dir")
    return values


def resolve_device(config: ExperimentConfig) -> ExperimentConfig:
    """Resolve auto and reject unavailable CUDA before creating output files."""
    import torch

    device = config.runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(
                f"runtime.device {device!r} is unavailable on this machine."
            )
    return replace(config, runtime=replace(config.runtime, device=device))


def parse_config(
    experiment: str,
    description: str,
    argv=None,
    *,
    config_required: bool = False,
    configure_parser=None,
    return_args: bool = False,
) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=description)
    argument = {"type": Path, "help": "Experiment YAML."}
    if config_required:
        argument["required"] = True
    else:
        argument["default"] = {
            "sac": Path("configs/sac/sac.yaml"),
            "diffcrl": Path("configs/diffcrl/diffcrl.yaml"),
        }[experiment]
        argument["help"] = "Experiment YAML (default: %(default)s)."
    parser.add_argument("--config", **argument)
    parser.add_argument("--seed", type=int, default=None, help="Override runtime.seed.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    parser.add_argument("--run-name", default=None, help="Override runtime.run_name.")
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args(argv)
    overrides = {
        name: getattr(args, name)
        for name in ("seed", "device", "run_name")
        if getattr(args, name) is not None
    }
    try:
        config = resolve_device(
            load_config(
                args.config, expected_experiment=experiment, overrides=overrides
            )
        )
    except ValueError as error:
        parser.error(str(error))
    return (config, args) if return_args else config


def output_directory(config: ExperimentConfig) -> Path:
    directory = Path(config.runtime.output)
    if config.runtime.run_name is not None:
        directory /= config.runtime.run_name
    return directory.resolve()


def save_resolved_config(config: ExperimentConfig, run_dir: Path) -> None:
    resolved = replace(
        config, runtime=replace(config.runtime, run_dir=str(run_dir.resolve()))
    )
    with (run_dir / "config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config_to_dict(resolved), stream, sort_keys=False)
