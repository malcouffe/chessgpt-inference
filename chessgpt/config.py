"""
Configuration for ChessGPT inference.

Provides dataclasses and a YAML + CLI-override loader.
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Type, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


# ===================================================================
# Generation config dataclasses
# ===================================================================


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    greedy: bool = False


@dataclass
class GenerateConfig:
    checkpoint: str = ""
    device: str = "auto"
    moves: str = ""
    max_new_moves: int = 80
    min_new_moves: int = 0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    legal_mask: bool = True
    stop_on_game_over: bool = True
    seed: int = 42
    verbose: bool = False


# ===================================================================
# YAML + CLI-override loader
# ===================================================================


def load_config(
    yaml_path: str | Path | None,
    config_cls: Type[T],
    overrides: list[str] | None = None,
) -> T:
    """Load a YAML config, apply CLI overrides, and build a typed dataclass.

    Args:
        yaml_path:   Path to YAML file, or None to use pure defaults.
        config_cls:  The top-level dataclass type.
        overrides:   List of "dotted.key=value" strings from --set.

    Returns:
        An instance of *config_cls* with all values populated.
    """
    raw: dict[str, Any] = {}

    if yaml_path is not None:
        path = Path(yaml_path)
        if path.exists() and path.stat().st_size > 0:
            with open(path) as f:
                loaded = yaml.safe_load(f)
            if loaded is not None:
                raw = loaded

    if overrides:
        for ov in overrides:
            _apply_override(raw, ov)

    return _build_dataclass(config_cls, raw)


# -------------------------------------------------------------------
# Internals
# -------------------------------------------------------------------


def _apply_override(d: dict, override: str) -> None:
    """Apply a single ``key.path=value`` override to a nested dict."""
    if "=" not in override:
        raise ValueError(
            f"Override must be in 'key.path=value' format, got: {override!r}"
        )

    key_path, value_str = override.split("=", 1)
    parts = key_path.split(".")

    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value_str


def _build_dataclass(cls: Type[T], raw: dict[str, Any]) -> T:
    """Recursively instantiate a dataclass from a (possibly nested) dict."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}

    for key in raw:
        if key not in known:
            print(f"[config] Warning: unknown key '{key}' for {cls.__name__}, ignoring.")

    for f in fields(cls):
        if f.name not in raw:
            continue

        value = raw[f.name]
        target_type = hints[f.name]

        if _is_dataclass_type(target_type):
            if isinstance(value, dict):
                kwargs[f.name] = _build_dataclass(target_type, value)
            else:
                raise TypeError(
                    f"Expected dict for nested config '{f.name}', got {type(value)}"
                )
        else:
            kwargs[f.name] = _coerce(value, target_type)

    return cls(**kwargs)


def _is_dataclass_type(tp: Any) -> bool:
    """Check if *tp* is a dataclass **type** (not an instance)."""
    try:
        return dataclasses.is_dataclass(tp) and isinstance(tp, type)
    except TypeError:
        return False


def _coerce(value: Any, target_type: type) -> Any:
    """Coerce *value* (possibly a CLI string) to *target_type*."""
    origin = getattr(target_type, "__origin__", None)

    if origin is None:
        if isinstance(value, target_type):
            return value

    if target_type is bool:
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no"):
                return False
            raise ValueError(f"Cannot coerce {value!r} to bool")
        return bool(value)

    if target_type is int:
        return int(float(value))

    if target_type is float:
        return float(value)

    if target_type is str:
        return str(value)

    if origin is tuple:
        if isinstance(value, (list, tuple)):
            args = getattr(target_type, "__args__", ())
            if args:
                return tuple(args[min(i, len(args) - 1)](v) for i, v in enumerate(value))
            return tuple(value)
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                args = getattr(target_type, "__args__", ())
                if args:
                    return tuple(args[min(i, len(args) - 1)](v) for i, v in enumerate(parsed))
                return tuple(parsed)

    if origin is list:
        if isinstance(value, str):
            return list(ast.literal_eval(value))
        if isinstance(value, (list, tuple)):
            return list(value)

    return target_type(value)
