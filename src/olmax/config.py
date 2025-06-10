from __future__ import annotations

import dataclasses
import sys
import typing
from pathlib import Path
from typing import Any, Sequence, Type, TypeVar

import numpy as np
import yaml
from dataclass_extensions import decode, encode

from .types import *

encode.register_encoder(str, np.dtype)

C = TypeVar("C", bound=Dataclass)


@typing.overload
def parse_config_from_args(
    config: PathOrStr,
    config_class: Type[C],
    /,
    *,
    args: Sequence[str] | None = None,
) -> C:
    ...


@typing.overload
def parse_config_from_args(
    config: C,
    /,
    *,
    args: Sequence[str] | None = None,
) -> C:
    ...


def parse_config_from_args(
    config: C | PathOrStr,
    config_class: Type[C] | None = None,
    /,
    *,
    args: Sequence[str] | None = None,
) -> C:
    """
    Parse a config dataclass from command-line args.
    """
    if args is None and sys.argv:
        args = sys.argv[1:]

    config_dict: dict[str, Any] | None = None
    if dataclasses.is_dataclass(config):
        if config_class is None:
            config_class = typing.cast(Type[C], config.__class__)
        else:
            if not isinstance(config, config_class):  # pyright: ignore
                raise ValueError(
                    f"Expected config to be a {config_class}, but got {type(config)} instead"
                )
        config_dict = encode(config)
    elif isinstance(config, (str, Path)):
        if config_class is None:
            raise ValueError(f"config_class is required to infer types for config at '{config}'")
        with open(config) as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(config)

    assert config_dict is not None
    assert config_class is not None
    if args:
        overrides = _clean_opts(args)
        for key, value in overrides:
            _set_nested(config_dict, key, value)

    return decode(config_class, config_dict)


def _set_nested(data: Any, key: str, value: Any):
    if "." in key:
        key, child_keys = key.split(".", 1)
        if isinstance(data, dict):
            _set_nested(data[key], child_keys, value)
        elif isinstance(data, list):
            _set_nested(data[int(key)], child_keys, value)
        else:
            raise ValueError(data)
    else:
        if isinstance(data, dict):
            data[key] = value
        elif isinstance(data, list):
            data[int(key)] = value
        else:
            raise ValueError(data)


def _clean_opts(opts: Sequence[str]) -> list[tuple[str, Any]]:
    return [_clean_opt(s) for s in opts]


def _clean_opt(arg: str) -> tuple[str, Any]:
    if "=" not in arg:
        name, val = arg, "true"
    else:
        name, val = arg.split("=", 1)
    name = name.strip(" -").replace("-", "_")
    val = yaml.safe_load(val)
    return (name, val)
