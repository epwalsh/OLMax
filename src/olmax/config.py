from __future__ import annotations

import dataclasses
import sys
import types
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Generator, Sequence, Type, TypeVar

import yaml

from .types import *

C = TypeVar("C")
R = TypeVar("R", bound="RegistrableConfig")


@dataclass
class RegistrableConfig:
    _registry: ClassVar[dict[str, Type[RegistrableConfig]]]

    type: str | None = dataclasses.field(default=None, repr=False)

    def __new__(cls, *args, type: str | None = None, **kwargs):
        del args, kwargs
        if type is not None and (
            not hasattr(cls, "registered_name") or type != cls.registered_name  # type: ignore
        ):
            if type not in cls._registry:
                raise KeyError(
                    f"'{type}' is not registered name for {cls.__name__}. "
                    f"Available choices are: {list(cls._registry.keys())}"
                )
            return super().__new__(cls._registry[type])
        else:
            return super().__new__(cls)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_choice_registry"):
            cls._registry = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[R]], Type[R]]:
        def register_subclass(subclass: Type[R]) -> Type[R]:
            fields = [
                (f.name, f.type, f) for f in dataclasses.fields(subclass) if f.name != "type"  # type: ignore
            ] + [
                ("registered_name", ClassVar[str], name),  # type: ignore
                ("registered_base", ClassVar[R], cls),  # type: ignore
                ("type", str | None, dataclasses.field(default=name, repr=False)),  # type: ignore
            ]
            subclass = dataclasses.make_dataclass(
                subclass.__name__,
                fields,  # type: ignore
                bases=(subclass,),
            )
            cls._registry[name] = subclass
            return subclass

        return register_subclass

    @classmethod
    def get_registered_name(cls: Type[R], subclass: Type[R] | None = None) -> str:
        if subclass is None:
            subclass = cls
        for name, registered_subclass in cls._registry.items():
            if registered_subclass == subclass:
                return name
        raise ValueError(
            f"class {subclass.__name__} is not a registered subclass of {cls.__name__}"
        )

    @classmethod
    def get_registered_class(cls: Type[R], type: str) -> Type[R]:
        if type not in cls._registry:
            raise KeyError(
                f"'{type}' is not registered name for {cls.__name__}. "
                f"Available choices are: {cls.get_registered_names()}"
            )
        return typing.cast(Type[R], cls._registry[type])

    @classmethod
    def get_registered_names(cls) -> list[str]:
        return list(cls._registry.keys())


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


def parse_config_from_args(
    config_class: Type[C],
    config: C | PathOrStr,
    *,
    args: Sequence[str] | None = None,
) -> C:
    """
    Parse a config dataclass from command-line args.
    """
    if args is None and sys.argv:
        args = sys.argv[1:]

    config_dict: dict[str, Any] | None = None
    if isinstance(config, config_class):
        if not args:
            return config
        else:
            config_dict = encode(config)
    elif isinstance(config, (str, Path)):
        with open(config) as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(config)

    assert config_dict is not None
    if args:
        overrides = _clean_opts(args)
        for key, value in overrides:
            _set_nested(config_dict, key, value)

    return decode(config_class, config_dict)


def encode(
    data: Any,
    *,
    exclude_none: bool = False,
    exclude_private_fields: bool = False,
    json_safe: bool = False,
    recurse: bool = True,
) -> dict[str, Any]:
    """
    Convert into a regular Python dictionary.

    :param exclude_none: Don't include values that are ``None``.
    :param exclude_private_fields: Don't include private fields.
    :param json_safe: Output only JSON-safe types.
    :param recurse: Recurse into fields that are also configs/dataclasses.
    """

    def iter_fields(d) -> Generator[tuple[str, Any], None, None]:
        for field in dataclasses.fields(d):
            value = getattr(d, field.name)
            if exclude_none and value is None:
                continue
            elif exclude_private_fields and field.name.startswith("_"):
                continue
            else:
                yield (field.name, value)

    def as_dict(d: Any, recurse: bool = True) -> Any:
        if dataclasses.is_dataclass(d):
            if recurse:
                out = {k: as_dict(v) for k, v in iter_fields(d)}
            else:
                out = {k: v for k, v in iter_fields(d)}
            if isinstance(d, RegistrableConfig):
                try:
                    registered_name = d.get_registered_name(d.__class__)
                    out["type"] = registered_name
                except ValueError:
                    pass
            return out
        elif isinstance(d, dict):
            return {k: as_dict(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple, set)):
            if json_safe:
                return [as_dict(x) for x in d]
            else:
                return d.__class__((as_dict(x) for x in d))
        elif d is None or isinstance(d, (float, int, bool, str)):
            return d
        elif json_safe:
            if hasattr(d, "__name__"):
                return d.__name__
            else:
                return str(d)
        else:
            return d

    return as_dict(data, recurse=recurse)


def decode(config_class: Type[C], data: dict[str, Any]) -> C:
    type_hints = typing.get_type_hints(config_class)
    kwargs = {k: _coerce(v, type_hints[k]) for k, v in data.items()}
    return config_class(**kwargs)


def _clean_opts(opts: Sequence[str]) -> list[tuple[str, Any]]:
    return [_clean_opt(s) for s in opts]


def _clean_opt(arg: str) -> tuple[str, Any]:
    if "=" not in arg:
        arg = f"{arg}=true"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    val = yaml.safe_load(val)
    return (name, val)


def _get_types(type_hint: Any) -> tuple[Any, ...]:
    if isinstance(type_hint, types.UnionType):
        return type_hint.__args__
    else:
        return (type_hint,)


def _coerce(value: Any, type_hint: Any) -> Any:
    allowed_types = _get_types(type_hint)
    for allowed_type in allowed_types:
        try:
            if isinstance(value, allowed_type):
                return value
            elif issubclass(allowed_type, Enum):
                return allowed_type(value)
        except TypeError:
            pass

        origin = getattr(allowed_type, "__origin__", None)
        args = getattr(allowed_type, "__args__", None)
        if origin is list and isinstance(value, (list, tuple)):
            if args:
                return [_coerce(v, args[0]) for v in value]
            else:
                return list(value)
        elif origin is set and isinstance(value, (list, tuple, set)):
            if args:
                return set(_coerce(v, args[0]) for v in value)
            else:
                return set(value)
        elif origin is tuple and isinstance(value, (list, tuple)):
            if args and ... in args:
                return tuple([_coerce(v, args[0]) for v in value])
            elif args:
                return tuple([_coerce(v, arg) for v, arg in zip(value, args)])
            else:
                return tuple(value)
        elif origin is dict and isinstance(value, dict):
            if args:
                return {_coerce(k, args[0]): _coerce(v, args[1]) for k, v in value.items()}
            else:
                return value
        elif dataclasses.is_dataclass(allowed_type) and isinstance(value, dict):
            type_hints = typing.get_type_hints(allowed_type)
            return allowed_type(**{k: _coerce(v, type_hints[k]) for k, v in value.items()})

    if Any in allowed_types:
        return value

    raise TypeError(f"Cannot coerce {value} to any of {allowed_types} from type hint '{type_hint}'")
