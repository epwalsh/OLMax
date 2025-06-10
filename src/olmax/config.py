from __future__ import annotations

import collections.abc
import dataclasses
import pathlib
import sys
import types
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Generator, Sequence, Type, TypeVar

import numpy as np
import yaml

from .types import *

C = TypeVar("C", bound=Dataclass)
R = TypeVar("R", bound="Registrable")
T = TypeVar("T")


MISSING = object()


@typing.overload
def required_field(name: str, *, strict: bool = False) -> T:  # type: ignore[type-var]
    ...


@typing.overload
def required_field() -> T:  # type: ignore[type-var]
    ...


def required_field(name: str | None = None, *, strict: bool = False, _: Type[T] | None = None) -> T:
    """
    Can be used in place of ``dataclasses.field()`` to mark a field required when non-default
    fields are not allowed.
    """
    if strict:
        if name is None:
            raise ValueError("'name' is required for a required_field with 'strict=True'")

        def err_out():
            raise ValueError(f"missing required field '{name}'")

        return typing.cast(T, dataclasses.field(default_factory=err_out))

    return typing.cast(T, dataclasses.field(default=MISSING))


@dataclass
class Registrable:
    _registry: ClassVar[dict[str, Type[Registrable]]]

    type: dataclasses.InitVar[str | None] = dataclasses.field(
        default=None, kw_only=True, repr=False
    )

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
            if not issubclass(subclass, cls):
                raise TypeError(
                    f"class {subclass.__name__} must be a subclass of {cls.__name__} in order to register it"
                )
            if not dataclasses.is_dataclass(subclass):
                raise TypeError(
                    f"class {subclass.__name__} must be a dataclass in order to register it"
                )

            fields = [
                (f.name, f.type, f) for f in dataclasses.fields(subclass) if f.name != "type"  # type: ignore
            ] + [
                ("registered_name", ClassVar[str], name),  # type: ignore
                ("registered_base", ClassVar[R], cls),  # type: ignore
                ("type", dataclasses.InitVar[str | None], dataclasses.field(default=name, kw_only=True, repr=False)),  # type: ignore
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
            if hasattr(cls, "registered_name"):
                return cls.registered_name  # type: ignore
            else:
                raise ValueError(
                    f"class {cls.__name__} is not a registered subclass of any base registrable class"
                )

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


class Encoder:
    custom_handlers: ClassVar[dict[Type, Callable[[Any], Any]]] = {}

    def register_encoder(self, encoder_fun: Callable[[Any], Any], *types: Type):
        for type in types:
            self.custom_handlers[type] = encoder_fun

    def __call__(
        self,
        data: Any,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        recurse: bool = True,
        strict: bool = True,
    ) -> Any:
        """
        Encode a Python object into JSON-safe dictionary. The inverse of :func:`decode()`.

        :param exclude_none: Don't include values that are ``None``.
        :param exclude_private_fields: Don't include private fields.
        :param recurse: Recurse into fields that are also configs/dataclasses.
        :param strict: If ``True`` a ``TypeError`` is raised when a type is encountered that doesn't
            have a safe encoding method. Otherwise ``str(value)`` is used.
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
            if type(d) in self.custom_handlers:
                return self.custom_handlers[type(d)](d)
            elif dataclasses.is_dataclass(d):
                if recurse:
                    out = {k: as_dict(v) for k, v in iter_fields(d)}
                else:
                    out = {k: v for k, v in iter_fields(d)}
                if isinstance(d, Registrable):
                    try:
                        registered_name = d.get_registered_name()
                        out["type"] = registered_name
                    except ValueError:
                        pass
                return out
            elif isinstance(d, dict):
                return {k: as_dict(v) for k, v in d.items()}
            elif isinstance(d, (list, tuple, set)):
                return [as_dict(x) for x in d]
            elif d is None or isinstance(d, (float, int, bool, str)):
                return d

            for t, h in self.custom_handlers.items():
                try:
                    if isinstance(d, t):
                        return h(d)
                except TypeError:
                    continue

            if strict:
                raise TypeError(f"not sure how to encode '{d}' of type {type(d)}")
            else:
                return str(d)

        return as_dict(data, recurse=recurse)


encode = Encoder()
encode.register_encoder(str, pathlib.Path, np.dtype)


class Decoder:
    custom_handlers: ClassVar[dict[Any, Callable[[Any], Any]]] = {}

    def register_decoder(self, encoder_fun: Callable[[Any], Any], *types: Any):
        for type in types:
            self.custom_handlers[type] = encoder_fun

    def __call__(self, config_class: Type[C], data: dict[str, Any]) -> C:
        """
        Decode a dataset from a JSON-safe dictionary. The inverse of :func:`encode()`.
        """
        type_hints = typing.get_type_hints(config_class)
        kwargs = {k: _coerce(v, type_hints[k], self.custom_handlers, k) for k, v in data.items()}
        return config_class(**kwargs)


decode = Decoder()


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


def _get_types(type_hint: Any) -> tuple[Any, ...]:
    # NOTE: 'types.UnionType' doesn't cover union types with 'typing.*' types.
    if _safe_isinstance(type_hint, (types.UnionType, type(typing.List | None))):
        return type_hint.__args__
    elif _safe_isinstance(type_hint, dataclasses.InitVar):
        return _get_types(type_hint.type)
    # TypeAliasType added in 3.12
    elif hasattr(typing, "TypeAliasType") and _safe_isinstance(type_hint, typing.TypeAliasType):  # type: ignore
        return _get_types(type_hint.__value__)
    else:
        return (type_hint,)


def _safe_isinstance(a, b) -> bool:
    try:
        return isinstance(a, b)
    except TypeError:
        return False


def _safe_issubclass(a, b) -> bool:
    try:
        return issubclass(a, b)
    except TypeError:
        return False


def _coerce(
    value: Any, type_hint: Any, custom_handlers: dict[Any, Callable[[Any], Any]], key: str
) -> Any:
    if value is MISSING:
        raise ValueError(f"Missing required field at '{key}'")

    if type_hint in custom_handlers:
        return custom_handlers[type_hint](value)

    allowed_types = _get_types(type_hint)
    for allowed_type in allowed_types:
        if allowed_type in custom_handlers:
            return custom_handlers[allowed_type](value)

        if _safe_isinstance(value, allowed_type):
            return value

        if _safe_issubclass(allowed_type, Enum):
            try:
                return allowed_type(value)
            except TypeError:
                pass

        # e.g. typing.NamedTuple
        if _safe_issubclass(allowed_type, tuple) and _safe_isinstance(value, (list, tuple)):
            try:
                return allowed_type(*value)
            except TypeError:
                pass

        origin = getattr(allowed_type, "__origin__", None)
        args = getattr(allowed_type, "__args__", None)
        if (origin is list or origin is collections.abc.MutableSequence) and _safe_isinstance(
            value, (list, tuple)
        ):
            if args:
                return [
                    _coerce(v, args[0], custom_handlers, f"{key}.{i}") for i, v in enumerate(value)
                ]
            else:
                return list(value)
        elif (
            origin is set or origin is collections.abc.Set or origin is collections.abc.MutableSet
        ) and _safe_isinstance(value, (list, tuple, set)):
            if args:
                return set(
                    _coerce(v, args[0], custom_handlers, f"{key}.{i}") for i, v in enumerate(value)
                )
            else:
                return set(value)
        elif origin is collections.abc.Sequence and _safe_isinstance(value, (list, tuple)):
            if args:
                return tuple(
                    [
                        _coerce(v, args[0], custom_handlers, f"{key}.{i}")
                        for i, v in enumerate(value)
                    ]
                )
            else:
                return tuple(value)
        elif origin is tuple and _safe_isinstance(value, (list, tuple)):
            if args and ... in args:
                return tuple(
                    [
                        _coerce(v, args[0], custom_handlers, f"{key}.{i}")
                        for i, v in enumerate(value)
                    ]
                )
            elif args:
                return tuple(
                    [
                        _coerce(v, arg, custom_handlers, f"{key}.{i}")
                        for i, (v, arg) in enumerate(zip(value, args))
                    ]
                )
            else:
                return tuple(value)
        elif (
            origin is dict
            or origin is collections.abc.Mapping
            or origin is collections.abc.MutableMapping
        ) and _safe_isinstance(value, dict):
            if args:
                return {
                    _coerce(k, args[0], custom_handlers, f"{key}.{k}"): _coerce(
                        v, args[1], custom_handlers, f"{key}.{k}"
                    )
                    for k, v in value.items()
                }
            else:
                return value
        elif origin is typing.Literal and args and value in args:
            return value
        elif (
            dataclasses.is_dataclass(allowed_type)
            # e.g. TypedDict
            or _safe_issubclass(allowed_type, dict)
        ) and _safe_isinstance(value, dict):
            type_hints = typing.get_type_hints(allowed_type)
            kwargs = {}
            for k, v in value.items():
                try:
                    type_hint = type_hints[k]
                except KeyError as e:
                    raise KeyError(
                        f"type {allowed_type} has no field '{k}' (full key '{key}.{k}')"
                    ) from e
                kwargs[k] = _coerce(v, type_hint, custom_handlers, f"{key}.{k}")
            return allowed_type(**kwargs)

    if Any in allowed_types:
        return value

    raise TypeError(
        f"Not sure how to coerce value {value} at key '{key}' to any "
        f"of {allowed_types} from type hint '{type_hint}'"
    )
