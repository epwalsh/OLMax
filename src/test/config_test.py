from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass

import pytest

from olmax.config import (
    Registrable,
    decode,
    encode,
    parse_config_from_args,
    required_field,
)
from olmax.types import *


@dataclass
class Foo:
    x: int


def test_required_field():
    @dataclass
    class Config:
        foo: Foo = required_field("foo", strict=True)

    with pytest.raises(ValueError, match="missing required field 'foo'"):
        Config()

    config = Config(foo=Foo(x=1))
    assert isinstance(config.foo, Foo)
    assert config.foo.x == 1


def test_registrable_class():
    @dataclass
    class BaseType(Registrable):
        x: int
        y: int = -1
        z: int = -1

    @BaseType.register("bar")
    @dataclass
    class SubType(BaseType):
        w: int = 2

    assert SubType.registered_base == BaseType  # type: ignore
    assert SubType.registered_name == "bar"  # type: ignore
    assert SubType.get_registered_name() == "bar"

    assert not isinstance(BaseType(x=0), SubType)
    assert isinstance(BaseType(x=0, type="bar"), SubType)
    assert encode(SubType(x=0)) == {"x": 0, "y": -1, "z": -1, "w": 2, "type": "bar"}


def test_decode_with_a_variety_of_required_complex_types():
    @dataclass
    class Config:
        foo: Foo
        bar: Foo | None
        lr: float
        path: PathOrStr
        set_data: set[str]
        list_data: typing.List[Foo]
        fixed_tuple: tuple[int, int]
        indefinite_tuple: tuple[Foo, ...]

    config = decode(
        Config,
        {
            "foo": {"x": 0},
            "bar": None,
            "lr": 0.0,
            "path": "/path",
            "set_data": ["a", "b", "c"],
            "list_data": [{"x": 0}],
            "fixed_tuple": [0, 1],
            "indefinite_tuple": [{"x": -1}],
        },
    )
    assert isinstance(config, Config)
    assert config.set_data == {"a", "b", "c"}
    assert all(isinstance(v, Foo) for v in config.list_data)
    assert all(isinstance(v, Foo) for v in config.indefinite_tuple)


def test_decode_with_a_variety_of_optional_complex_types():
    @dataclass
    class Config:
        foo: Foo | None = None
        lr: float = dataclasses.field(default=0.0)
        path: PathOrStr | None = None
        set_data: set[str] = dataclasses.field(default_factory=set)
        list_data: typing.List[Foo] = dataclasses.field(default_factory=list)
        fixed_tuple: tuple[int, int] = dataclasses.field(default_factory=lambda: (0, 0))
        indefinite_tuple: tuple[Foo, ...] = dataclasses.field(default_factory=tuple)

    config = decode(
        Config,
        {
            "foo": {"x": 0},
            "lr": 0.0,
            "path": None,
            "set_data": ["a", "b", "c"],
            "list_data": [{"x": 0}],
            "fixed_tuple": [0, 1],
            "indefinite_tuple": [{"x": -1}],
        },
    )
    assert isinstance(config, Config)
    assert config.set_data == {"a", "b", "c"}
    assert config.path is None
    assert all(isinstance(v, Foo) for v in config.list_data)
    assert all(isinstance(v, Foo) for v in config.indefinite_tuple)


def test_parse_config_from_args_with_a_variety_of_complex_types():
    @dataclass
    class Config:
        foo: Foo
        bar: Foo | None
        lr: float
        path: PathOrStr
        list_data: typing.List[Foo]
        tuple_data: tuple[int, int]

    config = parse_config_from_args(
        Config(
            foo=Foo(x=0),
            bar=None,
            lr=0.0,
            path="/path",
            list_data=[Foo(x=2)],
            tuple_data=(0, 1),
        ),
        args=["foo.x=1", "list_data.0.x=1"],
    )
    assert isinstance(config, Config)
    assert config.foo.x == 1
    assert config.list_data[0].x == 1
