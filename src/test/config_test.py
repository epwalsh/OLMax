from __future__ import annotations

import typing
from dataclasses import dataclass

from olmax.config import Registrable, decode, encode, parse_config_from_args
from olmax.types import *


def test_registrable_class():
    @dataclass
    class FooConfig(Registrable):
        x: int = -1
        y: int = -1
        z: int = -1

        @classmethod
        def zeros(cls) -> FooConfig:
            return cls(x=0, y=0, z=0)

    @FooConfig.register("bar")
    @dataclass
    class BarConfig(FooConfig):
        w: int = 2

    assert BarConfig.registered_base == FooConfig  # type: ignore
    assert BarConfig.registered_name == "bar"  # type: ignore
    assert BarConfig.get_registered_name() == "bar"

    assert not isinstance(FooConfig(), BarConfig)
    assert isinstance(FooConfig(type="bar"), BarConfig)
    assert encode(BarConfig()) == {"x": -1, "y": -1, "z": -1, "w": 2, "type": "bar"}


@dataclass
class Config:
    foo: Foo
    bar: Bar | None
    lr: float
    path: PathOrStr
    list_data: typing.List[Bar]
    tuple_data: tuple[int, int]


@dataclass
class Foo:
    x: int


@dataclass
class Bar:
    y: int


def test_decode():
    config = decode(
        Config,
        {
            "foo": {"x": 0},
            "bar": None,
            "lr": 0.0,
            "path": "/path",
            "list_data": [{"y": 0}],
            "tuple_data": [0, 1],
        },
    )
    assert isinstance(config, Config)
    assert all(isinstance(v, Bar) for v in config.list_data)


def test_parse_config_from_args():
    config = parse_config_from_args(
        Config,
        Config(
            foo=Foo(x=0),
            bar=None,
            lr=0.0,
            path="/path",
            list_data=[Bar(y=2)],
            tuple_data=(0, 1),
        ),
        args=["foo.x=1", "list_data.0.y=1"],
    )
    assert isinstance(config, Config)
    assert config.foo.x == 1
    assert config.list_data[0].y == 1
