from __future__ import annotations

import typing
from dataclasses import dataclass

import yaml
from dataclass_extensions import Registrable, encode

from olmax.config import parse_config_from_args
from olmax.types import *


@dataclass
class Foo:
    x: int


def test_parse_config_from_args_with_a_variety_of_complex_types():
    @dataclass
    class Config:
        foo: Foo
        bar: Foo | None
        lr: float
        path: PathOrStr
        list_data: typing.List[Foo]
        tuple_data: tuple[int, int]
        sequence_or_int: int | typing.Sequence[int] | None

    config = parse_config_from_args(
        Config(
            foo=Foo(x=0),
            bar=None,
            lr=0.0,
            path="/path",
            list_data=[Foo(x=2)],
            tuple_data=(0, 1),
            sequence_or_int=2,
        ),
        args=["foo.x=1", "list_data.0.x=1", "sequence_or_int=[0, 1]"],
    )
    assert isinstance(config, Config)
    assert config.foo.x == 1
    assert config.list_data[0].x == 1
    assert config.sequence_or_int == (0, 1)


@dataclass
class Fruit(Registrable):
    calories: int
    price: float


@Fruit.register("banana")
@dataclass
class Banana(Fruit):
    calories: int = 200
    price: float = 1.25


@Fruit.register("apple")
@dataclass
class Apple(Fruit):
    calories: int = 150
    price: float = 1.50


@dataclass
class FruitBasket:
    fruit: Fruit
    count: int


def test_parse_config_from_args_with_registrable_overrides(tmp_path):
    basket1 = FruitBasket(fruit=Banana(), count=2)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(encode(basket1), f)

    # When we only override the type name, the other fields will remain unchanged.
    for basket2 in (
        parse_config_from_args(basket1, args=["--fruit.type=apple"]),
        parse_config_from_args(config_path, FruitBasket, args=["--fruit.type=apple"]),
    ):
        assert isinstance(basket2.fruit, Apple)
        assert basket2.fruit.price == basket1.fruit.price

    # But we override the whole type, the defaults will follow from the selected type.
    for basket2 in (
        parse_config_from_args(basket1, args=["--fruit={type: apple}"]),
        parse_config_from_args(config_path, FruitBasket, args=["--fruit={type: apple}"]),
    ):
        assert isinstance(basket2.fruit, Apple)
        assert basket2.fruit.price == Apple().price
