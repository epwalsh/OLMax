from __future__ import annotations

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Type, TypeVar

import jax

from ...config import RegistrableConfig
from ...distributed.parallel import MeshResource
from ...types import Array, DTypeLike, PRNGKeyArray
from ..embedding import Embedding
from ..lm_head import LMHead, LMHeadConfig
from ..module import Module
from .block import TransformerBlock, TransformerBlockConfig


class Transformer(Module):
    embedding: Embedding
    blocks: list[TransformerBlock]
    lm_head: LMHead

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        block: TransformerBlockConfig,
        lm_head: LMHeadConfig,
        key: PRNGKeyArray,
        dtype: DTypeLike = float,
        mesh_resource: MeshResource | None = None,
    ):
        super().__init__(mesh_resource)
        emb_key, blocks_key, lm_head_key = jax.random.split(key, 3)
        self.embedding = Embedding(
            d_model, vocab_size, emb_key, dtype=dtype, mesh_resource=mesh_resource
        )
        self.blocks = []
        for block_idx in range(num_layers):
            block_key = jax.random.fold_in(blocks_key, block_idx)
            self.blocks.append(
                block.build(
                    d_model, hidden_size, block_key, dtype=dtype, mesh_resource=mesh_resource
                )
            )
        self.lm_head = lm_head.build(
            d_model,
            vocab_size,
            lm_head_key,
            dtype=dtype,
            mesh_resource=mesh_resource,
        )

    @classmethod
    def Config(cls, **kwargs) -> TransformerConfig:
        return DefaultTransformerConfig(**kwargs)

    @jax.named_scope("olmax.nn.Transformer")
    def __call__(self, x: Array) -> Array:
        assert x.ndim == 2  # shape: (batch_size, seq_len)

        # shape: (seq_len, d_model)
        h = self.embedding(x)

        for block in self.blocks:
            # shape: (seq_len, d_model)
            h = block(h)

        # shape: (seq_len, vocab_size)
        out = self.lm_head(h)
        return out


T = TypeVar("T", bound=Transformer)


@dataclass
class TransformerConfig(RegistrableConfig, Generic[T]):
    d_model: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    num_layers: int = 0
    block: TransformerBlockConfig = dataclasses.field(default_factory=TransformerBlockConfig)
    lm_head: LMHeadConfig = dataclasses.field(default_factory=LMHeadConfig)
    dtype: DTypeLike = float

    @classmethod
    @abstractmethod
    def get_class(cls) -> Type[T]:
        raise NotImplementedError

    @classmethod
    def Default(cls, **kwargs) -> DefaultTransformerConfig:
        return DefaultTransformerConfig(**kwargs)

    def build(
        self,
        key: PRNGKeyArray,
        *,
        d_model: int | None = None,
        vocab_size: int | None = None,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        block: TransformerBlockConfig | None = None,
        lm_head: LMHeadConfig | None = None,
        dtype: DTypeLike | None = None,
        mesh_resource: MeshResource | None = None,
    ) -> T:
        return self.get_class()(
            key=key,
            d_model=d_model if d_model is not None else self.d_model,
            vocab_size=vocab_size if vocab_size is not None else self.vocab_size,
            hidden_size=hidden_size if hidden_size is not None else self.hidden_size,
            num_layers=num_layers if num_layers is not None else self.num_layers,
            block=block if block is not None else self.block,
            lm_head=lm_head if lm_head is not None else self.lm_head,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
        )


@TransformerConfig.register("default")
@dataclass
class DefaultTransformerConfig(TransformerConfig[Transformer]):
    @classmethod
    def get_class(cls) -> Type[Transformer]:
        return Transformer
