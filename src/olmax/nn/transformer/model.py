from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

import jax

from ...distributed.parallel import ParallelConfig
from ...types import Array, DTypeLike, PRNGKeyArray
from ..embedding import Embedding
from ..lm_head import LMHead, LMHeadConfig
from ..module import Module
from .block import TransformerBlock, TransformerBlockConfig


@dataclass
class TransformerConfig:
    d_model: int
    hidden_size: int
    vocab_size: int
    num_layers: int
    block: TransformerBlockConfig
    lm_head: LMHeadConfig
    dtype: DTypeLike = float

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
        parallel_config: ParallelConfig | None = None,
    ) -> Transformer:
        return Transformer(
            key=key,
            d_model=d_model if d_model is not None else self.d_model,
            vocab_size=vocab_size if vocab_size is not None else self.vocab_size,
            hidden_size=hidden_size if hidden_size is not None else self.hidden_size,
            num_layers=num_layers if num_layers is not None else self.num_layers,
            block=block if block is not None else self.block,
            lm_head=lm_head if lm_head is not None else self.lm_head,
            dtype=dtype if dtype is not None else self.dtype,
            parallel_config=parallel_config,
        )


class Transformer(Module):
    Config: ClassVar[Type[TransformerConfig]] = TransformerConfig
    keepdims: ClassVar[int] = -1

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
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__(parallel_config)
        emb_key, blocks_key, lm_head_key = jax.random.split(key, 3)
        self.embedding = Embedding(
            d_model, vocab_size, emb_key, dtype=dtype, parallel_config=parallel_config
        )
        self.blocks = []
        for block_idx in range(num_layers):
            block_key = jax.random.fold_in(blocks_key, block_idx)
            self.blocks.append(
                block.build(
                    d_model, hidden_size, block_key, dtype=dtype, parallel_config=parallel_config
                )
            )
        self.lm_head = lm_head.build(
            d_model,
            vocab_size,
            lm_head_key,
            dtype=dtype,
            parallel_config=parallel_config,
        )

    @jax.named_scope("olmax.nn.Transformer")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 2  # shape: (batch_size, seq_len)

        # shape: (seq_len, d_model)
        h = self.embedding(x)

        for block in self.blocks:
            # shape: (seq_len, d_model)
            h = block(h)

        # shape: (seq_len, vocab_size)
        out = self.lm_head(h)
        return out
