from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Type, TypeVar

import equinox as eqx
import jax
from dataclass_extensions import Registrable

from ...activation_checkpointing import (
    ActivationCheckpointingPolicy,
    NamedCheckpointPolicy,
)
from ...distributed.parallel import MeshResource
from ...types import Array, DTypeLike, PRNGKeyArray
from ..embedding import Embedding
from ..functional import scan_module
from ..lm_head import LMHead, LMHeadConfig
from ..module import Module
from ..normalization import Normalizer, NormalizerConfig
from .layer import TransformerLayer, TransformerLayerConfig


class Transformer(Module):
    embedding: Embedding
    layers: list[TransformerLayer]
    norm: Normalizer
    lm_head: LMHead
    scan_layers: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        layer: TransformerLayerConfig,
        norm: NormalizerConfig,
        lm_head: LMHeadConfig,
        key: PRNGKeyArray,
        dtype: DTypeLike = float,
        scan_layers: bool = False,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
        ac_policy: ActivationCheckpointingPolicy | None = None,
    ):
        super().__init__(mesh_resource, checkpoint_name)
        emb_key, layers_key, norm_key, lm_head_key = jax.random.split(key, 4)
        self.embedding = Embedding(
            d_model,
            vocab_size,
            emb_key,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name="embedding"
            if checkpoint_name is None
            else f"{checkpoint_name}.embedding",
        )
        self.layers = layer.build_all(
            d_model,
            hidden_size,
            num_layers,
            layers_key,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=checkpoint_name,
            scan_layers=scan_layers,
            ac_policy=ac_policy,
        )
        self.norm = norm.build(
            d_model,
            norm_key,
            mesh_resource=mesh_resource,
            checkpoint_name="norm" if checkpoint_name is None else f"{checkpoint_name}.norm",
        )
        self.lm_head = lm_head.build(
            d_model,
            vocab_size,
            lm_head_key,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name="lm_head" if checkpoint_name is None else f"{checkpoint_name}.lm_head",
        )
        self.scan_layers = scan_layers
        if isinstance(ac_policy, NamedCheckpointPolicy):
            ac_policy.resolve_names(self.get_checkpoint_names())

    @classmethod
    def Config(cls, **kwargs) -> TransformerConfig:
        return DefaultTransformerConfig(**kwargs)

    @jax.named_scope("olmax.nn.Transformer")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 2  # shape: (batch_size, seq_len)

        # shape: (batch, seq_len, d_model)
        h = self.embedding(x)

        if self.scan_layers:
            layer = self.layers[0]
            h = scan_module(
                layer,
                h,
                input_sharding=None
                if self.mesh_resource is None
                else self.mesh_resource.get_data_sharding(),
                output_sharding=None
                if self.mesh_resource is None
                else self.mesh_resource.get_data_sharding(),
                #  param_sharding=None
                #  if self.mesh_resource is None
                #  else self.mesh_resource.get_data_sharding(),  # TODO: fix this
            )
        else:
            for layer in self.layers:
                # shape: (batch_size, seq_len, d_model)
                h = layer(h)

        # shape: (batch_size, seq_len, d_model)
        h = self.norm(h)

        # shape: (batch_size, seq_len, vocab_size)
        out = self.lm_head(h)
        return out


T = TypeVar("T", bound=Transformer)


@dataclass
class TransformerConfig(Registrable, Generic[T]):
    d_model: int
    hidden_size: int
    vocab_size: int
    num_layers: int
    layer: TransformerLayerConfig
    norm: NormalizerConfig
    lm_head: LMHeadConfig
    dtype: DTypeLike = float
    scan_layers: bool = False
    ac_policy: ActivationCheckpointingPolicy | None = None

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
        layer: TransformerLayerConfig | None = None,
        norm: NormalizerConfig | None = None,
        lm_head: LMHeadConfig | None = None,
        dtype: DTypeLike | None = None,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
        scan_layers: bool | None = None,
        ac_policy: ActivationCheckpointingPolicy | None = None,
    ) -> T:
        return self.get_class()(
            key=key,
            d_model=d_model if d_model is not None else self.d_model,
            vocab_size=vocab_size if vocab_size is not None else self.vocab_size,
            hidden_size=hidden_size if hidden_size is not None else self.hidden_size,
            num_layers=num_layers if num_layers is not None else self.num_layers,
            layer=layer if layer is not None else self.layer,
            norm=norm if norm is not None else self.norm,
            lm_head=lm_head if lm_head is not None else self.lm_head,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=checkpoint_name,
            scan_layers=scan_layers if scan_layers is not None else self.scan_layers,
            ac_policy=ac_policy if ac_policy is not None else self.ac_policy,
        )


@TransformerConfig.register("default")
@dataclass
class DefaultTransformerConfig(TransformerConfig[Transformer]):
    @classmethod
    def get_class(cls) -> Type[Transformer]:
        return Transformer
