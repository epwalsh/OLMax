from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

from ...config import RegistrableConfig
from ...types import *
from ..attention import MultiheadSelfAttentionConfig
from ..lm_head import LMHeadConfig
from ..normalization import RMSNormConfig
from ..rope import RotaryPositionalEmbeddingConfig
from .block import DefaultTransformerBlockConfig, GemmaTransformerBlockConfig
from .model import DefaultTransformerConfig, TransformerConfig


@dataclass
class TransformerRecipe(RegistrableConfig):
    default_vocab_size: ClassVar[int] = 50_304
    default_learning_rate: ClassVar[float] = 1e-3
    default_sequence_length: ClassVar[int] = 4096

    @classmethod
    @abstractmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_config(
        cls,
        vocab_size: int | None,
        param_dtype: DTypeLike = float,
        attn_window_size: int | tuple[int, int] | None = None,
        attn_implementation: Literal["xla", "cudnn"] | None = None,
    ) -> TransformerConfig:
        raise NotImplementedError


@TransformerRecipe.register_subclass("llama_like_271M")
@dataclass
class LlamaLike271MRecipe(TransformerRecipe):
    default_learning_rate: ClassVar[float] = 1e-3
    default_sequence_length: ClassVar[int] = 1024

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_H100:
            return 32 * 1024
        elif device_type == GPUType.NVIDIA_B200:
            return 64 * 1024
        else:
            return 16 * 1024

    @classmethod
    def build_config(
        cls,
        vocab_size: int | None,
        param_dtype: DTypeLike = float,
        attn_window_size: int | tuple[int, int] | None = None,
        attn_implementation: Literal["xla", "cudnn"] | None = None,
    ) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=vocab_size or cls.default_vocab_size,
            d_model=1024,
            hidden_size=2816,
            num_layers=16,
            block=DefaultTransformerBlockConfig(
                attention=MultiheadSelfAttentionConfig(
                    n_heads=8,
                    rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                    bias=False,
                    dtype=param_dtype,
                    window_size=attn_window_size,
                    implementation=attn_implementation,
                ),
                norm=norm,
                bias=False,
                dtype=param_dtype,
            ),
            lm_head=LMHeadConfig(norm=norm, bias=False, dtype=param_dtype),
            dtype=param_dtype,
        )


@TransformerRecipe.register_subclass("llama_like_7B")
@dataclass
class LlamaLike7BRecipe(TransformerRecipe):
    default_learning_rate: ClassVar[float] = 1e-4

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_H100:
            return 2 * 4096
        elif device_type == GPUType.NVIDIA_B200:
            return 4 * 4096
        else:
            return 1 * 4096

    @classmethod
    def build_config(
        cls,
        vocab_size: int | None = None,
        param_dtype: DTypeLike = float,
        attn_window_size: int | tuple[int, int] | None = None,
        attn_implementation: Literal["xla", "cudnn"] | None = None,
    ) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=vocab_size or cls.default_vocab_size,
            d_model=4096,
            hidden_size=11008,
            num_layers=32,
            block=DefaultTransformerBlockConfig(
                attention=MultiheadSelfAttentionConfig(
                    n_heads=32,
                    rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                    bias=False,
                    dtype=param_dtype,
                    window_size=attn_window_size,
                    implementation=attn_implementation,
                ),
                norm=norm,
                bias=False,
                dtype=param_dtype,
            ),
            lm_head=LMHeadConfig(norm=norm, bias=False, dtype=param_dtype),
            dtype=param_dtype,
        )


@TransformerRecipe.register_subclass("gemma2_like_27B")
@dataclass
class Gemma2Like27BRecipe(TransformerRecipe):
    default_vocab_size: ClassVar[int] = 256000
    default_learning_rate: ClassVar[float] = 1e-5

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_B200:
            return 2 * 4096
        else:
            return 1 * 4096

    @classmethod
    def build_config(
        cls,
        vocab_size: int | None = None,
        param_dtype: DTypeLike = float,
        attn_window_size: int | tuple[int, int] | None = None,
        attn_implementation: Literal["xla", "cudnn"] | None = None,
    ) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=vocab_size or cls.default_vocab_size,
            d_model=4608,
            hidden_size=36864,
            num_layers=46,
            block=GemmaTransformerBlockConfig(
                attention=MultiheadSelfAttentionConfig(
                    n_heads=32,
                    n_kv_heads=16,
                    head_dim=128,
                    rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                    bias=False,
                    dtype=param_dtype,
                    window_size=attn_window_size,
                    implementation=attn_implementation,
                ),
                norm=norm,
                bias=False,
                dtype=param_dtype,
            ),
            lm_head=LMHeadConfig(norm=norm, bias=False, dtype=param_dtype),
            dtype=param_dtype,
        )


@TransformerRecipe.register_subclass("gemma3_like_27B")
@dataclass
class Gemma3Like27BRecipe(TransformerRecipe):
    default_vocab_size: ClassVar[int] = 256000
    default_learning_rate: ClassVar[float] = 1e-5

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_B200:
            return 2 * 4096
        else:
            return 1 * 4096

    @classmethod
    def build_config(
        cls,
        vocab_size: int | None = None,
        param_dtype: DTypeLike = float,
        attn_window_size: int | tuple[int, int] | None = None,
        attn_implementation: Literal["xla", "cudnn"] | None = None,
    ) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=vocab_size or cls.default_vocab_size,
            d_model=5376,
            hidden_size=21504,
            num_layers=62,
            block=GemmaTransformerBlockConfig(
                attention=MultiheadSelfAttentionConfig(
                    n_heads=32,
                    n_kv_heads=16,
                    head_dim=128,
                    rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                    bias=False,
                    dtype=param_dtype,
                    window_size=attn_window_size,
                    implementation=attn_implementation,
                    qk_norm=norm,
                    qk_norm_headwise=True,
                ),
                norm=norm,
                bias=False,
                dtype=param_dtype,
            ),
            lm_head=LMHeadConfig(norm=norm, bias=False, dtype=param_dtype),
            dtype=param_dtype,
        )
