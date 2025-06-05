from abc import ABCMeta, abstractmethod
from enum import StrEnum
from typing import ClassVar, Literal, Type

from ...types import DTypeLike
from ..attention import MultiheadSelfAttentionConfig
from ..lm_head import LMHeadConfig
from ..normalization import RMSNormConfig
from ..rope import RotaryPositionalEmbeddingConfig
from .block import DefaultTransformerBlockConfig, GemmaTransformerBlockConfig
from .model import DefaultTransformerConfig, TransformerConfig


class TransformerRecipe(metaclass=ABCMeta):
    default_vocab_size: ClassVar[int] = 50_304
    default_learning_rate: ClassVar[float] = 1e-3
    default_sequence_length: ClassVar[int] = 4096

    @classmethod
    @abstractmethod
    def get_mbz_per_device(cls, device_type: str) -> int:
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


class LlamaLike271MRecipe(TransformerRecipe):
    default_learning_rate: ClassVar[float] = 1e-3
    default_sequence_length: ClassVar[int] = 1024

    @classmethod
    def get_mbz_per_device(cls, device_type: str) -> int:
        device_type = device_type.lower()
        mbz = 16 * 1024
        if "h100" in device_type:
            mbz *= 2
        elif "b200" in device_type:
            mbz *= 4
        return mbz

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


class LlamaLike7BRecipe(TransformerRecipe):
    default_learning_rate: ClassVar[float] = 1e-4

    @classmethod
    def get_mbz_per_device(cls, device_type: str) -> int:
        device_type = device_type.lower()
        mbz = 2 * 4096
        if "b200" in device_type:
            mbz *= 2
        return mbz

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


class Gemma2Like27BRecipe(TransformerRecipe):
    default_vocab_size: ClassVar[int] = 256000
    default_learning_rate: ClassVar[float] = 1e-5

    @classmethod
    def get_mbz_per_device(cls, device_type: str) -> int:
        del device_type
        mbz = 1 * 4096
        return mbz

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


class Gemma3Like27BRecipe(TransformerRecipe):
    default_vocab_size: ClassVar[int] = 256000
    default_learning_rate: ClassVar[float] = 1e-5

    @classmethod
    def get_mbz_per_device(cls, device_type: str) -> int:
        del device_type
        mbz = 1 * 4096
        return mbz

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


class TransformerRecipeName(StrEnum):
    llama_like_271M = "llama_like_271M"
    llama_like_7B = "llama_like_7B"
    gemma2_like_27B = "gemma2_like_27B"
    gemma3_like_27B = "gemma3_like_27B"

    def get_recipe(self) -> Type[TransformerRecipe]:
        if self == self.llama_like_271M:
            return LlamaLike271MRecipe
        elif self == self.llama_like_7B:
            return LlamaLike7BRecipe
        elif self == self.gemma2_like_27B:
            return Gemma2Like27BRecipe
        elif self == self.gemma3_like_27B:
            return Gemma3Like27BRecipe
        else:
            raise NotImplementedError(self)
