from abc import abstractmethod
from dataclasses import dataclass

from ...config import EnvConfig, RegistrableConfig
from ...types import *
from ..attention import MultiheadSelfAttentionConfig
from ..lm_head import LMHeadConfig
from ..normalization import RMSNormConfig
from ..rope import RotaryPositionalEmbeddingConfig
from .block import (
    DefaultTransformerBlockConfig,
    GemmaTransformerBlockConfig,
    ReorderedNormTransformerBlockConfig,
)
from .model import DefaultTransformerConfig, TransformerConfig


@dataclass
class TransformerRecipe(RegistrableConfig):
    vocab_size: int
    learning_rate: float
    sequence_length: int

    @classmethod
    @abstractmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def build_config(self, param_dtype: DTypeLike = float) -> TransformerConfig:
        raise NotImplementedError

    def set_env_defaults(self, env: EnvConfig):
        del env


@TransformerRecipe.register_subclass("llama_like_271M")
@dataclass
class LlamaLike271MRecipe(TransformerRecipe):
    vocab_size: int = 50_304
    learning_rate: float = 1e-3
    sequence_length: int = 1024

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_H100:
            return 32 * 1024
        elif device_type == GPUType.NVIDIA_B200:
            return 64 * 1024
        else:
            return 16 * 1024

    def build_config(self, param_dtype: DTypeLike = float) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=self.vocab_size,
            d_model=1024,
            hidden_size=2816,
            num_layers=16,
            block=DefaultTransformerBlockConfig(
                attention=MultiheadSelfAttentionConfig(
                    n_heads=8,
                    rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                    bias=False,
                    dtype=param_dtype,
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
    vocab_size: int = 50_304
    learning_rate: float = 1e-4
    sequence_length: int = 4096

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_H100:
            return 2 * 4096
        elif device_type == GPUType.NVIDIA_B200:
            return 4 * 4096
        else:
            return 1 * 4096

    def build_config(
        self,
        param_dtype: DTypeLike = float,
    ) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=self.vocab_size,
            d_model=4096,
            hidden_size=11008,
            num_layers=32,
            block=DefaultTransformerBlockConfig(
                attention=MultiheadSelfAttentionConfig(
                    n_heads=32,
                    rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                    bias=False,
                    dtype=param_dtype,
                ),
                norm=norm,
                bias=False,
                dtype=param_dtype,
            ),
            lm_head=LMHeadConfig(norm=norm, bias=False, dtype=param_dtype),
            dtype=param_dtype,
        )

    def set_env_defaults(self, env: EnvConfig):
        if env.xla.gpu_all_gather_combine_threshold_mib is None:
            env.xla.gpu_all_gather_combine_threshold_mib = 1024
        if env.xla.gpu_all_reduce_combine_threshold_mib is None:
            env.xla.gpu_all_reduce_combine_threshold_mib = 1024


@TransformerRecipe.register_subclass("olmo_7B")
@dataclass
class OLMo7BRecipe(TransformerRecipe):
    vocab_size: int = 100278
    learning_rate: float = 1e-4
    sequence_length: int = 4096

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_H100:
            return 2 * 4096
        elif device_type == GPUType.NVIDIA_B200:
            return 4 * 4096
        else:
            return 1 * 4096

    def build_config(
        self,
        param_dtype: DTypeLike = float,
    ) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=self.vocab_size,
            d_model=4096,
            hidden_size=11008,
            num_layers=32,
            block=ReorderedNormTransformerBlockConfig(
                attention=MultiheadSelfAttentionConfig(
                    n_heads=32,
                    rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                    qk_norm=norm,
                    qk_norm_headwise=True,
                    bias=False,
                    dtype=param_dtype,
                ),
                norm=norm,
                bias=False,
                dtype=param_dtype,
            ),
            lm_head=LMHeadConfig(norm=norm, bias=False, dtype=param_dtype),
            dtype=param_dtype,
        )

    def set_env_defaults(self, env: EnvConfig):
        if env.xla.gpu_all_gather_combine_threshold_mib is None:
            env.xla.gpu_all_gather_combine_threshold_mib = 1024
        if env.xla.gpu_all_reduce_combine_threshold_mib is None:
            env.xla.gpu_all_reduce_combine_threshold_mib = 1024


@TransformerRecipe.register_subclass("gemma2_like_27B")
@dataclass
class Gemma2Like27BRecipe(TransformerRecipe):
    vocab_size: int = 256000
    learning_rate: float = 1e-5
    sequence_length: int = 4096

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_B200:
            return 2 * 4096
        else:
            return 1 * 4096

    def build_config(self, param_dtype: DTypeLike = float) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=self.vocab_size,
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
    vocab_size: int = 256000
    learning_rate: float = 1e-5
    sequence_length: int = 4096

    @classmethod
    def get_mbz_per_device(cls, device_type: GPUType | None = None) -> int:
        if device_type == GPUType.NVIDIA_B200:
            return 2 * 4096
        else:
            return 1 * 4096

    def build_config(self, param_dtype: DTypeLike = float) -> DefaultTransformerConfig:
        norm = RMSNormConfig(bias=False)
        return DefaultTransformerConfig(
            vocab_size=self.vocab_size,
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
