from dataclasses import dataclass
from enum import StrEnum

from ...activation_checkpointing import NothingSaveable
from ...env import EnvConfig
from ...types import *
from ..attention import MultiheadSelfAttentionConfig
from ..lm_head import LMHeadConfig
from ..normalization import RMSNormConfig
from ..rope import RotaryPositionalEmbeddingConfig
from .layer import (
    DefaultTransformerLayerConfig,
    GemmaTransformerLayerConfig,
    ReorderedNormTransformerLayerConfig,
)
from .model import DefaultTransformerConfig, TransformerConfig


class TransformerRecipeType(StrEnum):
    llama_like_271M = "llama_like_271M"
    llama_like_7B = "llama_like_7B"
    llama_8B = "llama_8B"

    olmo2_7B = "olmo2_7B"
    olmo2_32B = "olmo2_32B"

    gemma2_like_27B = "gemma2_like_27B"
    gemma3_like_27B = "gemma3_like_27B"

    def build_recipe(self, env_defaults: EnvConfig, device_type: DeviceType) -> "TransformerRecipe":
        return getattr(TransformerRecipe, self.name)(
            env_defaults=env_defaults, device_type=device_type
        )


@dataclass
class TransformerRecipe:
    model: TransformerConfig
    sequence_length: int
    device_microbatch_size: int
    env: EnvConfig

    @classmethod
    def llama_like_271M(
        cls, env_defaults: EnvConfig, device_type: DeviceType
    ) -> "TransformerRecipe":
        device_mbz: int
        if device_type == DeviceType.NVIDIA_A100_40GB:
            device_mbz = 16 * 1024
        elif device_type == DeviceType.NVIDIA_H100:
            device_mbz = 32 * 1024
        elif device_type == DeviceType.NVIDIA_B200:
            device_mbz = 64 * 1024
        else:
            raise NotImplementedError(
                f"this recipe does not have a default configuration yet for device type {device_type}"
            )

        norm = RMSNormConfig(bias=False)
        return cls(
            model=DefaultTransformerConfig(
                vocab_size=50_304,
                d_model=1024,
                hidden_size=2816,
                num_layers=16,
                layer=DefaultTransformerLayerConfig(
                    attention=MultiheadSelfAttentionConfig(
                        n_heads=8,
                        rope=RotaryPositionalEmbeddingConfig(theta=10_000.0),
                        bias=False,
                    ),
                    norm=norm,
                    bias=False,
                ),
                norm=norm,
                lm_head=LMHeadConfig(bias=False),
            ),
            sequence_length=1024,
            device_microbatch_size=device_mbz,
            env=env_defaults,
        )

    @classmethod
    def llama_like_7B(cls, env_defaults: EnvConfig, device_type: DeviceType) -> "TransformerRecipe":
        device_mbz: int
        if device_type == DeviceType.NVIDIA_H100:
            device_mbz = 2 * 4096
        elif device_type == DeviceType.NVIDIA_B200:
            device_mbz = 4 * 4096
        else:
            raise NotImplementedError(
                f"this recipe does not have a default configuration yet for device type {device_type}"
            )

        if env_defaults.xla.gpu_all_gather_combine_threshold_mib is None:
            env_defaults.xla.gpu_all_gather_combine_threshold_mib = 1024
        if env_defaults.xla.gpu_all_reduce_combine_threshold_mib is None:
            env_defaults.xla.gpu_all_reduce_combine_threshold_mib = 1024

        norm = RMSNormConfig(bias=False)
        return cls(
            model=DefaultTransformerConfig(
                vocab_size=50_304,
                d_model=4096,
                hidden_size=11008,
                num_layers=32,
                layer=DefaultTransformerLayerConfig(
                    attention=MultiheadSelfAttentionConfig(
                        n_heads=32,
                        rope=RotaryPositionalEmbeddingConfig(theta=10_000.0),
                        bias=False,
                    ),
                    norm=norm,
                    bias=False,
                ),
                norm=norm,
                lm_head=LMHeadConfig(bias=False),
            ),
            sequence_length=4096,
            device_microbatch_size=device_mbz,
            env=env_defaults,
        )

    @classmethod
    def llama_8B(cls, env_defaults: EnvConfig, device_type: DeviceType) -> "TransformerRecipe":
        device_mbz: int
        if device_type == DeviceType.NVIDIA_H100:
            device_mbz = 1 * 8192
        elif device_type == DeviceType.NVIDIA_B200:
            device_mbz = 2 * 8192
        else:
            raise NotImplementedError(
                f"this recipe does not have a default configuration yet for device type {device_type}"
            )

        if env_defaults.xla.gpu_all_gather_combine_threshold_mib is None:
            env_defaults.xla.gpu_all_gather_combine_threshold_mib = 1024
        if env_defaults.xla.gpu_all_reduce_combine_threshold_mib is None:
            env_defaults.xla.gpu_all_reduce_combine_threshold_mib = 1024

        norm = RMSNormConfig(bias=False)
        return cls(
            model=DefaultTransformerConfig(
                vocab_size=128_256,
                d_model=4096,
                hidden_size=14_336,
                num_layers=32,
                layer=DefaultTransformerLayerConfig(
                    attention=MultiheadSelfAttentionConfig(
                        n_heads=32,
                        n_kv_heads=8,
                        rope=RotaryPositionalEmbeddingConfig(theta=500_000.0),
                        bias=False,
                    ),
                    norm=norm,
                    bias=False,
                ),
                norm=norm,
                lm_head=LMHeadConfig(bias=False),
            ),
            sequence_length=8192,
            device_microbatch_size=device_mbz,
            env=env_defaults,
        )

    @classmethod
    def olmo2_7B(cls, env_defaults: EnvConfig, device_type: DeviceType) -> "TransformerRecipe":
        device_mbz: int
        if device_type == DeviceType.NVIDIA_H100:
            device_mbz = 2 * 4096
        elif device_type == DeviceType.NVIDIA_B200:
            device_mbz = 4 * 4096
        else:
            raise NotImplementedError(
                f"this recipe does not have a default configuration yet for device type {device_type}"
            )

        if env_defaults.xla.gpu_all_gather_combine_threshold_mib is None:
            env_defaults.xla.gpu_all_gather_combine_threshold_mib = 1024
        if env_defaults.xla.gpu_all_reduce_combine_threshold_mib is None:
            env_defaults.xla.gpu_all_reduce_combine_threshold_mib = 1024

        norm = RMSNormConfig(bias=False)
        return cls(
            model=DefaultTransformerConfig(
                vocab_size=100278,
                d_model=4096,
                hidden_size=11008,
                num_layers=32,
                layer=ReorderedNormTransformerLayerConfig(
                    attention=MultiheadSelfAttentionConfig(
                        n_heads=32,
                        rope=RotaryPositionalEmbeddingConfig(theta=10_000.0),
                        qk_norm=norm,
                        qk_norm_headwise=True,
                        bias=False,
                    ),
                    norm=norm,
                    bias=False,
                ),
                norm=norm,
                lm_head=LMHeadConfig(bias=False),
            ),
            sequence_length=4096,
            device_microbatch_size=device_mbz,
            env=env_defaults,
        )

    @classmethod
    def olmo2_32B(cls, env_defaults: EnvConfig, device_type: DeviceType) -> "TransformerRecipe":
        device_mbz: int
        if device_type == DeviceType.NVIDIA_H100:
            device_mbz = 4 * 4096
        else:
            raise NotImplementedError(
                f"this recipe does not have a default configuration yet for device type {device_type}"
            )

        norm = RMSNormConfig(bias=False)
        return cls(
            model=DefaultTransformerConfig(
                vocab_size=100278,
                d_model=5120,
                hidden_size=27648,
                num_layers=64,
                layer=ReorderedNormTransformerLayerConfig(
                    attention=MultiheadSelfAttentionConfig(
                        n_heads=40,
                        n_kv_heads=8,
                        rope=RotaryPositionalEmbeddingConfig(theta=10_000.0),
                        qk_norm=norm,
                        qk_norm_headwise=True,
                        bias=False,
                    ),
                    norm=norm,
                    bias=False,
                ),
                norm=norm,
                lm_head=LMHeadConfig(bias=False),
                scan_layers=True,
                layer_ac_policy=NothingSaveable(prevent_cse=False),
            ),
            sequence_length=4096,
            device_microbatch_size=device_mbz,
            env=env_defaults,
        )

    @classmethod
    def gemma2_like_27B(
        cls, env_defaults: EnvConfig, device_type: DeviceType
    ) -> "TransformerRecipe":
        device_mbz: int
        if device_type == DeviceType.NVIDIA_B200:
            device_mbz = 2 * 4096
        else:
            raise NotImplementedError(
                f"this recipe does not have a default configuration yet for device type {device_type}"
            )

        norm = RMSNormConfig(bias=False)
        return cls(
            model=DefaultTransformerConfig(
                vocab_size=256000,
                d_model=4608,
                hidden_size=36864,
                num_layers=46,
                layer=GemmaTransformerLayerConfig(
                    attention=MultiheadSelfAttentionConfig(
                        n_heads=32,
                        n_kv_heads=16,
                        head_dim=128,
                        rope=RotaryPositionalEmbeddingConfig(theta=10_000.0),
                        bias=False,
                    ),
                    norm=norm,
                    bias=False,
                ),
                norm=norm,
                lm_head=LMHeadConfig(bias=False),
            ),
            sequence_length=4096,
            device_microbatch_size=device_mbz,
            env=env_defaults,
        )

    @classmethod
    def gemma3_like_27B(
        cls, env_defaults: EnvConfig, device_type: DeviceType
    ) -> "TransformerRecipe":
        device_mbz: int
        if device_type == DeviceType.NVIDIA_B200:
            device_mbz = 2 * 4096
        else:
            raise NotImplementedError(
                f"this recipe does not have a default configuration yet for device type {device_type}"
            )

        norm = RMSNormConfig(bias=False)
        return cls(
            model=DefaultTransformerConfig(
                vocab_size=256000,
                d_model=5376,
                hidden_size=21504,
                num_layers=62,
                layer=GemmaTransformerLayerConfig(
                    attention=MultiheadSelfAttentionConfig(
                        n_heads=32,
                        n_kv_heads=16,
                        head_dim=128,
                        rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                        bias=False,
                        qk_norm=norm,
                        qk_norm_headwise=True,
                    ),
                    norm=norm,
                    bias=False,
                ),
                norm=norm,
                lm_head=LMHeadConfig(bias=False),
            ),
            sequence_length=4096,
            device_microbatch_size=device_mbz,
            env=env_defaults,
        )
