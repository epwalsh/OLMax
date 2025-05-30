from ...types import DTypeLike
from ..attention import MultiheadSelfAttentionConfig
from ..lm_head import LMHeadConfig
from ..normalization import LayerNormConfig
from ..rope import RotaryPositionalEmbeddingConfig
from .block import TransformerBlockConfig
from .model import TransformerConfig


def llama_like_271M(
    vocab_size: int,
    param_dtype: DTypeLike = float,
    attn_window_size: int | tuple[int, int] | None = None,
) -> TransformerConfig:
    norm = LayerNormConfig.rms_norm(bias=False)
    return TransformerConfig(
        vocab_size=vocab_size,
        d_model=1024,
        hidden_size=2816,
        num_layers=16,
        block=TransformerBlockConfig(
            attention=MultiheadSelfAttentionConfig(
                n_heads=8,
                rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                bias=False,
                dtype=param_dtype,
                window_size=attn_window_size,
            ),
            norm=norm,
            bias=False,
            dtype=param_dtype,
        ),
        lm_head=LMHeadConfig(norm=norm, bias=False, dtype=param_dtype),
        dtype=param_dtype,
    )


def llama_like_7B(
    vocab_size: int,
    param_dtype: DTypeLike = float,
    attn_window_size: int | tuple[int, int] | None = None,
) -> TransformerConfig:
    norm = LayerNormConfig.rms_norm(bias=False)
    return TransformerConfig(
        vocab_size=vocab_size,
        d_model=4096,
        hidden_size=11008,
        num_layers=32,
        block=TransformerBlockConfig(
            attention=MultiheadSelfAttentionConfig(
                n_heads=32,
                rope=RotaryPositionalEmbeddingConfig(theta=10_000),
                bias=False,
                dtype=param_dtype,
                window_size=attn_window_size,
            ),
            norm=norm,
            bias=False,
            dtype=param_dtype,
        ),
        lm_head=LMHeadConfig(norm=norm, bias=False, dtype=param_dtype),
        dtype=param_dtype,
    )
