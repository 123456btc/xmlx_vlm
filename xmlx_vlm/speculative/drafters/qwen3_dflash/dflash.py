from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.qwen3 import MLP as Qwen3MLP

from .config import DFlashConfig

# Optional: initialize_rope supports rope_scaling (YaRN / linear / dynamic).
# Fall back to bare nn.RoPE when the symbol is unavailable.
try:
    from mlx_lm.models.rope_utils import initialize_rope as _initialize_rope
    _HAS_INITIALIZE_ROPE = True
except ImportError:
    _HAS_INITIALIZE_ROPE = False

# Optional: create_causal_mask is needed for sliding-window attention masking.
try:
    from mlx_lm.models.base import create_causal_mask as _create_causal_mask
    _HAS_CAUSAL_MASK = True
except ImportError:
    _HAS_CAUSAL_MASK = False


def _build_rope(config: DFlashConfig) -> nn.Module:
    """Build RoPE, using initialize_rope when rope_scaling is set."""
    if _HAS_INITIALIZE_ROPE and config.rope_scaling is not None:
        return _initialize_rope(
            dims=config.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
    return nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Sliding-window support
        layer_types = config.layer_types or []
        self.is_sliding = (
            len(layer_types) > layer_idx
            and layer_types[layer_idx] == "sliding_attention"
        )
        self.sliding_window = config.sliding_window if self.is_sliding else None

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache: KVCache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]

        # Sliding-window: trim context so cache stays within the window
        if self.is_sliding and self.sliding_window is not None:
            keep_ctx = self.sliding_window - 1
            if S > keep_ctx:
                skip = S - keep_ctx
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip

        # Project context and proposal separately — only context KV goes into cache
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)

        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)

        # Sliding-window attention mask
        mask = None
        if self.is_sliding and self.sliding_window is not None:
            if _HAS_CAUSAL_MASK and ctx_len + L > self.sliding_window:
                mask = _create_causal_mask(L, offset=ctx_len, window_size=self.sliding_window)
            else:
                mask = "causal"

        o = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        # Ensure layer_types is populated
        if not config.layer_types:
            config.layer_types = ["full_attention"] * config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(config)
        self.embed_tokens = None
        self.lm_head = None
        self.embed_scale: float = 1.0
        self.accept_lens: List[int] = []

    def bind(self, target_model) -> "DFlashDraftModel":
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        # Capture embed_scale (Gemma-style scaled embeddings; Qwen3 = 1.0)
        self.embed_scale = float(
            getattr(self.embed_tokens, "embed_scale", None)
            or getattr(inner, "embed_scale", 1.0)
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )
        return self

    def make_cache(self) -> List:
        """Allocate per-layer KV caches, using RotatingKVCache for sliding layers."""
        caches = []
        for lt in self.config.layer_types:
            if lt == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError(
                        "DFlashConfig.sliding_window must be set for sliding_attention layers."
                    )
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def reset(self, target_model) -> List:
        self.bind(target_model)
        self.accept_lens = []
        return self.make_cache()

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: List,
        block_size: int,
        sampler,
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        mask_id = int(self.config.mask_token_id)
        if isinstance(last_bonus, int):
            block = mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)],
                dtype=token_dtype,
            )
        else:
            B = last_bonus.shape[0]
            masks = mx.full((B, block_size - 1), mask_id, dtype=token_dtype)
            block = mx.concatenate(
                [last_bonus[:, None].astype(token_dtype), masks], axis=1
            )
        # logits_start=1: skip computing lm_head for the bonus token position
        draft_logits = self(block, hidden, cache, logits_start=1)
        return sampler(draft_logits)

    def __call__(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List,
        logits_start: int = 0,
    ) -> mx.array:
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        if logits_start:
            h = h[:, logits_start:]
        logits = self.lm_head(self.norm(h))
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[len("model."):]
            out[k] = v
        return out


DFlashKVCache = KVCache
