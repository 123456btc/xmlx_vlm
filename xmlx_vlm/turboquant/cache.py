from __future__ import annotations
import math
from typing import Optional
import mlx.core as mx
from mlx_lm.models.cache import _BaseCache, create_attention_mask

from .types import (
    DEFAULT_TURBOQUANT_SEED,
    _EPS,
    TurboQuantMSEState,
    TurboQuantProdState,
    TurboQuantPolarState,
    TurboQuantPolarProdState,
    TurboQuantSplitState,
)
from .utils import (
    _validate_bits,
    turboquant_enabled,
    _concat_state,
    _slice_state,
    _slice_state_range,
    _state_nbytes,
    _state_length,
    _allocate_state_like,
    _write_state,
    _filter_state,
    _pad_state_tokens,
    _concat_state_batch,
    _reserve_state_capacity,
)
from .kernels import (
    _metal_available,
    _metal_mse_score,
    _metal_qjl_score,
    _metal_prod_score,
    _metal_polar_prod_score,
    _metal_polar_turbo_score,
    _metal_mse_weighted_sum,
    _metal_mse_weighted_sum_from_scores,
    _metal_mse_weighted_sum_sum_from_scores,
    _fused_integer_decode_kernel,
    _multi_query_prod_score_kernel,
    _single_tile_value_weighted_sum_kernel,
    _fused_integer_decode_single_tile_kernel,
    _fully_fused_decode_kernel,
    _fused_mse_decode_kernel,
    _fused_mse_decode_2pass_1_kernel,
    _fused_mse_decode_2pass_2_kernel,
    _fused_kv_quantize_kernel,
    _fused_split_decode_kernel,
)
from .codecs import (
    _build_codec,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    _SplitCodec,
)

class _QuantizedStateProxy:
    """Wraps a quantized state tuple, providing .shape for model compatibility.

    Some models access keys.shape[-2] after cache.update_and_fetch() to slice
    masks. This proxy makes that work without dequantization.
    """

    __slots__ = ("_state", "shape")

    def __init__(self, state, n_tokens: int, n_heads: int):
        self._state = state
        # Mimic (B, H, T, D) shape — only T is needed by downstream code
        self.shape = (1, n_heads, n_tokens, 0)

    def __getattr__(self, name):
        return getattr(self._state, name)

    def __iter__(self):
        return iter(self._state)


class TurboQuantKVCache(_BaseCache):
    decode_key_chunk_size = 1 << 30
    prefill_key_chunk_size = 2048
    prefill_query_block_size = 16
    cache_step = 256

    def __init__(self, bits: float, seed: int = DEFAULT_TURBOQUANT_SEED):
        self.bits = _validate_bits(bits)
        self.seed = seed
        self.offset = 0
        self.keys = None
        self.values = None
        self.key_codec = None
        self.value_codec = None
        self._cached_state = None
        self._cached_state_offset = -1
        self._shadow_keys = None
        self._shadow_values = None

    @classmethod
    def from_cache(
        cls, cache, bits: float, seed: int = DEFAULT_TURBOQUANT_SEED
    ) -> "TurboQuantKVCache":
        turbo_cache = cls(bits=bits, seed=seed)
        keys, values = cache.state
        if keys is not None:
            turbo_cache.update_and_fetch(keys, values)
        return turbo_cache

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        if self.key_codec is None:
            # For fractional bits (e.g. 3.5), use lower bits for keys and higher
            # for values instead of SplitCodec. Both stay as fast integer codecs
            # with single-tile kernel support. Values benefit more from extra bits.
            key_bits = (
                math.floor(self.bits)
                if not math.isclose(self.bits, round(self.bits), abs_tol=1e-6)
                else self.bits
            )
            self.key_codec = _build_codec(keys, key_bits, mode="mse", seed=self.seed)
        if self.value_codec is None:
            val_bits = (
                math.ceil(self.bits)
                if not math.isclose(self.bits, round(self.bits), abs_tol=1e-6)
                else self.bits
            )
            self.value_codec = _build_codec(
                values, val_bits, mode="mse", seed=self.seed + 1
            )

    def _try_fused_kv_quantize(self, keys, values):
        """Fused key+value quantize in 1 dispatch. Returns (key_state, val_state) or (None, None)."""
        if (
            keys.shape[-2] != 1
            or not isinstance(self.key_codec, _TurboQuantMSECodec)
            or not isinstance(self.value_codec, _TurboQuantMSECodec)
        ):
            return None, None

        key_bits = int(self.key_codec.bits)
        val_bits = int(self.value_codec.bits)
        kernel = _fused_kv_quantize_kernel(key_bits, val_bits)
        if kernel is None:
            return None, None

        D = keys.shape[-1]
        k_flat = keys.reshape(-1, D)
        v_flat = values.reshape(-1, D)
        BH = k_flat.shape[0]
        k_pw = (D * key_bits + 31) // 32
        v_pw = (D * val_bits + 31) // 32

        k_norms, k_packed, v_norms, v_packed = kernel(
            inputs=[
                k_flat,
                v_flat,
                self.key_codec.rotation,
                self.value_codec.rotation,
                self.key_codec._midpoints,
                self.value_codec._midpoints,
            ],
            template=[
                ("Dim", D),
                ("KPackedWidth", k_pw),
                ("VPackedWidth", v_pw),
            ],
            grid=(D * BH, 2, 1),
            threadgroup=(D, 1, 1),
            output_shapes=[
                (BH,),
                (BH, k_pw),
                (BH,),
                (BH, v_pw),
            ],
            output_dtypes=[mx.float16, mx.uint32, mx.float16, mx.uint32],
        )

        orig = keys.shape[:-1]
        return (
            TurboQuantMSEState(k_norms.reshape(orig), k_packed.reshape(*orig, k_pw)),
            TurboQuantMSEState(v_norms.reshape(orig), v_packed.reshape(*orig, v_pw)),
        )

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._ensure_codecs(keys, values)

        # Try fused key+value quantize (1 dispatch instead of 2)
        new_keys, new_values = self._try_fused_kv_quantize(keys, values)
        if new_keys is None:
            new_keys = self.key_codec.quantize(keys)
            new_values = self.value_codec.quantize(values)

        new_end = self.offset + keys.shape[2]
        if self.keys is None:
            self.keys = _allocate_state_like(new_keys, new_end)
            self.values = _allocate_state_like(new_values, new_end)
        else:
            self.keys = _reserve_state_capacity(
                self.keys, self.offset, new_end, self.cache_step
            )
            self.values = _reserve_state_capacity(
                self.values, self.offset, new_end, self.cache_step
            )

        _write_state(self.keys, new_keys, self.offset)
        _write_state(self.values, new_values, self.offset)

        B, n_heads = keys.shape[0], keys.shape[1]
        D = keys.shape[-1]
        n_new = keys.shape[2]

        self.offset = new_end
        self._cached_state = None
        self._cached_state_offset = -1
        if n_new > 1 or (self.offset % 50 == 0):
            mx.eval(self.keys, self.values)
        ks, vs = self.state
        return (
            _QuantizedStateProxy(ks, self.offset, n_heads),
            _QuantizedStateProxy(vs, self.offset, n_heads),
        )

    @staticmethod
    def _unwrap(state):
        return state._state if isinstance(state, _QuantizedStateProxy) else state

    def dequantize(self, keys_state=None, values_state=None):
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)
        keys = self.key_codec.dequantize(keys_state).astype(mx.float32)
        values = self.value_codec.dequantize(values_state).astype(mx.float32)
        return keys, values

    def _apply_attention_mask(
        self,
        scores: mx.array,
        mask: Optional[mx.array],
        q_start: int,
        q_end: int,
        k_start: int,
        k_end: int,
        total_queries: int,
        total_tokens: int,
    ) -> mx.array:
        if mask is None:
            return scores
        if isinstance(mask, str):
            if mask == "causal":
                past_tokens = total_tokens - total_queries
                q_idx = mx.arange(past_tokens + q_start, past_tokens + q_end)
                k_idx = mx.arange(k_start, k_end)
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                causal_mask = causal_mask[None, None, None, :, :]
                return mx.where(causal_mask, scores, mx.finfo(scores.dtype).min)
            raise ValueError(f"Unsupported TurboQuant attention mask: {mask}")

        mask_chunk = mask[..., q_start:q_end, k_start:k_end]
        if mask_chunk.ndim == scores.ndim - 1:
            mask_chunk = mx.expand_dims(mask_chunk, axis=2)

        if mask_chunk.dtype == mx.bool_:
            return mx.where(mask_chunk, scores, mx.finfo(scores.dtype).min)
        return scores + mask_chunk

    def quantized_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)

        B, n_q_heads, L, D = queries.shape
        n_kv_heads = (
            keys_state.low.norms.shape[1]
            if isinstance(keys_state, TurboQuantSplitState)
            else keys_state.norms.shape[1]
        )
        n_repeats = n_q_heads // n_kv_heads

        grouped_queries = (queries * scale).reshape(
            B,
            n_kv_heads,
            n_repeats,
            L,
            D,
        )

        value_dim = self.value_codec.dim
        total_tokens = _state_length(keys_state)
        key_chunk_size = (
            self.decode_key_chunk_size if L == 1 else self.prefill_key_chunk_size
        )
        query_block_size = 1 if L == 1 else self.prefill_query_block_size

        outputs = []
        for q_start in range(0, L, query_block_size):
            q_end = min(L, q_start + query_block_size)
            q_block = grouped_queries[..., q_start:q_end, :]
            prepared_queries = self.key_codec.prepare_queries(q_block)

            output = mx.zeros(
                (B, n_kv_heads, n_repeats, q_end - q_start, value_dim),
                dtype=mx.float32,
            )
            normalizer = mx.zeros(
                (B, n_kv_heads, n_repeats, q_end - q_start),
                dtype=mx.float32,
            )
            max_score = mx.full(
                (B, n_kv_heads, n_repeats, q_end - q_start),
                -float("inf"),
                dtype=mx.float32,
            )

            for k_start in range(0, total_tokens, key_chunk_size):
                k_end = min(total_tokens, k_start + key_chunk_size)
                key_chunk = _slice_state_range(keys_state, k_start, k_end)
                value_chunk = _slice_state_range(values_state, k_start, k_end)

                scores = self.key_codec.score_prepared(prepared_queries, key_chunk)
                scores = self._apply_attention_mask(
                    scores,
                    mask,
                    q_start,
                    q_end,
                    k_start,
                    k_end,
                    L,
                    total_tokens,
                )

                chunk_output, chunk_denom, chunk_max = (
                    self.value_codec.weighted_sum_stats_from_scores(scores, value_chunk)
                )
                new_max = mx.maximum(max_score, chunk_max)
                prev_scale = mx.exp(max_score - new_max)
                chunk_scale = mx.exp(chunk_max - new_max)

                output = (
                    output * prev_scale[..., None]
                    + chunk_output * chunk_scale[..., None]
                )
                normalizer = normalizer * prev_scale + chunk_denom * chunk_scale
                max_score = new_max
                mx.eval(output, normalizer, max_score)

            outputs.append(output / mx.maximum(normalizer[..., None], _EPS))
            mx.eval(outputs[-1])

        output = mx.concatenate(outputs, axis=3)
        output = output.reshape(B, n_q_heads, L, value_dim)
        return output.astype(queries.dtype)

    def prefill_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Fast prefill: fold L queries into R dimension, reuse decode kernels.
        Avoids the expensive O(T×D²) dequantize rotation matmul."""
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)

        if not (
            isinstance(self.key_codec, _TurboQuantProdCodec)
            and isinstance(self.value_codec, _TurboQuantMSECodec)
            and isinstance(keys_state, TurboQuantProdState)
            and isinstance(values_state, TurboQuantMSEState)
        ):
            return None

        B, n_q_heads, L, D = queries.shape
        n_kv_heads = keys_state.norms.shape[1]
        n_repeats = n_q_heads // n_kv_heads
        T = keys_state.norms.shape[2]

        if T == 0:
            return None  # empty cache, let fallback handle it

        val_bits = int(self.value_codec.bits)
        if val_bits != self.value_codec.bits:
            return None
        dims_per_lane = (D + 31) // 32

        val_kernel = _single_tile_value_weighted_sum_kernel(
            val_bits, n_repeats * L, dims_per_lane
        )
        if val_kernel is None:
            return None

        # Multi-query score: unpack key ONCE per token, loop over L queries
        mq_score = _multi_query_prod_score_kernel(
            self.key_codec.mse_codec.bits,
            n_repeats,
            L,
            dims_per_lane,
        )
        if mq_score is None:
            return None

        grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)
        qt = mx.matmul(grouped, self.key_codec.query_transform_t)
        q_rot = qt[..., :D].reshape(B * n_kv_heads * n_repeats, L, D)
        q_proj = qt[..., D:].reshape(B * n_kv_heads * n_repeats, L, D)

        scores = mq_score(
            inputs=[
                q_rot,
                q_proj,
                keys_state.norms,
                keys_state.mse_indices,
                keys_state.residual_norms,
                keys_state.qjl_signs,
                self.key_codec.mse_codec.codebook,
                self.key_codec.scale_array,
            ],
            template=[
                ("Dim", D),
                ("KMsePackedWidth", keys_state.mse_indices.shape[-1]),
                ("KSignPackedWidth", keys_state.qjl_signs.shape[-1]),
            ],
            grid=(32, n_repeats, B * n_kv_heads * T),
            threadgroup=(32, 1, 1),
            output_shapes=[(B * n_kv_heads * n_repeats, L, T)],
            output_dtypes=[mx.float32],
        )[0]
        # Reshape: (B*H*R, L, T) → (B, H, R, L, T)
        scores = scores.reshape(B, n_kv_heads, n_repeats, L, T)

        # Apply mask (causal or explicit)
        if mask is not None:
            if isinstance(mask, mx.array):
                if mask.ndim == scores.ndim - 1:
                    mask = mx.expand_dims(mask, axis=2)
                if mask.dtype == mx.bool_:
                    scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                else:
                    scores = scores + mask
            # string masks not supported here, fall back
            elif isinstance(mask, str):
                return None

        # Softmax + reshape for value kernel: (B*H, R*L, T)
        weights = mx.softmax(scores, axis=-1).reshape(B * n_kv_heads, n_repeats * L, T)

        # Value weighted sum using TG=D kernel
        tok_tile_size = 1024
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size
        out_shape = (B * n_kv_heads * num_tok_tiles, n_repeats * L, D)

        out_tiled = val_kernel(
            inputs=[
                weights,
                values_state.norms,
                values_state.indices,
                self.value_codec.codebook,
            ],
            template=[
                ("Dim", D),
                ("RepeatCount", n_repeats * L),
                ("TokTileSize", tok_tile_size),
                ("DimsPerLane", dims_per_lane),
                ("PackedWidth", values_state.indices.shape[-1]),
            ],
            grid=(D, 1, B * n_kv_heads * num_tok_tiles),
            threadgroup=(D, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[mx.float32],
        )[0]

        # Reduce across tiles
        out_tiled = out_tiled.reshape(B * n_kv_heads, num_tok_tiles, n_repeats * L, D)
        if num_tok_tiles > 1:
            out_rotated = mx.sum(out_tiled, axis=1)
        else:
            out_rotated = out_tiled.squeeze(1)
        out_rotated = out_rotated.reshape(B, n_kv_heads, n_repeats, L, D)

        # Rotate values back
        output = self.value_codec._rotate_inverse(out_rotated)
        return output.reshape(B, n_q_heads, L, D).astype(queries.dtype)

    def _separate_score_value_decode(
        self,
        grouped_queries: mx.array,
        keys_state,
        values_state,
    ) -> Optional[mx.array]:
        """Separate-kernel decode: fast key scoring + single-tile value weighted sum.
        2-3x faster than the fused kernel at large D because each kernel only
        iterates its own dimensions once."""
        key_is_prod = isinstance(self.key_codec, _TurboQuantProdCodec)
        key_is_mse = isinstance(self.key_codec, _TurboQuantMSECodec)
        if not (
            _metal_available()
            and (key_is_prod or key_is_mse)
            and isinstance(self.value_codec, _TurboQuantMSECodec)
            and (
                isinstance(keys_state, TurboQuantProdState)
                if key_is_prod
                else isinstance(keys_state, TurboQuantMSEState)
            )
            and isinstance(values_state, TurboQuantMSEState)
        ):
            return None

        B = grouped_queries.shape[0]
        H = grouped_queries.shape[1]
        R = grouped_queries.shape[2]
        D = grouped_queries.shape[-1]
        T = keys_state.norms.shape[2]

        val_bits = int(self.value_codec.bits)
        if val_bits != self.value_codec.bits:
            return None
        dims_per_lane = (D + 31) // 32

        val_kernel = _single_tile_value_weighted_sum_kernel(val_bits, R, dims_per_lane)
        if val_kernel is None:
            return None

        # Step 1: Key scoring — polymorphic (works for both Prod and MSE keys)
        prepared_queries = self.key_codec.prepare_queries(grouped_queries)
        scores = self.key_codec.score_prepared(prepared_queries, keys_state)
        # scores: (B, H, R, 1, T) → (B*H, R, T)
        scores_2d = scores.reshape(B * H, R, T)

        # Step 2: Precompute softmax weights (avoids exp() in value kernel)
        weights = mx.softmax(scores_2d, axis=-1)  # (B*H, R, T)

        # Step 3: Single-tile value weighted sum with precomputed weights
        tok_tile_size = 1024
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size
        out_shape = (B * H * num_tok_tiles, R, D)

        out_tiled = val_kernel(
            inputs=[
                weights,
                values_state.norms,
                values_state.indices,
                self.value_codec.codebook,
            ],
            template=[
                ("Dim", D),
                ("RepeatCount", R),
                ("TokTileSize", tok_tile_size),
                ("DimsPerLane", dims_per_lane),
                ("PackedWidth", values_state.indices.shape[-1]),
            ],
            grid=(D, 1, B * H * num_tok_tiles),
            threadgroup=(D, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[mx.float32],
        )[0]

        # Cross-tile reduction (simple sum since weights are pre-normalized)
        out_tiled = out_tiled.reshape(B * H, num_tok_tiles, R, D)
        if num_tok_tiles > 1:
            out_rotated = mx.sum(out_tiled, axis=1)
        else:
            out_rotated = out_tiled.squeeze(1)
        out_rotated = out_rotated.reshape(B, H, R, D)

        # Rotate values back to original space
        output = self.value_codec._rotate_inverse(out_rotated)
        return mx.expand_dims(output, axis=3)

    def _compiled_split_decode_attention(
        self,
        grouped_queries: mx.array,
        keys_state,
        values_state,
    ) -> Optional[mx.array]:
        """Fused decode for SplitCodec — single Metal kernel for score + softmax
        + weighted_sum across both low/high halves."""
        if not (
            _metal_available()
            and isinstance(self.key_codec, _SplitCodec)
            and isinstance(self.value_codec, _SplitCodec)
            and isinstance(keys_state, TurboQuantSplitState)
            and isinstance(values_state, TurboQuantSplitState)
            and isinstance(self.key_codec.low_codec, _TurboQuantProdCodec)
            and isinstance(self.value_codec.low_codec, _TurboQuantMSECodec)
        ):
            return None

        kc = self.key_codec
        vc = self.value_codec
        low_bits = kc.lower_bits
        high_bits = kc.upper_bits

        if kc.low_codec.mse_codec.bits <= 0 or kc.high_codec.mse_codec.bits <= 0:
            return None

        B = grouped_queries.shape[0]
        H = grouped_queries.shape[1]
        R = grouped_queries.shape[2]
        dim_low = kc.low_codec.dim
        dim_high = kc.high_codec.dim
        T = keys_state.low.norms.shape[2]

        kernel = _fused_split_decode_kernel(low_bits, high_bits, R)
        if kernel is None:
            return None

        # Single combined query transform: 1 matmul replaces 2 takes + 2 matmuls
        if kc.combined_query_transform_t is not None:
            qt = mx.matmul(grouped_queries, kc.combined_query_transform_t)
            dl2 = dim_low * 2
            q_rot_low = qt[..., :dim_low].reshape(B, H, R, dim_low)
            q_proj_low = qt[..., dim_low:dl2].reshape(B, H, R, dim_low)
            q_rot_high = qt[..., dl2 : dl2 + dim_high].reshape(B, H, R, dim_high)
            q_proj_high = qt[..., dl2 + dim_high :].reshape(B, H, R, dim_high)
        else:
            q_low = mx.take(grouped_queries, kc.low_idx, axis=-1)
            q_high = mx.take(grouped_queries, kc.high_idx, axis=-1)
            qt_low = mx.matmul(q_low, kc.low_codec.query_transform_t)
            qt_high = mx.matmul(q_high, kc.high_codec.query_transform_t)
            q_rot_low = qt_low[..., :dim_low].reshape(B, H, R, dim_low)
            q_proj_low = qt_low[..., dim_low:].reshape(B, H, R, dim_low)
            q_rot_high = qt_high[..., :dim_high].reshape(B, H, R, dim_high)
            q_proj_high = qt_high[..., dim_high:].reshape(B, H, R, dim_high)

        low_mse_bits = max(low_bits - 1, 0)
        high_mse_bits = max(high_bits - 1, 0)
        val_dim = dim_low + dim_high

        tok_tile_size = 1024
        num_val_tiles = (val_dim + 31) // 32
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size

        acc_shape = (B * H * num_tok_tiles, R, val_dim)
        sm_shape = (B * H * num_tok_tiles * R,)  # scalar per (bh, tile, repeat)
        out_acc, out_sum, out_max = kernel(
            inputs=[
                q_rot_low,
                q_proj_low,
                q_rot_high,
                q_proj_high,
                keys_state.low.norms,
                keys_state.low.mse_indices,
                keys_state.low.residual_norms,
                keys_state.low.qjl_signs,
                keys_state.high.norms,
                keys_state.high.mse_indices,
                keys_state.high.residual_norms,
                keys_state.high.qjl_signs,
                values_state.low.norms,
                values_state.low.indices,
                values_state.high.norms,
                values_state.high.indices,
                kc.low_codec.mse_codec.codebook,
                kc.high_codec.mse_codec.codebook,
                kc.low_codec.scale_array,
                kc.high_codec.scale_array,
                vc.low_codec.codebook,
                vc.high_codec.codebook,
            ],
            template=[
                ("DimLow", dim_low),
                ("DimHigh", dim_high),
                ("RepeatCount", R),
                ("TokTileSize", tok_tile_size),
                ("KLMsePackedWidth", keys_state.low.mse_indices.shape[-1]),
                ("KLSignPackedWidth", keys_state.low.qjl_signs.shape[-1]),
                ("KHMsePackedWidth", keys_state.high.mse_indices.shape[-1]),
                ("KHSignPackedWidth", keys_state.high.qjl_signs.shape[-1]),
                ("VLPackedWidth", values_state.low.indices.shape[-1]),
                ("VHPackedWidth", values_state.high.indices.shape[-1]),
                ("VLBits", vc.low_codec.bits),
                ("VHBits", vc.high_codec.bits),
            ],
            grid=(32, num_val_tiles, B * H * num_tok_tiles),
            threadgroup=(32, 1, 1),
            output_shapes=[acc_shape, sm_shape, sm_shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )

        # Cross-tile reduction: sum/max are (BH*tiles*R,) scalars, acc is (BH*tiles, R, D)
        out_acc = out_acc.reshape(B * H, num_tok_tiles, R, val_dim)
        out_sum = out_sum.reshape(B * H, num_tok_tiles, R)
        out_max = out_max.reshape(B * H, num_tok_tiles, R)
        global_max = mx.max(out_max, axis=1, keepdims=True)  # (BH, 1, R)
        scale_factors = mx.exp(out_max - global_max)  # (BH, tiles, R)
        scaled_acc = mx.sum(out_acc * scale_factors[..., None], axis=1)  # (BH, R, D)
        denom = mx.sum(out_sum * scale_factors, axis=1)  # (BH, R)
        out_rotated = (scaled_acc / mx.maximum(denom[..., None], _EPS)).reshape(
            B, H, R, val_dim
        )

        out_low = mx.matmul(out_rotated[..., :dim_low], vc.low_codec.rotation)
        out_high = mx.matmul(out_rotated[..., dim_low:], vc.high_codec.rotation)
        merged = mx.concatenate([out_low, out_high], axis=-1)
        output = mx.take(merged, vc.restore_order, axis=-1)
        return mx.expand_dims(output, axis=3)

    def _compiled_integer_decode_attention(
        self,
        grouped_queries: mx.array,
        keys_state,
        values_state,
    ) -> Optional[mx.array]:
        if not (
            _metal_available()
            and isinstance(self.key_codec, _TurboQuantProdCodec)
            and isinstance(self.value_codec, _TurboQuantMSECodec)
            and self.key_codec.mse_codec.bits > 0
            and isinstance(keys_state, TurboQuantProdState)
            and isinstance(values_state, TurboQuantMSEState)
        ):
            return None

        bits = int(self.value_codec.bits)
        if bits != self.value_codec.bits:
            return None

        B, H, R = (
            grouped_queries.shape[0],
            grouped_queries.shape[1],
            grouped_queries.shape[2],
        )
        D = grouped_queries.shape[-1]
        T = keys_state.norms.shape[2]

        dims_per_lane = (D + 31) // 32
        key_mse_bits = self.key_codec.mse_codec.bits
        use_rht = self.value_codec.use_rht

        # Fully fused path: score + softmax + value + rotation in 1 dispatch.
        if T <= 32768:
            ff_kernel = _fully_fused_decode_kernel(
                # Dense rotation for decode kernel — butterfly WHT barriers
                # outweigh the D² compute savings at small threadgroup counts.
                bits,
                R,
                dims_per_lane,
                key_mse_bits,
                use_rht=False,
            )
            if ff_kernel is not None:
                q_rot, q_proj = self.key_codec.prepare_queries(grouped_queries)
                q_rot = q_rot.reshape(B, H, R, D)
                q_proj = q_proj.reshape(B, H, R, D)

                rot_input = self.value_codec.rotation_t
                template = [
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("DimsPerLane", dims_per_lane),
                    ("KMsePackedWidth", keys_state.mse_indices.shape[-1]),
                    ("KSignPackedWidth", keys_state.qjl_signs.shape[-1]),
                    ("VPackedWidth", values_state.indices.shape[-1]),
                ]

                out = ff_kernel(
                    inputs=[
                        q_rot,
                        q_proj,
                        keys_state.norms,
                        keys_state.mse_indices,
                        keys_state.residual_norms,
                        keys_state.qjl_signs,
                        values_state.norms,
                        values_state.indices,
                        self.key_codec.mse_codec.codebook,
                        self.key_codec.scale_array,
                        self.value_codec.codebook,
                        rot_input,
                    ],
                    template=template,
                    grid=(32, 1, B * H),
                    threadgroup=(32, 1, 1),
                    output_shapes=[(B * H, R, D)],
                    output_dtypes=[mx.float32],
                )[0]
                output = out.reshape(B, H, R, D)
                return mx.expand_dims(output, axis=3)

        # Tiled fused paths for longer contexts
        qt = mx.matmul(grouped_queries, self.key_codec.query_transform_t)
        q_rot = qt[..., :D].reshape(B, H, R, D)
        q_proj = qt[..., D:].reshape(B, H, R, D)
        tok_tile_size = 1024
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size

        # Single-tile path: each lane handles all its value dims.
        # Zero key read redundancy — faster at 256k+ where bandwidth dominates.
        single_kernel = _fused_integer_decode_single_tile_kernel(
            bits, R, dims_per_lane, key_mse_bits
        )
        # Multi-tile path: one val_dim per lane, multiple tiles read keys redundantly.
        # Better parallelism at shorter contexts.
        multi_kernel = _fused_integer_decode_kernel(bits, R, key_mse_bits)

        # Single-tile wins when val_tile redundancy outweighs parallelism benefit.
        # More val_tiles (larger D) → lower crossover. With 8 val_tiles (D=256),
        # single-tile wins even at 128k. With 5 tiles (D=160), crossover ~256k.
        num_val_tiles = (D + 31) // 32
        min_threadgroups = 64
        use_single = (
            single_kernel is not None
            and num_tok_tiles * B * H >= min_threadgroups
            and (num_val_tiles >= 8 or T >= 262144)
        )

        if use_single:
            acc_shape = (B * H * num_tok_tiles, R, D)
            sm_shape = (B * H * num_tok_tiles * R,)

            out_acc, out_sum, out_max = single_kernel(
                inputs=[
                    q_rot,
                    q_proj,
                    keys_state.norms,
                    keys_state.mse_indices,
                    keys_state.residual_norms,
                    keys_state.qjl_signs,
                    values_state.norms,
                    values_state.indices,
                    self.key_codec.mse_codec.codebook,
                    self.key_codec.scale_array,
                    self.value_codec.codebook,
                ],
                template=[
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("TokTileSize", tok_tile_size),
                    ("DimsPerLane", dims_per_lane),
                    ("KMsePackedWidth", keys_state.mse_indices.shape[-1]),
                    ("KSignPackedWidth", keys_state.qjl_signs.shape[-1]),
                    ("VPackedWidth", values_state.indices.shape[-1]),
                ],
                grid=(32, 1, B * H * num_tok_tiles),
                threadgroup=(32, 1, 1),
                output_shapes=[acc_shape, sm_shape, sm_shape],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )

            # Same cross-tile reduction as multi-tile
            out_acc = out_acc.reshape(B * H, num_tok_tiles, R, D)
            out_sum = out_sum.reshape(B * H, num_tok_tiles, R)
            out_max = out_max.reshape(B * H, num_tok_tiles, R)
            global_max = mx.max(out_max, axis=1, keepdims=True)
            scale_factors = mx.exp(out_max - global_max)
            scaled_acc = mx.sum(out_acc * scale_factors[..., None], axis=1)
            denom = mx.sum(out_sum * scale_factors, axis=1)
            out_rotated = (scaled_acc / mx.maximum(denom[..., None], _EPS)).reshape(
                B, H, R, D
            )

            output = self.value_codec._rotate_inverse(out_rotated)
            return mx.expand_dims(output, axis=3)

        elif multi_kernel is not None:
            num_val_tiles = (D + 31) // 32
            acc_shape = (B * H * num_tok_tiles, R, D)
            sm_shape = (B * H * num_tok_tiles * R,)

            out_acc, out_sum, out_max = multi_kernel(
                inputs=[
                    q_rot,
                    q_proj,
                    keys_state.norms,
                    keys_state.mse_indices,
                    keys_state.residual_norms,
                    keys_state.qjl_signs,
                    values_state.norms,
                    values_state.indices,
                    self.key_codec.mse_codec.codebook,
                    self.key_codec.scale_array,
                    self.value_codec.codebook,
                ],
                template=[
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("TokTileSize", tok_tile_size),
                    ("KMsePackedWidth", keys_state.mse_indices.shape[-1]),
                    ("KSignPackedWidth", keys_state.qjl_signs.shape[-1]),
                    ("VPackedWidth", values_state.indices.shape[-1]),
                    ("ValBits", bits),
                ],
                grid=(32, num_val_tiles, B * H * num_tok_tiles),
                threadgroup=(32, 1, 1),
                output_shapes=[acc_shape, sm_shape, sm_shape],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )

            # Cross-tile reduction with scalar sum/max
            out_acc = out_acc.reshape(B * H, num_tok_tiles, R, D)
            out_sum = out_sum.reshape(B * H, num_tok_tiles, R)
            out_max = out_max.reshape(B * H, num_tok_tiles, R)
            global_max = mx.max(out_max, axis=1, keepdims=True)
            scale_factors = mx.exp(out_max - global_max)
            scaled_acc = mx.sum(out_acc * scale_factors[..., None], axis=1)
            denom = mx.sum(out_sum * scale_factors, axis=1)
            out_rotated = (scaled_acc / mx.maximum(denom[..., None], _EPS)).reshape(
                B, H, R, D
            )

            output = self.value_codec._rotate_inverse(out_rotated)
            return mx.expand_dims(output, axis=3)

        # Fallback: compiled two-dispatch path
        decode = _compiled_integer_decode_kernel(bits)
        return decode(
            grouped_queries,
            keys_state.norms,
            keys_state.mse_indices,
            keys_state.residual_norms,
            keys_state.qjl_signs,
            values_state.norms,
            values_state.indices,
            self.key_codec.query_transform_t,
            self.key_codec.mse_codec.codebook,
            self.key_codec.scale_array,
            self.value_codec.codebook,
            self.value_codec.rotation,
        )

    def decode_attention(
        self,
        queries: mx.array,
        keys_state=None,
        values_state=None,
        scale: float = 1.0,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if keys_state is None or values_state is None:
            keys_state, values_state = self.state
        keys_state = self._unwrap(keys_state)
        values_state = self._unwrap(values_state)

        if queries.shape[-2] != 1:
            raise ValueError(
                "TurboQuant decode attention expects a single query token."
            )

        B, n_q_heads, L, D = queries.shape
        n_kv_heads = (
            keys_state.low.norms.shape[1]
            if isinstance(keys_state, TurboQuantSplitState)
            else keys_state.norms.shape[1]
        )
        n_repeats = n_q_heads // n_kv_heads

        grouped_queries = (queries * scale).reshape(
            B,
            n_kv_heads,
            n_repeats,
            L,
            D,
        )

        value_dim = self.value_codec.dim
        total_tokens = _state_length(keys_state)

        if total_tokens <= self.decode_key_chunk_size and (
            mask is None or (isinstance(mask, str) and mask == "causal")
        ):
            # Fused quantized SDPA matching MLX architecture.
            if (
                isinstance(self.key_codec, _TurboQuantMSECodec)
                and isinstance(self.value_codec, _TurboQuantMSECodec)
                and isinstance(keys_state, TurboQuantMSEState)
                and isinstance(values_state, TurboQuantMSEState)
            ):
                key_bits = int(self.key_codec.bits)
                val_bits = int(self.value_codec.bits)
                dtype = queries.dtype
                q_rot = self.key_codec.prepare_queries(grouped_queries)
                q_rot_flat = q_rot.reshape(B * n_kv_heads * n_repeats, D)
                BQH = B * n_q_heads

                if total_tokens <= 2048:
                    # Single-pass: 32 simdgroups cooperate per q_head
                    fused_kernel = _fused_mse_decode_kernel(key_bits, val_bits, D)
                    if fused_kernel is not None:
                        out = fused_kernel(
                            inputs=[
                                q_rot_flat,
                                keys_state.norms,
                                keys_state.indices,
                                self.key_codec.codebook,
                                values_state.norms,
                                values_state.indices,
                                self.value_codec.codebook,
                            ],
                            template=[
                                ("Dim", D),
                                ("RepeatCount", n_repeats),
                                ("KPackedWidth", keys_state.indices.shape[-1]),
                                ("VPackedWidth", values_state.indices.shape[-1]),
                            ],
                            grid=(BQH * 1024, 1, 1),
                            threadgroup=(1024, 1, 1),
                            output_shapes=[(BQH, D)],
                            output_dtypes=[mx.float32],
                        )[0]
                        out_rotated = out.reshape(B, n_kv_heads, n_repeats, D)
                        output = self.value_codec._rotate_inverse(out_rotated)
                        return output.reshape(B, n_q_heads, L, value_dim).astype(dtype)

                # 2-pass: split KV across blocks for GPU saturation
                pass1 = _fused_mse_decode_2pass_1_kernel(key_bits, val_bits, D)
                pass2 = _fused_mse_decode_2pass_2_kernel()
                if pass1 is not None and pass2 is not None:
                    if total_tokens <= 8192:
                        num_blocks = 64
                    elif total_tokens <= 32768:
                        num_blocks = 128
                    elif total_tokens <= 65536:
                        num_blocks = 256
                    else:
                        num_blocks = 512

                    acc_shape = (BQH * num_blocks, D)
                    sm_shape = (BQH * num_blocks,)
                    out_acc, out_sums, out_maxs = pass1(
                        inputs=[
                            q_rot_flat,
                            keys_state.norms,
                            keys_state.indices,
                            self.key_codec.codebook,
                            values_state.norms,
                            values_state.indices,
                            self.value_codec.codebook,
                        ],
                        template=[
                            ("Dim", D),
                            ("RepeatCount", n_repeats),
                            ("Blocks", num_blocks),
                            ("KPackedWidth", keys_state.indices.shape[-1]),
                            ("VPackedWidth", values_state.indices.shape[-1]),
                        ],
                        grid=(
                            n_kv_heads * 32,
                            B * n_repeats,
                            num_blocks,
                        ),
                        threadgroup=(32, n_repeats, 1),
                        output_shapes=[acc_shape, sm_shape, sm_shape],
                        output_dtypes=[mx.float32, mx.float32, mx.float32],
                    )

                    out = pass2(
                        inputs=[out_acc, out_sums, out_maxs],
                        template=[("Dim", D), ("Blocks", num_blocks)],
                        grid=(BQH * 1024, 1, 1),
                        threadgroup=(1024, 1, 1),
                        output_shapes=[(BQH, D)],
                        output_dtypes=[mx.float32],
                    )[0]

                    out_rotated = out.reshape(B, n_kv_heads, n_repeats, D)
                    output = self.value_codec._rotate_inverse(out_rotated)
                    return output.reshape(B, n_q_heads, L, value_dim).astype(dtype)

            # Separate-kernel path fallback
            sep_output = self._separate_score_value_decode(
                grouped_queries,
                keys_state,
                values_state,
            )
            if sep_output is not None:
                output = sep_output.reshape(B, n_q_heads, L, value_dim)
                return output.astype(queries.dtype)

            # Fallback: fused kernel paths
            fast_output = self._compiled_integer_decode_attention(
                grouped_queries,
                keys_state,
                values_state,
            )
            if fast_output is not None:
                output = fast_output.reshape(B, n_q_heads, L, value_dim)
                return output.astype(queries.dtype)

            fast_output = self._compiled_split_decode_attention(
                grouped_queries,
                keys_state,
                values_state,
            )
            if fast_output is not None:
                output = fast_output.reshape(B, n_q_heads, L, value_dim)
                return output.astype(queries.dtype)

            prepared_queries = self.key_codec.prepare_queries(grouped_queries)
            scores = self.key_codec.score_prepared(prepared_queries, keys_state)
            output = self.value_codec.weighted_sum_from_scores(scores, values_state)
            output = output.reshape(B, n_q_heads, L, value_dim)
            return output.astype(queries.dtype)

        prepared_queries = self.key_codec.prepare_queries(grouped_queries)

        output = mx.zeros((B, n_kv_heads, n_repeats, L, value_dim), dtype=mx.float32)
        normalizer = mx.zeros((B, n_kv_heads, n_repeats, L), dtype=mx.float32)
        max_score = mx.full(
            (B, n_kv_heads, n_repeats, L),
            -float("inf"),
            dtype=mx.float32,
        )

        for k_start in range(0, total_tokens, self.decode_key_chunk_size):
            k_end = min(total_tokens, k_start + self.decode_key_chunk_size)
            key_chunk = _slice_state_range(keys_state, k_start, k_end)
            value_chunk = _slice_state_range(values_state, k_start, k_end)

            scores = self.key_codec.score_prepared(prepared_queries, key_chunk)
            scores = self._apply_attention_mask(
                scores,
                mask,
                0,
                L,
                k_start,
                k_end,
                L,
                total_tokens,
            )

            chunk_output, chunk_denom, chunk_max = (
                self.value_codec.weighted_sum_stats_from_scores(scores, value_chunk)
            )
            new_max = mx.maximum(max_score, chunk_max)
            prev_scale = mx.exp(max_score - new_max)
            chunk_scale = mx.exp(chunk_max - new_max)

            output = (
                output * prev_scale[..., None] + chunk_output * chunk_scale[..., None]
            )
            normalizer = normalizer * prev_scale + chunk_denom * chunk_scale
            max_score = new_max
            mx.eval(output, normalizer, max_score)

        output = output / mx.maximum(normalizer[..., None], _EPS)
        output = output.reshape(B, n_q_heads, L, value_dim)
        return output.astype(queries.dtype)

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.keys is None:
            return None, None
        if self._cached_state_offset == self.offset:
            return self._cached_state
        sliced = _slice_state(self.keys, self.offset), _slice_state(
            self.values, self.offset
        )
        self._cached_state = sliced
        self._cached_state_offset = self.offset
        return sliced

    @state.setter
    def state(self, value):
        self._cached_state = None
        self._cached_state_offset = -1
        if value is None:
            self.keys, self.values = None, None
            self.offset = 0
            return
        self.keys, self.values = value
        self.offset = _state_length(self.keys)

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.bits, self.seed)))

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0])
        self.bits = float(value[1])
        self.seed = int(value[2])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._cached_state = None
        self._cached_state_offset = -1
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return _state_nbytes(self.state)


# ── Batch-aware TurboQuant cache for continuous batching ────────────────


class BatchTurboQuantKVCache(_BaseCache):
    """Batch-aware TurboQuant KV cache for continuous batching.

    Wraps TurboQuant's quantization codecs with per-sequence offsets and
    left-padding so that ``Batch.extend`` / ``Batch.filter`` work during
    continuous batching.

    Unlike ``BatchQuantizedKVCache`` (uniform ``mx.quantize``), this uses
    TurboQuant's MSE/Prod codecs for higher quality at the same bit-rate.
    """

    cache_step = 256

    def __init__(
        self,
        left_padding: list,
        bits: float,
        seed: int = DEFAULT_TURBOQUANT_SEED,
    ):
        self.bits = _validate_bits(bits)
        self.seed = seed
        self.keys = None
        self.values = None
        self.key_codec = None
        self.value_codec = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-lp for lp in left_padding])
        self._idx = 0

    # ------------------------------------------------------------------
    # Codec initialisation (deferred until first update)
    # ------------------------------------------------------------------

    def _ensure_codecs(self, keys: mx.array):
        if self.key_codec is not None:
            return
        D = keys.shape[-1]
        # Delegate to a temporary TurboQuantKVCache to get codec setup right
        tmp = TurboQuantKVCache(bits=self.bits, seed=self.seed)
        tmp._ensure_codecs(keys, keys)  # values have same D
        self.key_codec = tmp.key_codec
        self.value_codec = tmp.value_codec

    # ------------------------------------------------------------------
    # Core cache operation
    # ------------------------------------------------------------------

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._ensure_codecs(keys)
        prev = self._idx

        new_keys = self.key_codec.quantize(keys)
        new_values = self.value_codec.quantize(values)

        new_end = prev + keys.shape[2]
        if self.keys is None:
            self.keys = _allocate_state_like(new_keys, new_end)
            self.values = _allocate_state_like(new_values, new_end)
        else:
            self.keys = _reserve_state_capacity(
                self.keys, prev, new_end, self.cache_step
            )
            self.values = _reserve_state_capacity(
                self.values, prev, new_end, self.cache_step
            )

        _write_state(self.keys, new_keys, prev)
        _write_state(self.values, new_values, prev)

        self.offset += keys.shape[2]
        self._idx = new_end

        if keys.shape[2] > 1 or (self._idx % 50 == 0):
            mx.eval(self.keys, self.values)

        ks = _slice_state(self.keys, self._idx)
        vs = _slice_state(self.values, self._idx)
        n_heads = keys.shape[1]
        return (
            _QuantizedStateProxy(ks, self._idx, n_heads),
            _QuantizedStateProxy(vs, self._idx, n_heads),
        )

    # ------------------------------------------------------------------
    # Batch operations for Batch.filter / Batch.extend
    # ------------------------------------------------------------------

    def filter(self, batch_indices: mx.array):
        if self.keys is not None:
            self.keys = _filter_state(self.keys, batch_indices)
            self.values = _filter_state(self.values, batch_indices)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_lp = self.left_padding.min().item()
        if min_lp > 0:
            if self.keys is not None:
                # Trim leading padding tokens
                def _trim(a, ndim):
                    if ndim == 3:
                        return a[..., min_lp:]
                    return a[..., min_lp:, :]

                self.keys = _map_state(self.keys, _trim)
                self.values = _map_state(self.values, _trim)
            self._idx -= min_lp
            self.left_padding -= min_lp

    def extend(self, other: "BatchTurboQuantKVCache"):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)

        def _pad_side(cache_obj):
            if cache_obj.keys is None:
                return None
            left = max_idx - cache_obj._idx
            right = 0
            k = _pad_state_tokens(
                _slice_state(cache_obj.keys, cache_obj._idx), left, right
            )
            v = _pad_state_tokens(
                _slice_state(cache_obj.values, cache_obj._idx), left, right
            )
            lp = cache_obj.left_padding + left
            return k, v, cache_obj.offset, lp

        r_self = _pad_side(self)
        r_other = _pad_side(other)

        if r_self is None and r_other is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        if r_self is None:
            self.keys, self.values, so, slp = r_other
            self.offset = mx.concatenate([self.offset, so])
            self.left_padding = mx.concatenate([self.left_padding + max_idx, slp])
            self._idx = max_idx
            return
        if r_other is None:
            self.keys, self.values, so, slp = r_self
            self.offset = mx.concatenate([so, other.offset])
            self.left_padding = mx.concatenate([slp, other.left_padding + max_idx])
            self._idx = max_idx
            return

        sk, sv, so, slp = r_self
        ok, ov, oo, olp = r_other

        self.keys = _concat_state_batch(sk, ok)
        self.values = _concat_state_batch(sv, ov)
        self.offset = mx.concatenate([so, oo])
        self.left_padding = mx.concatenate([slp, olp])
        self._idx = max_idx

    # ------------------------------------------------------------------
    # Dequantize (for attention fallback)
    # ------------------------------------------------------------------

    def dequantize(self, keys_state=None, values_state=None):
        if keys_state is None or values_state is None:
            keys_state = _slice_state(self.keys, self._idx)
            values_state = _slice_state(self.values, self._idx)
        if isinstance(keys_state, _QuantizedStateProxy):
            keys_state = keys_state._state
        if isinstance(values_state, _QuantizedStateProxy):
            values_state = values_state._state
        k = self.key_codec.dequantize(keys_state).astype(mx.float32)
        v = self.value_codec.dequantize(values_state).astype(mx.float32)
        return k, v

    # ------------------------------------------------------------------
    # State / introspection
    # ------------------------------------------------------------------

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.offset, self.left_padding
        k = _slice_state(self.keys, self._idx)
        v = _slice_state(self.values, self._idx)
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, val):
        self.keys, self.values, self.offset, self.left_padding = val
        if self.keys is not None:
            self._idx = _state_length(self.keys)

    @property
    def meta_state(self):
        return tuple(map(str, (self._idx, self.bits, self.seed)))

    @meta_state.setter
    def meta_state(self, v):
        self._idx = int(v[0])
        self.bits = float(v[1])
        self.seed = int(v[2])

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0

    def empty(self):
        return self.keys is None

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def group_size(self):
        # Required by mlx_lm's scaled_dot_product_attention dispatch
        # but not used for TurboQuant (it has its own attention path)
        return 64

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        s = _slice_state(self.keys, self._idx)
        v = _slice_state(self.values, self._idx)
        return _state_nbytes(s) + _state_nbytes(v)
