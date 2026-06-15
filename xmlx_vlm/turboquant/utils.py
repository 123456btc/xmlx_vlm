from __future__ import annotations
import math
from functools import lru_cache
from typing import Optional
import mlx.core as mx
import numpy as np

from .types import (
    DEFAULT_TURBOQUANT_SEED,
    _EPS,
    _POLAR_MAX_LEVELS,
    TurboQuantMSEState,
    TurboQuantProdState,
    TurboQuantPolarState,
    TurboQuantPolarProdState,
    TurboQuantSplitState,
)
from .kernels import (
    _metal_available,
    _mse_score_kernel,
    _mse_score_tiled_kernel,
    _pack_lowbit_kernel,
    _unpack_lowbit_kernel,
    _qjl_score_kernel,
    _prod_score_kernel,
    _mse_weighted_rot_kernel,
    _prod_score_repeat_kernel,
    _polar_prod_score_kernel,
    _polar_turbo_score_repeat_kernel,
    _mse_weighted_rot_repeat_kernel,
    _mse_scores_weighted_rot_repeat_kernel,
    _mse_scores_weighted_rot_sum_repeat_kernel,
    _metal_mse_score,
    _metal_qjl_score,
    _metal_prod_score,
    _metal_polar_prod_score,
    _metal_polar_turbo_score,
    _metal_mse_weighted_sum,
    _metal_mse_weighted_sum_from_scores,
    _metal_mse_weighted_sum_sum_from_scores,
    _compiled_integer_decode_kernel,
    _fused_integer_decode_kernel,
    _multi_query_prod_score_kernel,
    _single_tile_value_weighted_sum_kernel,
    _fused_integer_decode_single_tile_kernel,
    _fully_fused_decode_kernel,
    _gen_unrolled_extract,
    _gen_unrolled_score,
    _gen_unrolled_value,
    _fused_mse_decode_kernel,
    _fused_mse_decode_2pass_1_kernel,
    _fused_mse_decode_2pass_2_kernel,
    _metal_butterfly_wht_forward,
    _metal_butterfly_wht_inverse,
    _fused_kv_quantize_kernel,
    _fused_norot_quantize_kernel,
    _fused_mse_quantize_kernel,
    _fused_prod_quantize_kernel,
    _fused_split_decode_kernel,
    _compiled_split_decode_kernel,
)

def _validate_bits(bits: float) -> float:
    bits = float(bits)
    if bits < 1:
        raise ValueError("TurboQuant requires kv_bits >= 1.")
    rounded = round(bits * 2) / 2
    if not math.isclose(bits, rounded, abs_tol=1e-6):
        raise ValueError(
            f"TurboQuant currently supports integer and .5 bit-widths, got {bits}."
        )
    return rounded


def turboquant_enabled(bits: Optional[float], scheme: Optional[str] = None) -> bool:
    if bits is None:
        return False
    if scheme == "turboquant":
        return True
    bits = float(bits)
    return not math.isclose(bits, round(bits), abs_tol=1e-6)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _polar_levels(dim: int) -> int:
    if dim <= 1:
        return 0
    return min(_POLAR_MAX_LEVELS, int(math.log2(dim)))


def _polar_level_bits(dim: int, bits: int) -> tuple[int, ...]:
    if bits != 4:
        raise ValueError(f"PolarQuant key codec currently expects 4 bits, got {bits}.")
    levels = _polar_levels(dim)
    if levels == 0:
        return ()
    return (4,) + (2,) * (levels - 1)


@lru_cache(maxsize=None)
def _rotation_matrix(dim: int, seed: int) -> mx.array:
    if dim <= 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    if dim == 1:
        return mx.ones((1, 1), dtype=mx.float32)

    rng = np.random.default_rng(seed + dim * 7919)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    return mx.array(q.astype(np.float32))


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


@lru_cache(maxsize=None)
def _rht_padded_dim(dim: int) -> int:
    return _next_power_of_two(dim)


@lru_cache(maxsize=None)
def _rht_sign_vector(dim: int, seed: int) -> mx.array:
    """Deterministic random sign vector for Randomized Hadamard Transform."""
    if dim <= 0:
        return mx.zeros((0,), dtype=mx.float32)
    rng = np.random.default_rng(seed + dim * 7919)
    signs = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)
    return mx.array(signs)


def _rht_forward(x: mx.array, signs: mx.array) -> mx.array:
    """RHT forward: hadamard(signs * x) / sqrt(D)."""
    dim = signs.shape[0]
    D_padded = _rht_padded_dim(dim)
    y = x * signs
    if D_padded > dim:
        pad_width = [(0, 0)] * (y.ndim - 1) + [(0, D_padded - dim)]
        y = mx.pad(y, pad_width)
    y = mx.hadamard_transform(y, scale=1.0 / math.sqrt(D_padded))
    if D_padded > dim:
        y = y[..., :dim]
    return y


def _rht_inverse(x: mx.array, signs: mx.array) -> mx.array:
    """RHT inverse: signs * hadamard(x) / sqrt(D)."""
    dim = signs.shape[0]
    D_padded = _rht_padded_dim(dim)
    y = x
    if D_padded > dim:
        pad_width = [(0, 0)] * (y.ndim - 1) + [(0, D_padded - dim)]
        y = mx.pad(y, pad_width)
    y = mx.hadamard_transform(y, scale=1.0 / math.sqrt(D_padded))
    if D_padded > dim:
        y = y[..., :dim]
    return y * signs


@lru_cache(maxsize=None)
def _projection_matrix(dim: int, seed: int) -> mx.array:
    if dim <= 0:
        return mx.zeros((0, 0), dtype=mx.float32)
    rng = np.random.default_rng(seed + dim * 2971 + 17)
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    return mx.array(matrix.astype(np.float32))


def _beta_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    if dim <= 1:
        pdf = np.ones_like(grid)
    else:
        # Use lgamma to avoid overflow for large dim (e.g. dim=512)
        log_coeff = (
            math.lgamma(dim / 2) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1) / 2)
        )
        log_pdf = log_coeff + ((dim - 3) / 2) * np.log(
            np.clip(1.0 - grid**2, 1e-30, None)
        )
        pdf = np.exp(log_pdf - np.max(log_pdf))  # normalize to avoid overflow
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


@lru_cache(maxsize=None)
def _codebook(dim: int, bits: int) -> mx.array:
    if bits <= 0:
        return mx.zeros((0,), dtype=mx.float32)
    levels = 1 << bits
    if dim <= 1:
        centroids = np.linspace(-1.0, 1.0, levels, dtype=np.float32)
        return mx.array(centroids)

    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = _beta_pdf(grid, dim)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(levels):
            if i == levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return mx.array(centroids.astype(np.float32))


def _polar_angle_pdf(grid: np.ndarray, level: int) -> np.ndarray:
    if level <= 1:
        pdf = np.ones_like(grid)
    else:
        exponent = (1 << (level - 1)) - 1
        pdf = np.power(np.clip(np.sin(2.0 * grid), 0.0, None), exponent)
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


@lru_cache(maxsize=None)
def _polar_angle_codebook(level: int, bits: int) -> mx.array:
    if bits <= 0:
        return mx.zeros((0,), dtype=mx.float32)

    level_count = 1 << bits
    if level <= 1:
        step = (2.0 * math.pi) / level_count
        centroids = np.arange(level_count, dtype=np.float32) * step + step / 2.0
        return mx.array(centroids.astype(np.float32))

    grid = np.linspace(1e-6, math.pi / 2 - 1e-6, 32768, dtype=np.float32)
    weights = _polar_angle_pdf(grid, level)
    cdf = np.cumsum(weights)
    quantiles = (np.arange(level_count, dtype=np.float32) + 0.5) / level_count
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    for _ in range(100):
        boundaries = np.empty(level_count + 1, dtype=np.float32)
        boundaries[0] = 0.0
        boundaries[-1] = math.pi / 2
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        new_centroids = centroids.copy()
        for i in range(level_count):
            if i == level_count - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            else:
                mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight
        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return mx.array(centroids.astype(np.float32))


def _packed_width(length: int, bits: int) -> int:
    if length == 0 or bits == 0:
        return 0
    return (length * bits + 31) // 32


def _pack_lowbit(values: mx.array, bits: int) -> mx.array:
    if bits == 0:
        return mx.zeros((*values.shape[:-1], 0), dtype=mx.uint32)

    values = values.astype(mx.uint32)
    length = values.shape[-1]
    packed_width = _packed_width(length, bits)
    flat = values.reshape((-1, length))

    kernel = _pack_lowbit_kernel()
    if kernel is not None:
        packed = kernel(
            inputs=[flat],
            template=[
                ("Bits", bits),
                ("Length", length),
                ("PackedWidth", packed_width),
            ],
            grid=(packed_width, flat.shape[0], 1),
            threadgroup=(min(32, packed_width), 1, 1),
            output_shapes=[(flat.shape[0], packed_width)],
            output_dtypes=[mx.uint32],
        )[0]
        return packed.reshape((*values.shape[:-1], packed_width))

    packed = mx.zeros((flat.shape[0], packed_width), dtype=mx.uint32)

    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        packed[:, word_idx] |= flat[:, idx] << offset
        spill = offset + bits - 32
        if spill > 0:
            packed[:, word_idx + 1] |= flat[:, idx] >> (bits - spill)

    return packed.reshape((*values.shape[:-1], packed_width))


def _unpack_lowbit(packed: mx.array, bits: int, length: int) -> mx.array:
    if bits == 0:
        return mx.zeros((*packed.shape[:-1], 0), dtype=mx.uint32)

    packed = packed.astype(mx.uint32)
    flat = packed.reshape((-1, packed.shape[-1]))

    kernel = _unpack_lowbit_kernel()
    if kernel is not None:
        unpacked = kernel(
            inputs=[flat],
            template=[
                ("Bits", bits),
                ("Length", length),
                ("PackedWidth", flat.shape[-1]),
            ],
            grid=(length, flat.shape[0], 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(flat.shape[0], length)],
            output_dtypes=[mx.uint32],
        )[0]
        return unpacked.reshape((*packed.shape[:-1], length))

    unpacked = mx.zeros((flat.shape[0], length), dtype=mx.uint32)
    mask = (1 << bits) - 1

    for idx in range(length):
        bit_offset = idx * bits
        word_idx = bit_offset // 32
        offset = bit_offset % 32
        value = flat[:, word_idx] >> offset
        spill = offset + bits - 32
        if spill > 0:
            value |= flat[:, word_idx + 1] << (bits - spill)
        unpacked[:, idx] = value & mask

    return unpacked.reshape((*packed.shape[:-1], length))


def _concat_state(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if isinstance(lhs, TurboQuantMSEState):
        return TurboQuantMSEState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            mx.concatenate([lhs.indices, rhs.indices], axis=2),
        )
    if isinstance(lhs, TurboQuantProdState):
        return TurboQuantProdState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            mx.concatenate([lhs.mse_indices, rhs.mse_indices], axis=2),
            mx.concatenate([lhs.residual_norms, rhs.residual_norms], axis=2),
            mx.concatenate([lhs.qjl_signs, rhs.qjl_signs], axis=2),
        )
    if isinstance(lhs, TurboQuantPolarState):
        return TurboQuantPolarState(
            mx.concatenate([lhs.radii, rhs.radii], axis=2),
            tuple(
                mx.concatenate([lhs_idx, rhs_idx], axis=2)
                for lhs_idx, rhs_idx in zip(lhs.level_indices, rhs.level_indices)
            ),
        )
    if isinstance(lhs, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            mx.concatenate([lhs.norms, rhs.norms], axis=2),
            _concat_state(lhs.polar_state, rhs.polar_state),
            mx.concatenate([lhs.residual_norms, rhs.residual_norms], axis=2),
            mx.concatenate([lhs.qjl_signs, rhs.qjl_signs], axis=2),
        )
    if isinstance(lhs, TurboQuantSplitState):
        return TurboQuantSplitState(
            _concat_state(lhs.low, rhs.low),
            _concat_state(lhs.high, rhs.high),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(lhs)!r}")


def _slice_state(state, end: int):
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(state.norms[..., :end], state.indices[..., :end, :])
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            state.norms[..., :end],
            state.mse_indices[..., :end, :],
            state.residual_norms[..., :end],
            state.qjl_signs[..., :end, :],
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            state.radii[..., :end, :],
            tuple(level[..., :end, :] for level in state.level_indices),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            state.norms[..., :end],
            _slice_state(state.polar_state, end),
            state.residual_norms[..., :end],
            state.qjl_signs[..., :end, :],
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _slice_state(state.low, end),
            _slice_state(state.high, end),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _slice_state_range(state, start: int, end: int):
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(
            state.norms[..., start:end],
            state.indices[..., start:end, :],
        )
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            state.norms[..., start:end],
            state.mse_indices[..., start:end, :],
            state.residual_norms[..., start:end],
            state.qjl_signs[..., start:end, :],
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            state.radii[..., start:end, :],
            tuple(level[..., start:end, :] for level in state.level_indices),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            state.norms[..., start:end],
            _slice_state_range(state.polar_state, start, end),
            state.residual_norms[..., start:end],
            state.qjl_signs[..., start:end, :],
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _slice_state_range(state.low, start, end),
            _slice_state_range(state.high, start, end),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _state_nbytes(state) -> int:
    if state is None:
        return 0
    if isinstance(state, TurboQuantSplitState):
        return _state_nbytes(state.low) + _state_nbytes(state.high)
    if isinstance(state, tuple):
        return sum(_state_nbytes(part) for part in state)
    if isinstance(state, mx.array):
        return state.nbytes
    return 0


def _state_length(state) -> int:
    if state is None:
        return 0
    if isinstance(state, TurboQuantSplitState):
        return _state_length(state.low)
    if isinstance(state, TurboQuantMSEState):
        return state.norms.shape[2]
    if isinstance(state, TurboQuantProdState):
        return state.norms.shape[2]
    if isinstance(state, TurboQuantPolarState):
        return state.radii.shape[2]
    if isinstance(state, TurboQuantPolarProdState):
        return state.norms.shape[2]
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _allocate_state_like(state, length: int):
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(
            mx.zeros((*state.norms.shape[:2], length), dtype=state.norms.dtype),
            mx.zeros(
                (*state.indices.shape[:2], length, state.indices.shape[-1]),
                dtype=state.indices.dtype,
            ),
        )
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            mx.zeros((*state.norms.shape[:2], length), dtype=state.norms.dtype),
            mx.zeros(
                (*state.mse_indices.shape[:2], length, state.mse_indices.shape[-1]),
                dtype=state.mse_indices.dtype,
            ),
            mx.zeros(
                (*state.residual_norms.shape[:2], length),
                dtype=state.residual_norms.dtype,
            ),
            mx.zeros(
                (*state.qjl_signs.shape[:2], length, state.qjl_signs.shape[-1]),
                dtype=state.qjl_signs.dtype,
            ),
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            mx.zeros(
                (*state.radii.shape[:2], length, state.radii.shape[-1]),
                dtype=state.radii.dtype,
            ),
            tuple(
                mx.zeros(
                    (*level.shape[:2], length, level.shape[-1]),
                    dtype=level.dtype,
                )
                for level in state.level_indices
            ),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            mx.zeros((*state.norms.shape[:2], length), dtype=state.norms.dtype),
            _allocate_state_like(state.polar_state, length),
            mx.zeros(
                (*state.residual_norms.shape[:2], length),
                dtype=state.residual_norms.dtype,
            ),
            mx.zeros(
                (*state.qjl_signs.shape[:2], length, state.qjl_signs.shape[-1]),
                dtype=state.qjl_signs.dtype,
            ),
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _allocate_state_like(state.low, length),
            _allocate_state_like(state.high, length),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _write_state(dst, src, start: int):
    if src is None:
        return
    end = start + _state_length(src)
    if isinstance(dst, TurboQuantMSEState):
        dst.norms[..., start:end] = src.norms
        dst.indices[..., start:end, :] = src.indices
        return
    if isinstance(dst, TurboQuantProdState):
        dst.norms[..., start:end] = src.norms
        dst.mse_indices[..., start:end, :] = src.mse_indices
        dst.residual_norms[..., start:end] = src.residual_norms
        dst.qjl_signs[..., start:end, :] = src.qjl_signs
        return
    if isinstance(dst, TurboQuantPolarState):
        dst.radii[..., start:end, :] = src.radii
        for dst_level, src_level in zip(dst.level_indices, src.level_indices):
            dst_level[..., start:end, :] = src_level
        return
    if isinstance(dst, TurboQuantPolarProdState):
        dst.norms[..., start:end] = src.norms
        _write_state(dst.polar_state, src.polar_state, start)
        dst.residual_norms[..., start:end] = src.residual_norms
        dst.qjl_signs[..., start:end, :] = src.qjl_signs
        return
    if isinstance(dst, TurboQuantSplitState):
        _write_state(dst.low, src.low, start)
        _write_state(dst.high, src.high, start)
        return
    raise TypeError(f"Unsupported TurboQuant state type: {type(dst)!r}")


def _map_state(state, fn):
    """Apply *fn* to every ``mx.array`` leaf in a TurboQuant state NamedTuple.

    *fn* receives ``(array, ndim)`` where *ndim* is the number of dimensions
    the array has (3 for norm-like ``(B,H,T)``, 4 for index-like
    ``(B,H,T,P)``).  The function must return an ``mx.array`` with the same
    number of dimensions.
    """
    if state is None:
        return None
    if isinstance(state, TurboQuantMSEState):
        return TurboQuantMSEState(fn(state.norms, 3), fn(state.indices, 4))
    if isinstance(state, TurboQuantProdState):
        return TurboQuantProdState(
            fn(state.norms, 3),
            fn(state.mse_indices, 4),
            fn(state.residual_norms, 3),
            fn(state.qjl_signs, 4),
        )
    if isinstance(state, TurboQuantPolarState):
        return TurboQuantPolarState(
            fn(state.radii, 4),
            tuple(fn(level, 4) for level in state.level_indices),
        )
    if isinstance(state, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            fn(state.norms, 3),
            _map_state(state.polar_state, fn),
            fn(state.residual_norms, 3),
            fn(state.qjl_signs, 4),
        )
    if isinstance(state, TurboQuantSplitState):
        return TurboQuantSplitState(
            _map_state(state.low, fn),
            _map_state(state.high, fn),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(state)!r}")


def _map_state_pair(s1, s2, fn):
    """Apply *fn(a1, a2, ndim)* element-wise across two matching states."""
    if s1 is None and s2 is None:
        return None
    if isinstance(s1, TurboQuantMSEState):
        return TurboQuantMSEState(
            fn(s1.norms, s2.norms, 3), fn(s1.indices, s2.indices, 4)
        )
    if isinstance(s1, TurboQuantProdState):
        return TurboQuantProdState(
            fn(s1.norms, s2.norms, 3),
            fn(s1.mse_indices, s2.mse_indices, 4),
            fn(s1.residual_norms, s2.residual_norms, 3),
            fn(s1.qjl_signs, s2.qjl_signs, 4),
        )
    if isinstance(s1, TurboQuantPolarState):
        return TurboQuantPolarState(
            fn(s1.radii, s2.radii, 4),
            tuple(fn(l1, l2, 4) for l1, l2 in zip(s1.level_indices, s2.level_indices)),
        )
    if isinstance(s1, TurboQuantPolarProdState):
        return TurboQuantPolarProdState(
            fn(s1.norms, s2.norms, 3),
            _map_state_pair(s1.polar_state, s2.polar_state, fn),
            fn(s1.residual_norms, s2.residual_norms, 3),
            fn(s1.qjl_signs, s2.qjl_signs, 4),
        )
    if isinstance(s1, TurboQuantSplitState):
        return TurboQuantSplitState(
            _map_state_pair(s1.low, s2.low, fn),
            _map_state_pair(s1.high, s2.high, fn),
        )
    raise TypeError(f"Unsupported TurboQuant state type: {type(s1)!r}")


def _filter_state(state, batch_indices: mx.array):
    """Select batch elements from a TurboQuant state."""
    return _map_state(state, lambda a, ndim: a[batch_indices])


def _pad_state_tokens(state, left: int, right: int):
    """Pad along the token dimension (index 2)."""
    if left == 0 and right == 0:
        return state

    def _pad(a, ndim):
        if ndim == 3:  # (B, H, T)
            return mx.pad(a, [(0, 0), (0, 0), (left, right)])
        else:  # (B, H, T, P)
            return mx.pad(a, [(0, 0), (0, 0), (left, right), (0, 0)])

    return _map_state(state, _pad)


def _concat_state_batch(s1, s2):
    """Concatenate two states along the batch dimension (index 0)."""
    return _map_state_pair(s1, s2, lambda a1, a2, ndim: mx.concatenate([a1, a2]))


def _reserve_state_capacity(state, used: int, needed: int, step: int):
    if state is None:
        return None
    capacity = _state_length(state)
    if capacity >= needed:
        return state
    # Round up to next step boundary — avoids 2x growth spikes
    new_capacity = ((needed + step - 1) // step) * step
    grown = _allocate_state_like(state, new_capacity)
    if used > 0:
        _write_state(grown, _slice_state(state, used), 0)
    return grown


