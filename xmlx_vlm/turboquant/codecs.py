from __future__ import annotations
import math
from typing import Optional
import mlx.core as mx
import numpy as np

from .types import (
    DEFAULT_TURBOQUANT_SEED,
    _EPS,
    TurboQuantMSEState,
    TurboQuantProdState,
    TurboQuantPolarState,
    TurboQuantPolarProdState,
    TurboQuantSplitState,
)
from .kernels import (
    _compiled_integer_decode_kernel,
    _metal_butterfly_wht_forward,
    _metal_butterfly_wht_inverse,
    _fused_norot_quantize_kernel,
    _fused_mse_quantize_kernel,
    _fused_prod_quantize_kernel,
)
from .utils import (
    _validate_bits,
    _packed_width,
    _pack_lowbit,
    _unpack_lowbit,
    _rht_padded_dim,
    _rht_sign_vector,
    _rht_forward,
    _rht_inverse,
    _projection_matrix,
    _rotation_matrix,
    _codebook,
    _polar_angle_codebook,
    _polar_angle_pdf,
    _polar_levels,
    _polar_level_bits,
)

def dyn_unpack_lowbit(packed: mx.array, bits: int, length: int) -> mx.array:
    import sys
    tq = sys.modules.get("xmlx_vlm.turboquant")
    func = getattr(tq, "_unpack_lowbit", _unpack_lowbit) if tq else _unpack_lowbit
    return func(packed, bits, length)

class _TurboQuantMSECodec:
    enable_rht_padding = True

    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        # Use mx.hadamard_transform or fused Metal WHT for power-of-2 dims or padded non-power-of-2 dims
        is_pow2 = dim > 0 and (dim & (dim - 1)) == 0
        enable_padding = getattr(self, "enable_rht_padding", True)
        self.use_rht = dim > 0 and (is_pow2 or enable_padding)
        if self.use_rht:
            self.signs = _rht_sign_vector(dim, seed)
        else:
            self.signs = None
        # Dense rotation always available (needed for Metal helper fallbacks)
        self.rotation = _rotation_matrix(dim, seed)
        self.rotation_t = self.rotation.transpose() if dim > 0 else self.rotation
        self.codebook = _codebook(dim, bits)
        if bits > 0 and self.codebook.shape[0] > 1:
            self._midpoints = (self.codebook[:-1] + self.codebook[1:]) / 2
        else:
            self._midpoints = mx.zeros((0,), dtype=mx.float32)

    def _rotate_forward(self, x: mx.array) -> mx.array:
        if self.use_rht:
            return _rht_forward(x, self.signs)
        return mx.matmul(x, self.rotation_t)

    def _rotate_inverse(self, x: mx.array) -> mx.array:
        if self.use_rht:
            return _rht_inverse(x, self.signs)
        return mx.matmul(x, self.rotation)

    def _quantize_unit_with_estimate(
        self, unit_vectors: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.bits == 0:
            return (
                mx.zeros((*unit_vectors.shape[:-1], 0), dtype=mx.uint32),
                mx.zeros(unit_vectors.shape, dtype=mx.float32),
            )

        rotated = self._rotate_forward(unit_vectors)
        indices = mx.zeros(rotated.shape, dtype=mx.uint32)
        for m in range(self._midpoints.shape[0]):
            indices = indices + (rotated > self._midpoints[m]).astype(mx.uint32)
        packed = _pack_lowbit(indices, self.bits)
        estimated_rotated = mx.take(self.codebook, indices.astype(mx.int32), axis=0)
        return packed, self._rotate_inverse(estimated_rotated)

    def _quantize_unit(self, unit_vectors: mx.array) -> mx.array:
        """Quantize without computing the estimate (no wasted D×D rotation)."""
        if self.bits == 0:
            return mx.zeros((*unit_vectors.shape[:-1], 0), dtype=mx.uint32)
        rotated = self._rotate_forward(unit_vectors)
        indices = mx.zeros(rotated.shape, dtype=mx.uint32)
        for m in range(self._midpoints.shape[0]):
            indices = indices + (rotated > self._midpoints[m]).astype(mx.uint32)
        return _pack_lowbit(indices, self.bits)

    def _dequantize_unit(self, packed_indices: mx.array) -> mx.array:
        if self.bits == 0:
            return mx.zeros((*packed_indices.shape[:-1], self.dim), dtype=mx.float32)

        indices = dyn_unpack_lowbit(packed_indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.codebook, indices, axis=0)
        return self._rotate_inverse(rotated)

    def quantize(self, vectors: mx.array) -> TurboQuantMSEState:
        # Fast path for single-token decode: fused rotation + quantize + pack in 1 dispatch
        if vectors.shape[-2] == 1 and self.bits > 0:
            use_rht = self.use_rht
            kernel = _fused_mse_quantize_kernel(self.bits, use_rht=use_rht)
            if kernel is not None:
                D = self.dim
                D_padded = _rht_padded_dim(D) if use_rht else D
                flat = vectors.reshape(-1, D)
                BH = flat.shape[0]
                dims_per_lane = (D + 31) // 32
                packed_width = (D * self.bits + 31) // 32
                rot_input = self.signs if use_rht else self.rotation
                
                template = [
                    ("Dim", D),
                    ("DimsPerLane", dims_per_lane),
                    ("PackedWidth", packed_width),
                ]
                if use_rht:
                    template += [
                        ("DimPadded", D_padded),
                        ("DimsPerLanePadded", (D_padded + 31) // 32),
                    ]
                
                tg_x = D_padded if use_rht else D
                grid_x = tg_x * BH
                
                norms, packed = kernel(
                    inputs=[flat, rot_input, self._midpoints],
                    template=template,
                    grid=(grid_x, 1, 1),
                    threadgroup=(tg_x, 1, 1),
                    output_shapes=[(BH,), (BH, packed_width)],
                    output_dtypes=[mx.float16, mx.uint32],
                )
                orig_shape = vectors.shape[:-1]
                return TurboQuantMSEState(
                    norms.reshape(orig_shape),
                    packed.reshape(*orig_shape, packed_width),
                )

        vectors_f32 = vectors.astype(mx.float32)
        norms = mx.linalg.norm(vectors_f32, axis=-1)
        unit_vectors = vectors_f32 / mx.maximum(norms[..., None], _EPS)
        return TurboQuantMSEState(
            norms.astype(mx.float16),
            self._quantize_unit(unit_vectors),
        )

    def dequantize(self, state: TurboQuantMSEState) -> mx.array:
        unit_vectors = self._dequantize_unit(state.indices)
        return state.norms[..., None].astype(unit_vectors.dtype) * unit_vectors

    def prepare_queries(self, queries: mx.array) -> mx.array:
        return self._rotate_forward(queries)

    def score_prepared(
        self, prepared_queries: mx.array, state: TurboQuantMSEState
    ) -> mx.array:
        if prepared_queries.shape[-2] == 1:
            fast_scores = _metal_mse_score(
                prepared_queries.reshape(
                    prepared_queries.shape[0],
                    prepared_queries.shape[1],
                    prepared_queries.shape[2],
                    prepared_queries.shape[-1],
                ),
                state,
                self.bits,
                self.codebook,
            )
            if fast_scores is not None:
                return fast_scores

        indices = dyn_unpack_lowbit(state.indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.codebook, indices, axis=0)
        dots = mx.einsum("bhmld,bhtd->bhmlt", prepared_queries, rotated)
        return dots * state.norms.astype(mx.float32)[:, :, None, None, :]

    def score(self, queries: mx.array, state: TurboQuantMSEState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)

    def weighted_sum(self, weights: mx.array, state: TurboQuantMSEState) -> mx.array:
        if weights.shape[-2] == 1:
            fast_output = _metal_mse_weighted_sum(
                weights,
                state,
                self.bits,
                self.codebook,
                self.rotation,
            )
            if fast_output is not None:
                return fast_output

        indices = dyn_unpack_lowbit(state.indices, self.bits, self.dim).astype(mx.int32)
        rotated = mx.take(self.codebook, indices, axis=0)
        weighted_rot = mx.einsum(
            "bhmlt,bht,bhtd->bhmld",
            weights,
            state.norms.astype(mx.float32),
            rotated,
        )
        return self._rotate_inverse(weighted_rot)

    def weighted_sum_from_scores(
        self, scores: mx.array, state: TurboQuantMSEState
    ) -> mx.array:
        if not self.use_rht:
            fast_output = _metal_mse_weighted_sum_from_scores(
                scores,
                state,
                self.bits,
                self.codebook,
                self.rotation,
            )
            if fast_output is not None:
                return fast_output
        return self.weighted_sum(mx.softmax(scores, axis=-1), state)

    def weighted_sum_stats_from_scores(
        self, scores: mx.array, state: TurboQuantMSEState
    ) -> tuple[mx.array, mx.array, mx.array]:
        max_scores = mx.max(scores, axis=-1)
        # Metal kernel fast path: only for single-query decode (L=1)
        if scores.ndim == 5 and scores.shape[-2] == 1:
            max_scores_2d = max_scores.reshape(
                max_scores.shape[0],
                max_scores.shape[1],
                max_scores.shape[2],
            )
            fast_output = _metal_mse_weighted_sum_sum_from_scores(
                scores,
                state,
                self.bits,
                self.codebook,
                self.rotation,
                max_scores_2d,
            )
            if fast_output is not None:
                denom = mx.sum(mx.exp(scores - max_scores[..., None]), axis=-1)
                return fast_output, denom, max_scores

        weights = mx.exp(scores - max_scores[..., None])
        output = self.weighted_sum(weights, state)
        denom = mx.sum(weights, axis=-1)
        return output, denom, max_scores


class _PolarQuantUnitCodec:
    def __init__(self, dim: int, bits: int, seed: int):
        if not _is_power_of_two(dim):
            raise ValueError(
                f"PolarQuant requires a power-of-two dimension, got {dim}."
            )
        self.dim = dim
        self.bits = bits
        self.level_bits = _polar_level_bits(dim, bits)
        self.levels = len(self.level_bits)
        self.rotation = _rotation_matrix(dim, seed)
        self.rotation_t = self.rotation.transpose() if dim > 0 else self.rotation
        self.angle_codebooks = tuple(
            _polar_angle_codebook(level, level_bits)
            for level, level_bits in enumerate(self.level_bits, start=1)
        )
        self.cos_tables = tuple(mx.cos(codebook) for codebook in self.angle_codebooks)
        self.sin_tables = tuple(mx.sin(codebook) for codebook in self.angle_codebooks)

    def _quantize_level(self, angles: mx.array, level: int) -> mx.array:
        codebook = self.angle_codebooks[level - 1]
        diffs = mx.abs(angles[..., None] - codebook)
        if level == 1:
            diffs = mx.minimum(diffs, (2.0 * math.pi) - diffs)
        return mx.argmin(diffs, axis=-1).astype(mx.uint32)

    def _dequantize_preconditioned(self, state: TurboQuantPolarState) -> mx.array:
        radii = state.radii.astype(mx.float32)
        for bits, indices_packed, cos_table, sin_table in zip(
            reversed(self.level_bits),
            reversed(state.level_indices),
            reversed(self.cos_tables),
            reversed(self.sin_tables),
        ):
            angle_count = radii.shape[-1]
            indices = dyn_unpack_lowbit(indices_packed, bits, angle_count).astype(mx.int32)
            cosines = mx.take(cos_table, indices, axis=0)
            sines = mx.take(sin_table, indices, axis=0)
            radii = mx.stack([radii * cosines, radii * sines], axis=-1).reshape(
                (*radii.shape[:-1], radii.shape[-1] * 2)
            )
        return radii

    def quantize_unit_with_estimate(
        self, unit_vectors: mx.array, storage_dtype
    ) -> tuple[TurboQuantPolarState, mx.array]:
        preconditioned = mx.matmul(unit_vectors, self.rotation_t)
        radii = preconditioned
        packed_levels = []
        for level, bits in enumerate(self.level_bits, start=1):
            pairs = radii.reshape((*radii.shape[:-1], radii.shape[-1] // 2, 2))
            angles = mx.arctan2(pairs[..., 1], pairs[..., 0])
            if level == 1:
                angles = mx.where(angles < 0, angles + 2.0 * math.pi, angles)
            indices = self._quantize_level(angles, level)
            packed_levels.append(_pack_lowbit(indices, bits))
            radii = mx.linalg.norm(pairs, axis=-1)

        state = TurboQuantPolarState(
            radii.astype(storage_dtype),
            tuple(packed_levels),
        )
        approx_preconditioned = self._dequantize_preconditioned(state)
        approx_unit = mx.matmul(approx_preconditioned, self.rotation)
        return state, approx_unit

    def dequantize_unit(self, state: TurboQuantPolarState) -> mx.array:
        return mx.matmul(self._dequantize_preconditioned(state), self.rotation)

    def score_prepared(
        self, prepared_queries: mx.array, state: TurboQuantPolarState, norms: mx.array
    ) -> mx.array:
        if prepared_queries.shape[-2] == 1:
            fast_scores = _metal_polar_prod_score(
                prepared_queries.reshape(
                    prepared_queries.shape[0],
                    prepared_queries.shape[1],
                    prepared_queries.shape[2],
                    prepared_queries.shape[-1],
                ),
                TurboQuantPolarProdState(
                    norms,
                    state,
                    mx.zeros_like(norms),
                    mx.zeros((*norms.shape, 0), dtype=mx.uint32),
                ),
                self.level_bits,
                self.cos_tables,
                self.sin_tables,
            )
            if fast_scores is not None:
                return fast_scores

        approx_preconditioned = self._dequantize_preconditioned(state)
        dots = mx.einsum("bhmld,bhtd->bhmlt", prepared_queries, approx_preconditioned)
        return dots * norms.astype(mx.float32)[:, :, None, None, :]


class _TurboQuantPolarProdCodec:
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.polar_codec = _PolarQuantUnitCodec(dim, bits, seed)
        self.projection = _projection_matrix(dim, seed + 1)
        self.projection_t = self.projection.transpose() if dim > 0 else self.projection
        self.query_transform_t = (
            mx.concatenate([self.polar_codec.rotation_t, self.projection_t], axis=-1)
            if dim > 0
            else mx.zeros((0, 0), dtype=mx.float32)
        )
        self.scale = math.sqrt(math.pi / 2) / dim if dim > 0 else 0.0
        self.scale_array = mx.array([self.scale], dtype=mx.float32)

    def quantize(self, vectors: mx.array) -> TurboQuantPolarProdState:
        vectors_f32 = vectors.astype(mx.float32)
        norms = mx.linalg.norm(vectors_f32, axis=-1)
        unit_vectors = vectors_f32 / mx.maximum(norms[..., None], _EPS)

        polar_state, approx_unit = self.polar_codec.quantize_unit_with_estimate(
            unit_vectors,
            storage_dtype=vectors.dtype,
        )
        residual = unit_vectors - approx_unit
        residual_norms = mx.linalg.norm(residual, axis=-1)
        projected = mx.matmul(residual, self.projection_t)
        signs = mx.where(projected >= 0, 1, 0).astype(mx.uint32)

        return TurboQuantPolarProdState(
            norms.astype(mx.float16),
            polar_state,
            residual_norms.astype(mx.float16),
            _pack_lowbit(signs, 1),
        )

    def dequantize(self, state: TurboQuantPolarProdState) -> mx.array:
        polar_unit = self.polar_codec.dequantize_unit(state.polar_state)
        sign_bits = dyn_unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_unit = (
            self.scale
            * state.residual_norms[..., None].astype(mx.float32)
            * mx.matmul(signs, self.projection)
        )
        return state.norms[..., None].astype(mx.float32) * (polar_unit + qjl_unit)

    def prepare_queries(self, queries: mx.array) -> tuple[mx.array, mx.array]:
        transformed = mx.matmul(queries, self.query_transform_t)
        return transformed[..., : self.dim], transformed[..., self.dim :]

    def score_prepared(
        self,
        prepared_queries: tuple[mx.array, mx.array],
        state: TurboQuantPolarProdState,
    ) -> mx.array:
        polar_queries, proj_queries = prepared_queries
        if proj_queries.shape[-2] == 1:
            fast_scores = _metal_polar_turbo_score(
                polar_queries.reshape(
                    polar_queries.shape[0],
                    polar_queries.shape[1],
                    polar_queries.shape[2],
                    polar_queries.shape[-1],
                ),
                proj_queries.reshape(
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[-1],
                ),
                state,
                self.polar_codec.level_bits,
                self.polar_codec.cos_tables,
                self.polar_codec.sin_tables,
                self.scale_array,
            )
            if fast_scores is not None:
                return fast_scores

        polar_score = self.polar_codec.score_prepared(
            polar_queries,
            state.polar_state,
            state.norms,
        )

        if proj_queries.shape[-2] == 1:
            fast_qjl = _metal_qjl_score(
                proj_queries.reshape(
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[-1],
                ),
                state,
                self.scale_array,
            )
            if fast_qjl is not None:
                return polar_score + fast_qjl

        sign_bits = dyn_unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_score = (
            self.scale
            * state.residual_norms.astype(mx.float32)[:, :, None, None, :]
            * mx.einsum(
                "bhmld,bhtd->bhmlt",
                proj_queries,
                signs,
            )
        )

        norms = state.norms.astype(mx.float32)[:, :, None, None, :]
        return polar_score + norms * qjl_score

    def score(self, queries: mx.array, state: TurboQuantPolarProdState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)


class _TurboQuantProdCodec:
    def __init__(self, dim: int, bits: int, seed: int):
        self.dim = dim
        self.bits = bits
        self.mse_codec = _TurboQuantMSECodec(dim, max(bits - 1, 0), seed)
        self.projection = _projection_matrix(dim, seed + 1)
        self.projection_t = self.projection.transpose() if dim > 0 else self.projection
        self.query_transform_t = (
            mx.concatenate([self.mse_codec.rotation_t, self.projection_t], axis=-1)
            if dim > 0
            else mx.zeros((0, 0), dtype=mx.float32)
        )
        self.scale = math.sqrt(math.pi / 2) / dim if dim > 0 else 0.0
        self.scale_array = mx.array([self.scale], dtype=mx.float32)
        # Precompute for fused quantize: project rotated residual directly
        # combined_proj_t = (rotation @ projection_t).T = projection @ rotation_t
        if dim > 0:
            self._combined_proj_t = mx.matmul(
                self.projection, self.mse_codec.rotation_t
            )
        else:
            self._combined_proj_t = mx.zeros((0, 0), dtype=mx.float32)

    def quantize(self, vectors: mx.array) -> TurboQuantProdState:
        mse_bits = self.mse_codec.bits
        use_rht = self.mse_codec.use_rht
        # Fused Metal kernel for single-token decode
        if vectors.shape[-2] == 1 and mse_bits > 0 and 32 % mse_bits == 0:
            kernel = _fused_prod_quantize_kernel(mse_bits, use_rht=use_rht)
            if kernel is not None:
                D = self.dim
                D_padded = _rht_padded_dim(D) if use_rht else D
                flat = vectors.reshape(-1, D)
                BH = flat.shape[0]
                dims_per_lane = (D + 31) // 32
                mse_packed_width = (D * mse_bits + 31) // 32
                sign_packed_width = (D + 31) // 32
                rot_input = self.mse_codec.signs if use_rht else self.mse_codec.rotation
                template = [
                    ("Dim", D),
                    ("DimsPerLane", dims_per_lane),
                    ("MsePackedWidth", mse_packed_width),
                    ("SignPackedWidth", sign_packed_width),
                ]
                if use_rht:
                    template += [
                        ("DimPadded", D_padded),
                        ("DimsPerLanePadded", (D_padded + 31) // 32),
                    ]
                norms, mse_packed, res_norms, signs = kernel(
                    inputs=[
                        flat,
                        rot_input,
                        self.mse_codec._midpoints,
                        self.mse_codec.codebook,
                        self._combined_proj_t,
                    ],
                    template=template,
                    grid=(32, 1, BH),
                    threadgroup=(32, 1, 1),
                    output_shapes=[
                        (BH,),
                        (BH, mse_packed_width),
                        (BH,),
                        (BH, sign_packed_width),
                    ],
                    output_dtypes=[
                        mx.float16,
                        mx.uint32,
                        mx.float16,
                        mx.uint32,
                    ],
                )
                orig_shape = vectors.shape[:-1]  # (B, H, 1)
                return TurboQuantProdState(
                    norms.reshape(orig_shape),
                    mse_packed.reshape(*orig_shape, mse_packed_width),
                    res_norms.reshape(orig_shape),
                    signs.reshape(*orig_shape, sign_packed_width),
                )

        vectors_f32 = vectors.astype(mx.float32)
        norms = mx.linalg.norm(vectors_f32, axis=-1)
        unit_vectors = vectors_f32 / mx.maximum(norms[..., None], _EPS)

        mse_indices, mse_unit = self.mse_codec._quantize_unit_with_estimate(
            unit_vectors
        )
        residual = unit_vectors - mse_unit
        residual_norms = mx.linalg.norm(residual, axis=-1)
        projected = mx.matmul(residual, self.projection_t)
        signs = mx.where(projected >= 0, 1, 0).astype(mx.uint32)

        return TurboQuantProdState(
            norms.astype(mx.float16),
            mse_indices,
            residual_norms.astype(mx.float16),
            _pack_lowbit(signs, 1),
        )

    def dequantize(self, state: TurboQuantProdState) -> mx.array:
        mse_unit = self.mse_codec._dequantize_unit(state.mse_indices)
        sign_bits = dyn_unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_unit = (
            self.scale
            * state.residual_norms[..., None].astype(mx.float32)
            * mx.matmul(signs, self.projection)
        )
        return state.norms[..., None].astype(mx.float32) * (mse_unit + qjl_unit)

    def prepare_queries(self, queries: mx.array) -> tuple[mx.array, mx.array]:
        if self.mse_codec.use_rht:
            q_rot = self.mse_codec._rotate_forward(queries)
            q_proj = mx.matmul(queries, self.projection_t)
            return q_rot, q_proj
        transformed = mx.matmul(queries, self.query_transform_t)
        return transformed[..., : self.dim], transformed[..., self.dim :]

    def score_prepared(
        self,
        prepared_queries: tuple[mx.array, mx.array],
        state: TurboQuantProdState,
    ) -> mx.array:
        mse_queries, proj_queries = prepared_queries
        if proj_queries.shape[-2] == 1:
            fast_scores = _metal_prod_score(
                mse_queries.reshape(
                    mse_queries.shape[0],
                    mse_queries.shape[1],
                    mse_queries.shape[2],
                    mse_queries.shape[-1],
                ),
                proj_queries.reshape(
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[-1],
                ),
                state,
                self.mse_codec.bits,
                self.mse_codec.codebook,
                self.scale_array,
            )
            if fast_scores is not None:
                return fast_scores

        if self.mse_codec.bits > 0:
            mse_score = self.mse_codec.score_prepared(
                mse_queries,
                TurboQuantMSEState(state.norms, state.mse_indices),
            )
        else:
            mse_score = mx.zeros(
                (
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[3],
                    state.norms.shape[2],
                ),
                dtype=mx.float32,
            )

        if proj_queries.shape[-2] == 1:
            fast_qjl = _metal_qjl_score(
                proj_queries.reshape(
                    proj_queries.shape[0],
                    proj_queries.shape[1],
                    proj_queries.shape[2],
                    proj_queries.shape[-1],
                ),
                state,
                self.scale_array,
            )
            if fast_qjl is not None:
                return mse_score + fast_qjl

        sign_bits = dyn_unpack_lowbit(state.qjl_signs, 1, self.dim).astype(mx.float32)
        signs = sign_bits * 2.0 - 1.0
        qjl_score = (
            self.scale
            * state.residual_norms.astype(mx.float32)[:, :, None, None, :]
            * mx.einsum(
                "bhmld,bhtd->bhmlt",
                proj_queries,
                signs,
            )
        )

        norms = state.norms.astype(mx.float32)[:, :, None, None, :]
        return mse_score + norms * qjl_score

    def score(self, queries: mx.array, state: TurboQuantProdState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)


def _select_outlier_indices(
    tensor: mx.array, avg_bits: float
) -> tuple[np.ndarray, np.ndarray]:
    lower_bits = math.floor(avg_bits)
    upper_bits = math.ceil(avg_bits)
    if lower_bits == upper_bits:
        raise ValueError("Mixed-precision selection requires a fractional bit-width.")

    dim = tensor.shape[-1]
    high_count = int(round((avg_bits - lower_bits) * dim / (upper_bits - lower_bits)))
    high_count = max(1, min(dim - 1, high_count))

    scores = mx.mean(mx.abs(tensor.astype(mx.float32)), axis=(0, 1, 2))
    order = np.argsort(np.asarray(scores))
    high_idx = np.sort(order[-high_count:].astype(np.int32))
    low_mask = np.ones(dim, dtype=bool)
    low_mask[high_idx] = False
    low_idx = np.nonzero(low_mask)[0].astype(np.int32)
    return low_idx, high_idx


class _SplitCodec:
    def __init__(self, tensor: mx.array, bits: float, mode: str, seed: int):
        self.bits = bits
        self.mode = mode
        self.dim = tensor.shape[-1]
        self.lower_bits = math.floor(bits)
        self.upper_bits = math.ceil(bits)
        low_idx, high_idx = _select_outlier_indices(tensor, bits)
        self.low_idx = mx.array(low_idx, dtype=mx.int32)
        self.high_idx = mx.array(high_idx, dtype=mx.int32)

        concat_order = np.concatenate([low_idx, high_idx])
        self.restore_order = mx.array(np.argsort(concat_order), dtype=mx.int32)

        codec_cls = _TurboQuantProdCodec if mode == "prod" else _TurboQuantMSECodec
        self.low_codec = codec_cls(len(low_idx), self.lower_bits, seed)
        self.high_codec = codec_cls(len(high_idx), self.upper_bits, seed + 97)

        # Pre-build combined query transform for fused decode:
        # single (D, 2*dim_low + 2*dim_high) matrix replaces 2 takes + 2 matmuls
        if mode == "prod" and isinstance(self.low_codec, _TurboQuantProdCodec):
            dim = tensor.shape[-1]
            dl = len(low_idx)
            dh = len(high_idx)
            combined = mx.zeros((dim, 2 * dl + 2 * dh), dtype=mx.float32)
            combined[self.low_idx, :dl] = self.low_codec.query_transform_t[:, :dl]
            combined[self.low_idx, dl : 2 * dl] = self.low_codec.query_transform_t[
                :, dl:
            ]
            combined[self.high_idx, 2 * dl : 2 * dl + dh] = (
                self.high_codec.query_transform_t[:, :dh]
            )
            combined[self.high_idx, 2 * dl + dh :] = self.high_codec.query_transform_t[
                :, dh:
            ]
            self.combined_query_transform_t = combined
        else:
            self.combined_query_transform_t = None

    def quantize(self, tensor: mx.array) -> TurboQuantSplitState:
        low_tensor = mx.take(tensor, self.low_idx, axis=-1)
        high_tensor = mx.take(tensor, self.high_idx, axis=-1)
        return TurboQuantSplitState(
            self.low_codec.quantize(low_tensor),
            self.high_codec.quantize(high_tensor),
        )

    def dequantize(self, state: TurboQuantSplitState) -> mx.array:
        low_tensor = self.low_codec.dequantize(state.low)
        high_tensor = self.high_codec.dequantize(state.high)
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)

    def prepare_queries(self, queries: mx.array):
        low_tensor = mx.take(queries, self.low_idx, axis=-1)
        high_tensor = mx.take(queries, self.high_idx, axis=-1)
        return (
            self.low_codec.prepare_queries(low_tensor),
            self.high_codec.prepare_queries(high_tensor),
        )

    def score_prepared(self, prepared_queries, state: TurboQuantSplitState) -> mx.array:
        low_queries, high_queries = prepared_queries
        # Launch both sub-codec scores before any sync — enables GPU overlap
        low_scores = self.low_codec.score_prepared(low_queries, state.low)
        high_scores = self.high_codec.score_prepared(high_queries, state.high)
        return low_scores + high_scores

    def score(self, queries: mx.array, state: TurboQuantSplitState) -> mx.array:
        return self.score_prepared(self.prepare_queries(queries), state)

    def weighted_sum(self, weights: mx.array, state: TurboQuantSplitState) -> mx.array:
        low_tensor = self.low_codec.weighted_sum(weights, state.low)
        high_tensor = self.high_codec.weighted_sum(weights, state.high)
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)

    def weighted_sum_from_scores(
        self, scores: mx.array, state: TurboQuantSplitState
    ) -> mx.array:
        # Launch both before concat to enable overlap
        low_tensor = self.low_codec.weighted_sum_from_scores(scores, state.low)
        high_tensor = self.high_codec.weighted_sum_from_scores(scores, state.high)
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1)

    def weighted_sum_stats_from_scores(
        self, scores: mx.array, state: TurboQuantSplitState
    ) -> tuple[mx.array, mx.array, mx.array]:
        # Launch both before concat to enable overlap
        low_tensor, denom, max_scores = self.low_codec.weighted_sum_stats_from_scores(
            scores, state.low
        )
        high_tensor, _, _ = self.high_codec.weighted_sum_stats_from_scores(
            scores, state.high
        )
        merged = mx.concatenate([low_tensor, high_tensor], axis=-1)
        return mx.take(merged, self.restore_order, axis=-1), denom, max_scores


def _build_codec(tensor: mx.array, bits: float, mode: str, seed: int):
    bits = _validate_bits(bits)
    if math.isclose(bits, round(bits), abs_tol=1e-6):
        codec_cls = _TurboQuantProdCodec if mode == "prod" else _TurboQuantMSECodec
        return codec_cls(tensor.shape[-1], int(round(bits)), seed)
    return _SplitCodec(tensor, bits, mode, seed)


