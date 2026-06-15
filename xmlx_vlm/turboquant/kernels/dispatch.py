from __future__ import annotations
import math
from functools import lru_cache
import mlx.core as mx
from .base import _metal_available
from .scoring import (
    _mse_score_kernel,
    _mse_score_tiled_kernel,
    _qjl_score_kernel,
    _prod_score_kernel,
    _prod_score_repeat_kernel,
    _polar_prod_score_kernel,
    _polar_turbo_score_repeat_kernel,
)
from .rotation import (
    _mse_weighted_rot_kernel,
    _mse_weighted_rot_repeat_kernel,
    _mse_scores_weighted_rot_repeat_kernel,
    _mse_scores_weighted_rot_sum_repeat_kernel,
)

def _metal_mse_score(
    q_rot: mx.array,
    state: TurboQuantMSEState,
    bits: int,
    codebook: mx.array,
) -> Optional[mx.array]:
    if (
        bits <= 0
        or not _metal_available()
        or q_rot.ndim != 4
        or state.norms.shape[2] == 0
    ):
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]
    dims_per_lane = (D + 31) // 32

    # Tiled kernel: preload queries into registers, loop over token tiles.
    tiled_kernel = _mse_score_tiled_kernel(R)
    if tiled_kernel is not None:
        tok_tile_size = 64
        num_tok_tiles = (T + tok_tile_size - 1) // tok_tile_size
        scores = tiled_kernel(
            inputs=[q_rot, state.norms, state.indices, codebook],
            template=[
                ("Dim", D),
                ("DimsPerLane", dims_per_lane),
                ("Bits", bits),
                ("PackedWidth", state.indices.shape[-1]),
                ("RepeatCount", R),
                ("TokTileSize", tok_tile_size),
            ],
            grid=(32, 1, B * H * num_tok_tiles),
            threadgroup=(32, 1, 1),
            output_shapes=[(B, H, R, T)],
            output_dtypes=[mx.float32],
        )[0]
        return mx.expand_dims(scores, axis=3)

    kernel = _mse_score_kernel()
    if kernel is None:
        return None

    scores = kernel(
        inputs=[q_rot, state.norms, state.indices, codebook],
        template=[
            ("Dim", D),
            ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)

def _metal_qjl_score(
    q_proj: mx.array,
    state: TurboQuantProdState,
    scale: mx.array,
) -> Optional[mx.array]:
    if not _metal_available() or q_proj.ndim != 4 or state.norms.shape[2] == 0:
        return None

    kernel = _qjl_score_kernel()
    if kernel is None:
        return None

    B, H, R, D = q_proj.shape
    T = state.norms.shape[2]
    scores = kernel(
        inputs=[
            q_proj,
            state.norms,
            state.residual_norms,
            state.qjl_signs,
            scale,
        ],
        template=[
            ("Dim", D),
            ("PackedWidth", state.qjl_signs.shape[-1]),
        ],
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)

def _metal_prod_score(
    q_rot: mx.array,
    q_proj: mx.array,
    state: TurboQuantProdState,
    mse_bits: int,
    codebook: mx.array,
    scale: mx.array,
) -> Optional[mx.array]:
    if (
        mse_bits <= 0
        or not _metal_available()
        or q_rot.ndim != 4
        or q_proj.ndim != 4
        or state.norms.shape[2] == 0
    ):
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]
    if R > 1:
        kernel = _prod_score_repeat_kernel(R)
        if kernel is not None:
            scores = kernel(
                inputs=[
                    q_rot,
                    q_proj,
                    state.norms,
                    state.residual_norms,
                    state.mse_indices,
                    state.qjl_signs,
                    codebook,
                    scale,
                ],
                template=[
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("MseBits", mse_bits),
                    ("MsePackedWidth", state.mse_indices.shape[-1]),
                    ("SignPackedWidth", state.qjl_signs.shape[-1]),
                ],
                grid=(32, 1, B * H * T),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, H, R, T)],
                output_dtypes=[mx.float32],
            )[0]
            return mx.expand_dims(scores, axis=3)

    kernel = _prod_score_kernel()
    if kernel is None:
        return None

    scores = kernel(
        inputs=[
            q_rot,
            q_proj,
            state.norms,
            state.residual_norms,
            state.mse_indices,
            state.qjl_signs,
            codebook,
            scale,
        ],
        template=[
            ("Dim", D),
            ("MseBits", mse_bits),
            ("MsePackedWidth", state.mse_indices.shape[-1]),
            ("SignPackedWidth", state.qjl_signs.shape[-1]),
        ],
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)

def _metal_polar_prod_score(
    q_rot: mx.array,
    state: TurboQuantPolarProdState,
    level_bits: tuple[int, ...],
    cos_tables: tuple[mx.array, ...],
    sin_tables: tuple[mx.array, ...],
) -> Optional[mx.array]:
    if (
        not _metal_available()
        or q_rot.ndim != 4
        or state.norms.shape[2] == 0
        or len(level_bits) == 0
    ):
        return None

    kernel = _polar_prod_score_kernel(level_bits)
    if kernel is None:
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]
    levels = len(level_bits)
    inputs = [q_rot, state.norms, state.polar_state.radii]
    inputs.extend(level for level in state.polar_state.level_indices)
    for cos_table, sin_table in zip(cos_tables, sin_tables):
        inputs.extend([cos_table, sin_table])

    template = [
        ("Dim", D),
        ("Levels", levels),
        ("BlockCount", state.polar_state.radii.shape[-1]),
    ]
    for level_idx, level in enumerate(state.polar_state.level_indices, start=1):
        template.append((f"PackedWidth{level_idx}", level.shape[-1]))

    scores = kernel(
        inputs=inputs,
        template=template,
        grid=(32, R, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)

def _metal_polar_turbo_score(
    q_rot: mx.array,
    q_proj: mx.array,
    state: TurboQuantPolarProdState,
    level_bits: tuple[int, ...],
    cos_tables: tuple[mx.array, ...],
    sin_tables: tuple[mx.array, ...],
    scale: mx.array,
) -> Optional[mx.array]:
    if (
        not _metal_available()
        or q_rot.ndim != 4
        or q_proj.ndim != 4
        or q_rot.shape != q_proj.shape
        or state.norms.shape[2] == 0
        or len(level_bits) == 0
    ):
        return None

    B, H, R, D = q_rot.shape
    T = state.norms.shape[2]
    levels = len(level_bits)
    kernel = _polar_turbo_score_repeat_kernel(level_bits, R)
    if kernel is None:
        return None

    inputs = [q_rot, q_proj, state.norms, state.polar_state.radii]
    inputs.extend(level for level in state.polar_state.level_indices)
    inputs.extend(
        [
            state.residual_norms,
            state.qjl_signs,
            scale,
        ]
    )
    for cos_table, sin_table in zip(cos_tables, sin_tables):
        inputs.extend([cos_table, sin_table])

    template = [
        ("Dim", D),
        ("Levels", levels),
        ("RepeatCount", R),
        ("BlockCount", state.polar_state.radii.shape[-1]),
        ("SignPackedWidth", state.qjl_signs.shape[-1]),
    ]
    for level_idx, level in enumerate(state.polar_state.level_indices, start=1):
        template.append((f"PackedWidth{level_idx}", level.shape[-1]))

    scores = kernel(
        inputs=inputs,
        template=template,
        grid=(32, 1, B * H * T),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.expand_dims(scores, axis=3)

def _metal_mse_weighted_sum(
    weights: mx.array,
    state: TurboQuantMSEState,
    bits: int,
    codebook: mx.array,
    rotation: mx.array,
) -> Optional[mx.array]:
    if (
        bits <= 0
        or not _metal_available()
        or weights.ndim != 5
        or weights.shape[-2] != 1
        or state.norms.shape[2] == 0
    ):
        return None

    weights_2d = weights.reshape(
        weights.shape[0],
        weights.shape[1],
        weights.shape[2],
        weights.shape[-1],
    )
    B, H, R, T = weights_2d.shape
    D = rotation.shape[0]
    if R > 1:
        kernel = _mse_weighted_rot_repeat_kernel(R)
        if kernel is not None:
            weighted_rot = kernel(
                inputs=[
                    weights_2d,
                    state.norms,
                    state.indices,
                    codebook,
                ],
                template=[
                    ("Dim", D),
                    ("RepeatCount", R),
                    ("Bits", bits),
                    ("PackedWidth", state.indices.shape[-1]),
                ],
                grid=(32, D, B * H),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, H, R, D)],
                output_dtypes=[mx.float32],
            )[0]
            output = mx.matmul(weighted_rot, rotation)
            return mx.expand_dims(output, axis=3)

    kernel = _mse_weighted_rot_kernel()
    if kernel is None:
        return None

    weighted_rot = kernel(
        inputs=[
            weights_2d,
            state.norms,
            state.indices,
            codebook,
        ],
        template=[
            ("Dim", D),
            ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, D, B * H * R),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, D)],
        output_dtypes=[mx.float32],
    )[0]
    output = mx.matmul(weighted_rot, rotation)
    return mx.expand_dims(output, axis=3)

def _metal_mse_weighted_sum_from_scores(
    scores: mx.array,
    state: TurboQuantMSEState,
    bits: int,
    codebook: mx.array,
    rotation: mx.array,
) -> Optional[mx.array]:
    if (
        bits <= 0
        or not _metal_available()
        or scores.ndim != 5
        or scores.shape[-2] != 1
        or state.norms.shape[2] == 0
    ):
        return None

    scores_2d = scores.reshape(
        scores.shape[0],
        scores.shape[1],
        scores.shape[2],
        scores.shape[-1],
    )
    B, H, R, T = scores_2d.shape
    if R <= 1:
        return None

    kernel = _mse_scores_weighted_rot_repeat_kernel(R)
    if kernel is None:
        return None

    # Precompute max scores on the host to avoid a second pass in the kernel
    max_scores = mx.max(scores_2d, axis=-1)  # (B, H, R)

    D = rotation.shape[0]
    weighted_rot = kernel(
        inputs=[
            scores_2d,
            state.norms,
            state.indices,
            codebook,
            max_scores,
        ],
        template=[
            ("Dim", D),
            ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, D, B * H),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, D)],
        output_dtypes=[mx.float32],
    )[0]
    output = mx.matmul(weighted_rot, rotation)
    return mx.expand_dims(output, axis=3)

def _metal_mse_weighted_sum_sum_from_scores(
    scores: mx.array,
    state: TurboQuantMSEState,
    bits: int,
    codebook: mx.array,
    rotation: mx.array,
    max_scores: mx.array,
) -> Optional[mx.array]:
    if (
        bits <= 0
        or not _metal_available()
        or scores.ndim != 5
        or scores.shape[-2] != 1
        or state.norms.shape[2] == 0
    ):
        return None

    scores_2d = scores.reshape(
        scores.shape[0],
        scores.shape[1],
        scores.shape[2],
        scores.shape[-1],
    )
    B, H, R, T = scores_2d.shape
    if R <= 1:
        return None

    kernel = _mse_scores_weighted_rot_sum_repeat_kernel(R)
    if kernel is None:
        return None

    # max_scores shape: (B, H, R) — already precomputed by caller
    D = rotation.shape[0]
    weighted_rot = kernel(
        inputs=[
            scores_2d,
            state.norms,
            state.indices,
            codebook,
            max_scores,
        ],
        template=[
            ("Dim", D),
            ("Bits", bits),
            ("PackedWidth", state.indices.shape[-1]),
        ],
        grid=(32, D, B * H),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, R, D)],
        output_dtypes=[mx.float32],
    )[0]
    output = mx.matmul(weighted_rot, rotation)
    return mx.expand_dims(output, axis=3)

