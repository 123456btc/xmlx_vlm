from __future__ import annotations
import math
from functools import lru_cache
import mlx.core as mx
from .base import _metal_available

def _mse_weighted_rot_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto dim_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        if (dim_idx >= Dim) {
            return;
        }

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = weights_shape[2];
        auto b = n / (kv_heads * repeat_count);
        auto rem = n % (kv_heads * repeat_count);
        auto h = rem / repeat_count;
        auto repeat_idx = rem % repeat_count;

        auto weights_ptr = weights + ((b * kv_heads + h) * repeat_count + repeat_idx) * token_count;
        auto norms_ptr = norms + (b * kv_heads + h) * token_count;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

        float acc = 0.0f;
        for (int t = lane; t < token_count; t += 32) {
            auto token_ptr = packed_ptr + t * PackedWidth;
            int bit_offset = dim_idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = token_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= token_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(weights_ptr[t])
                * static_cast<float>(norms_ptr[t])
                * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * Dim + dim_idx] = acc;
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_mse_weighted_rot",
        input_names=["weights", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )

def _mse_weighted_rot_repeat_kernel(repeat_count: int):
    if not _metal_available() or repeat_count <= 1:
        return None

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto dim_idx = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        if (dim_idx >= Dim) {",
        "            return;",
        "        }",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = weights_shape[2];",
        "        auto b = n / kv_heads;",
        "        auto h = n % kv_heads;",
        "",
        "        auto weights_base = weights + ((b * kv_heads + h) * repeat_count) * token_count;",
        "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
        "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
        "",
        "        int bit_offset = dim_idx * Bits;",
        "        int word_idx = bit_offset / 32;",
        "        int offset = bit_offset % 32;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float acc_{r} = 0.0f;")
    lines += [
        "",
        "        for (int t = lane; t < token_count; t += 32) {",
        "            auto token_ptr = packed_ptr + t * PackedWidth;",
        "            uint value = token_ptr[word_idx] >> offset;",
        "            int spill = offset + Bits - 32;",
        "            if (spill > 0) {",
        "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
        "            }",
        "            value &= ((1u << Bits) - 1u);",
        "            float code = codebook[value];",
        "            float norm = static_cast<float>(norms_ptr[t]);",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            acc_{r} += static_cast<float>(weights_base[{r} * token_count + t]) * norm * code;"
        )
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float acc_sum_{r} = simd_sum(acc_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * repeat_count + {r}) * Dim + dim_idx] = acc_sum_{r};"
        )
    lines += [
        "        }",
    ]

    source = "\n".join(lines)
    return mx.fast.metal_kernel(
        name=f"turboquant_mse_weighted_rot_repeat_{repeat_count}",
        input_names=["weights", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )

def _mse_scores_weighted_rot_repeat_kernel(repeat_count: int):
    """Single-pass fused softmax + weighted sum kernel.

    Takes precomputed max_scores to avoid a separate token-dimension pass.
    """
    if not _metal_available() or repeat_count <= 1:
        return None

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto dim_idx = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        if (dim_idx >= Dim) {",
        "            return;",
        "        }",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = scores_shape[2];",
        "        auto b = n / kv_heads;",
        "        auto h = n % kv_heads;",
        "",
        "        auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;",
        "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
        "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
        "        auto max_base = max_scores + (b * kv_heads + h) * repeat_count;",
        "",
        "        int bit_offset = dim_idx * Bits;",
        "        int word_idx = bit_offset / 32;",
        "        int offset = bit_offset % 32;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"        float max_score_{r} = static_cast<float>(max_base[{r}]);"
        )
        lines.append(f"        float acc_{r} = 0.0f;")
        lines.append(f"        float denom_{r} = 0.0f;")
    lines += [
        "",
        "        for (int t = lane; t < token_count; t += 32) {",
        "            auto token_ptr = packed_ptr + t * PackedWidth;",
        "            uint value = token_ptr[word_idx] >> offset;",
        "            int spill = offset + Bits - 32;",
        "            if (spill > 0) {",
        "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
        "            }",
        "            value &= ((1u << Bits) - 1u);",
        "            float code = codebook[value];",
        "            float norm = static_cast<float>(norms_ptr[t]);",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            float weight_{r} = exp(static_cast<float>(scores_base[{r} * token_count + t]) - max_score_{r});"
        )
        lines.append(f"            acc_{r} += weight_{r} * norm * code;")
        lines.append(f"            denom_{r} += weight_{r};")
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float acc_sum_{r} = simd_sum(acc_{r});")
        lines.append(f"        float denom_sum_{r} = simd_sum(denom_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * repeat_count + {r}) * Dim + dim_idx] ="
        )
        lines.append(f"                acc_sum_{r} / max(denom_sum_{r}, 1e-6f);")
    lines += [
        "        }",
    ]

    source = "\n".join(lines)
    return mx.fast.metal_kernel(
        name=f"turboquant_mse_scores_weighted_rot_repeat_{repeat_count}",
        input_names=["scores", "norms", "packed", "codebook", "max_scores"],
        output_names=["out"],
        source=source,
    )

def _mse_scores_weighted_rot_sum_repeat_kernel(repeat_count: int):
    """Single-pass kernel for unnormalized weighted sum (used in chunked attention).

    Takes precomputed max_scores to avoid a separate token-dimension pass.
    """
    if not _metal_available() or repeat_count <= 1:
        return None

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto dim_idx = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        if (dim_idx >= Dim) {",
        "            return;",
        "        }",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = scores_shape[2];",
        "        auto b = n / kv_heads;",
        "        auto h = n % kv_heads;",
        "",
        "        auto scores_base = scores + ((b * kv_heads + h) * repeat_count) * token_count;",
        "        auto norms_ptr = norms + (b * kv_heads + h) * token_count;",
        "        auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;",
        "        auto max_base = max_scores + (b * kv_heads + h) * repeat_count;",
        "",
        "        int bit_offset = dim_idx * Bits;",
        "        int word_idx = bit_offset / 32;",
        "        int offset = bit_offset % 32;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"        float max_score_{r} = static_cast<float>(max_base[{r}]);"
        )
        lines.append(f"        float acc_{r} = 0.0f;")
    lines += [
        "",
        "        for (int t = lane; t < token_count; t += 32) {",
        "            auto token_ptr = packed_ptr + t * PackedWidth;",
        "            uint value = token_ptr[word_idx] >> offset;",
        "            int spill = offset + Bits - 32;",
        "            if (spill > 0) {",
        "                value |= token_ptr[word_idx + 1] << (Bits - spill);",
        "            }",
        "            value &= ((1u << Bits) - 1u);",
        "            float code = codebook[value];",
        "            float norm = static_cast<float>(norms_ptr[t]);",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            float weight_{r} = exp(static_cast<float>(scores_base[{r} * token_count + t]) - max_score_{r});"
        )
        lines.append(f"            acc_{r} += weight_{r} * norm * code;")
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        float acc_sum_{r} = simd_sum(acc_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * repeat_count + {r}) * Dim + dim_idx] = acc_sum_{r};"
        )
    lines += [
        "        }",
    ]

    source = "\n".join(lines)
    return mx.fast.metal_kernel(
        name=f"turboquant_mse_scores_weighted_rot_sum_repeat_{repeat_count}",
        input_names=["scores", "norms", "packed", "codebook", "max_scores"],
        output_names=["out"],
        source=source,
    )

def _metal_butterfly_wht_forward(
    shared_name: str,
    sign_name: str,
    temp_name: str,
    dims_per_lane_padded: str = "DimsPerLanePadded",
):
    """Generate Metal code for in-place RHT forward in shared memory.
    RHT forward = WHT(signs * x) / sqrt(DimPadded).
    Requires: shared[DimPadded], sign_vec[Dim], temp[DimsPerLanePadded] (thread-local).
    Returns list of Metal source lines."""
    return [
        f"        // RHT forward: apply signs, butterfly WHT, scale",
        f"        for (int i = 0, d = lane; i < {dims_per_lane_padded}; d += 32, i++)",
        f"            if (d < Dim) {shared_name}[d] *= {sign_name}[d];",
        f"        threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"",
        f"        for (int stride = 1; stride < DimPadded; stride *= 2) {{",
        f"            for (int i = 0, d = lane; i < {dims_per_lane_padded}; d += 32, i++) {{",
        f"                if (d < DimPadded) {{",
        f"                    int pair = d ^ stride;",
        f"                    float a = {shared_name}[min(d, pair)];",
        f"                    float b = {shared_name}[max(d, pair)];",
        f"                    {temp_name}[i] = (d < pair) ? (a + b) : (a - b);",
        f"                }}",
        f"            }}",
        f"            threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"            for (int i = 0, d = lane; i < {dims_per_lane_padded}; d += 32, i++)",
        f"                if (d < DimPadded) {shared_name}[d] = {temp_name}[i];",
        f"            threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"        }}",
        f"",
        f"        // Scale by 1/sqrt(DimPadded)",
        f"        float rht_scale = 1.0f / sqrt((float)DimPadded);",
    ]

def _metal_butterfly_wht_inverse(
    shared_name: str,
    sign_name: str,
    temp_name: str,
    dims_per_lane_padded: str = "DimsPerLanePadded",
):
    """Generate Metal code for in-place RHT inverse in shared memory.
    RHT inverse = signs * WHT(x) / sqrt(DimPadded).
    Same butterfly as forward, then multiply by signs."""
    return [
        f"        // RHT inverse: butterfly WHT, then apply signs and scale",
        f"        for (int stride = 1; stride < DimPadded; stride *= 2) {{",
        f"            for (int i = 0, d = lane; i < {dims_per_lane_padded}; d += 32, i++) {{",
        f"                if (d < DimPadded) {{",
        f"                    int pair = d ^ stride;",
        f"                    float a = {shared_name}[min(d, pair)];",
        f"                    float b = {shared_name}[max(d, pair)];",
        f"                    {temp_name}[i] = (d < pair) ? (a + b) : (a - b);",
        f"                }}",
        f"            }}",
        f"            threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"            for (int i = 0, d = lane; i < {dims_per_lane_padded}; d += 32, i++)",
        f"                if (d < DimPadded) {shared_name}[d] = {temp_name}[i];",
        f"            threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"        }}",
        f"",
        f"        // Scale and apply signs",
        f"        float rht_scale = 1.0f / sqrt((float)DimPadded);",
        f"        for (int i = 0, d = lane; i < {dims_per_lane_padded}; d += 32, i++)",
        f"            if (d < Dim) {shared_name}[d] *= rht_scale * {sign_name}[d];",
        f"        threadgroup_barrier(mem_flags::mem_threadgroup);",
    ]

