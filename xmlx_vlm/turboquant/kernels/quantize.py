from __future__ import annotations
import math
from functools import lru_cache
import mlx.core as mx
from .base import _metal_available
from .rotation import _metal_butterfly_wht_forward

def _pack_lowbit_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto word = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= values_shape[0] || word >= PackedWidth) {
            return;
        }

        auto values_ptr = values + row * Length;
        uint packed_word = 0u;
        int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
        int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

        for (int idx = start; idx < end; ++idx) {
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = values_ptr[idx] & ((1u << Bits) - 1u);
            if (word_idx == word) {
                packed_word |= value << offset;
            }
            if (word_idx + 1 == word) {
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    packed_word |= value >> (Bits - spill);
                }
            }
        }

        out[row * PackedWidth + word] = packed_word;
    """
    return mx.fast.metal_kernel(
        name="turboquant_pack_lowbit",
        input_names=["values"],
        output_names=["out"],
        source=source,
    )

def _unpack_lowbit_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto idx = thread_position_in_grid.x;
        auto row = thread_position_in_grid.y;

        if (row >= packed_shape[0] || idx >= Length) {
            return;
        }

        auto packed_ptr = packed + row * PackedWidth;
        int bit_offset = idx * Bits;
        int word_idx = bit_offset / 32;
        int offset = bit_offset % 32;
        uint value = packed_ptr[word_idx] >> offset;
        int spill = offset + Bits - 32;
        if (spill > 0) {
            value |= packed_ptr[word_idx + 1] << (Bits - spill);
        }
        out[row * Length + idx] = value & ((1u << Bits) - 1u);
    """
    return mx.fast.metal_kernel(
        name="turboquant_unpack_lowbit",
        input_names=["packed"],
        output_names=["out"],
        source=source,
    )

def _fused_kv_quantize_kernel(key_bits: int, val_bits: int):
    """Fused key+value quantize in 1 dispatch."""
    if not _metal_available() or key_bits <= 0 or val_bits <= 0:
        return None

    k_midpoints = (1 << key_bits) - 1
    v_midpoints = (1 << val_bits) - 1
    k_mask = (1 << key_bits) - 1
    v_mask = (1 << val_bits) - 1

    source = f"""
        auto d = thread_position_in_threadgroup.x;
        auto bh = threadgroup_position_in_grid.x;
        auto is_val = threadgroup_position_in_grid.y;  // 0=key, 1=value
        auto sg_id = simdgroup_index_in_threadgroup;
        auto sg_lid = thread_index_in_simdgroup;

        // Select key or value params based on grid.y
        int bits = is_val ? {val_bits} : {key_bits};
        int n_mid = is_val ? {v_midpoints} : {k_midpoints};
        uint idx_mask = is_val ? {v_mask}u : {k_mask}u;
        int pw = is_val ? VPackedWidth : KPackedWidth;

        // Step 1: Load vector element
        float v;
        if (is_val)
            v = (d < Dim) ? static_cast<float>(val_vectors[bh * Dim + d]) : 0.0f;
        else
            v = (d < Dim) ? static_cast<float>(key_vectors[bh * Dim + d]) : 0.0f;

        // Compute norm (2-stage cross-simdgroup reduction)
        float sq = v * v;
        float sg_sum = simd_sum(sq);
        threadgroup float sg_norms[8];
        if (d < 8) sg_norms[d] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_lid == 0 && sg_id < 8) sg_norms[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_sq = (sg_id == 0 && sg_lid < 8) ? sg_norms[sg_lid] : 0.0f;
        total_sq = simd_sum(total_sq);
        if (sg_id == 0 && sg_lid == 0) sg_norms[0] = total_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float norm = sqrt(sg_norms[0]);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        if (d == 0) {{
            if (is_val) out_val_norms[bh] = half(norm);
            else out_key_norms[bh] = half(norm);
        }}

        // Step 2: Unit vector → shared
        threadgroup float shared[Dim];
        if (d < Dim) shared[d] = v * inv_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Rotate (1 dim per thread, 256 FMAs)
        float rotated = 0.0f;
        if (d < Dim) {{
            auto row = is_val ? (val_rotation + d * Dim) : (key_rotation + d * Dim);
            for (int j = 0; j < (int)Dim; j++)
                rotated += shared[j] * row[j];
        }}

        // Step 4: Comparison-based quantize
        threadgroup uint shared_idx[Dim];
        uint idx = 0;
        if (d < Dim) {{
            if (is_val) {{
                for (int m = 0; m < n_mid; m++)
                    idx += (rotated > val_midpoints[m]) ? 1u : 0u;
            }} else {{
                for (int m = 0; m < n_mid; m++)
                    idx += (rotated > key_midpoints[m]) ? 1u : 0u;
            }}
            shared_idx[d] = idx;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Pack indices — thread-per-word, race-free, no atomics.
        // Each thread d (d < pw) walks the dims whose
        // [i*bits, (i+1)*bits) range intersects [32d, 32(d+1)) and
        // accumulates them into a private register, then writes the
        // word once. `bits` and `pw` are runtime-uniform here so the
        // loop bound is well-defined per dispatch.
        if (d < pw) {{
            uint w_val = 0u;
            int word_start = (int)d * 32;
            int i_min = word_start / bits;
            int i_max = (word_start + 31) / bits;
            if (i_max >= (int)Dim) i_max = (int)Dim - 1;
            for (int i = i_min; i <= i_max; i++) {{
                uint idx_val = shared_idx[i] & idx_mask;
                int bit_off = i * bits - word_start;
                if (bit_off >= 0) {{
                    w_val |= idx_val << bit_off;
                }} else {{
                    w_val |= idx_val >> (-bit_off);
                }}
            }}
            if (is_val)
                out_val_packed[bh * pw + d] = w_val;
            else
                out_key_packed[bh * pw + d] = w_val;
        }}
    """

    return mx.fast.metal_kernel(
        name=f"turboquant_fused_kv_quantize_k{key_bits}_v{val_bits}",
        input_names=[
            "key_vectors",
            "val_vectors",
            "key_rotation",
            "val_rotation",
            "key_midpoints",
            "val_midpoints",
        ],
        output_names=[
            "out_key_norms",
            "out_key_packed",
            "out_val_norms",
            "out_val_packed",
        ],
        source=source,
    )

def _fused_norot_quantize_kernel(bits: int):
    """Quantize pre-rotated vectors: comparison + pack. No rotation inside.
    Used with mx.hadamard_transform for external rotation. TG=Dim."""
    if not _metal_available() or bits <= 0:
        return None

    num_midpoints = (1 << bits) - 1
    mask = (1 << bits) - 1

    # Pack step has to combine `Dim` low-bit indices into `PackedWidth`
    # 32-bit words. The original kernel had every dim-thread write
    # `packed_shared[w] |= idx_val << shift`, but `|=` on threadgroup
    # memory is *not* atomic on Metal, so dim-threads writing to the same
    # word raced and only one contribution per word survived (every other
    # slot was silently zeroed). The result was decode-time KV cache
    # corruption that grew worse at higher bit-widths.
    #
    # An earlier attempt swapped the buffer to `threadgroup atomic_uint`
    # + `atomic_fetch_or_explicit`, which is correct but ran into Metal
    # GPU watchdog hangs on the high-contention cases (e.g. bits=2,
    # Dim=128 → 16 dim-threads contending for the same word).
    #
    # The race-free *and* atomic-free fix is `thread-per-word packing`:
    # only threads with `d < PackedWidth` participate, and each one
    # walks exactly the dims that touch its word, OR-ing them into a
    # private register. No threadgroup memory is shared during the pack,
    # so there is neither a race nor any atomic contention.
    source = f"""
        auto d = thread_position_in_threadgroup.x;
        auto bh = threadgroup_position_in_grid.x;

        // Read pre-rotated, pre-normalized value
        float val = (d < Dim) ? static_cast<float>(rotated[bh * Dim + d]) : 0.0f;

        // Comparison-based quantize
        threadgroup uint shared_idx[Dim];
        uint idx = 0;
        if (d < Dim) {{
            for (int m = 0; m < {num_midpoints}; m++)
                idx += (val > midpoints[m]) ? 1u : 0u;
            shared_idx[d] = idx;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pack indices: thread d builds word d in a private register
        // and writes it once (no shared-memory race, no atomics).
        // Walks every dim whose [i*bits, (i+1)*bits) range intersects
        // [32d, 32(d+1)), including a single spill from the previous
        // word for non-32-aligned bit-widths (3-bit, 5-bit, etc.).
        if (d < PackedWidth) {{
            uint w_val = 0u;
            int word_start = (int)d * 32;
            int i_min = word_start / {bits};
            int i_max = (word_start + 31) / {bits};
            if (i_max >= (int)Dim) i_max = (int)Dim - 1;
            for (int i = i_min; i <= i_max; i++) {{
                uint idx_val = shared_idx[i] & {mask}u;
                int bit_off = i * {bits} - word_start;
                if (bit_off >= 0) {{
                    w_val |= idx_val << bit_off;
                }} else {{
                    w_val |= idx_val >> (-bit_off);
                }}
            }}
            out[bh * PackedWidth + d] = w_val;
        }}
    """

    return mx.fast.metal_kernel(
        name=f"turboquant_norot_quantize_{bits}",
        input_names=["rotated", "midpoints"],
        output_names=["out"],
        source=source,
    )

def _fused_mse_quantize_kernel(bits: int, use_rht: bool = False):
    """Fused MSE quantize: norm + rotate + quantize + pack in 1 dispatch."""
    if not _metal_available() or bits <= 0:
        return None
    num_entries = 1 << bits
    num_midpoints = num_entries - 1
    mask = num_entries - 1

    if use_rht:
        source = f"""
        auto d = thread_position_in_threadgroup.x;  // 0..DimPadded-1
        auto bh = threadgroup_position_in_grid.x;
        auto sg_id = simdgroup_index_in_threadgroup;
        auto sg_lid = thread_index_in_simdgroup;
        auto vec_ptr = vectors + bh * Dim;

        // Step 1: Load & compute norm
        float val = (d < Dim) ? static_cast<float>(vec_ptr[d]) : 0.0f;
        float sq = val * val;
        float sg_sum = simd_sum(sq);

        threadgroup float sg_norms[8];
        if (d < 8) sg_norms[d] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_lid == 0 && sg_id < 8) sg_norms[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = (sg_id == 0 && sg_lid < 8) ? sg_norms[sg_lid] : 0.0f;
        total_sq = simd_sum(total_sq);
        if (sg_id == 0 && sg_lid == 0) sg_norms[0] = total_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float norm = sqrt(sg_norms[0]);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        if (d == 0) out_norms[bh] = half(norm);

        // Step 2: Unit vector -> shared for rotation
        threadgroup float shared[DimPadded];
        if (d < DimPadded) {{
            shared[d] = (d < Dim) ? (val * inv_norm * static_cast<float>(rotation[d < Dim ? d : 0])) : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: In-place Fast Walsh-Hadamard Transform (FWHT)
        float temp[1];
        for (int stride = 1; stride < DimPadded; stride *= 2) {{
            if (d < DimPadded) {{
                int pair = d ^ stride;
                uint upair = static_cast<uint>(pair);
                float a = shared[min(d, upair)];
                float b = shared[max(d, upair)];
                temp[0] = (d < upair) ? (a + b) : (a - b);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (d < DimPadded) shared[d] = temp[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Scale by 1/sqrt(DimPadded)
        float rotated = 0.0f;
        if (d < DimPadded) {{
            rotated = shared[d] * (1.0f / sqrt((float)DimPadded));
        }}

        // Step 4: Comparison-based quantize
        threadgroup uint shared_idx[DimPadded];
        uint idx = 0;
        if (d < Dim) {{
            for (int m = 0; m < {num_midpoints}; m++)
                idx += (rotated > midpoints[m]) ? 1u : 0u;
            shared_idx[d] = idx;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Pack indices
        if (d < PackedWidth) {{
            uint w_val = 0u;
            int word_start = (int)d * 32;
            int i_min = word_start / {bits};
            int i_max = (word_start + 31) / {bits};
            if (i_max >= (int)Dim) i_max = (int)Dim - 1;
            for (int i = i_min; i <= i_max; i++) {{
                uint idx_val = shared_idx[i] & {mask}u;
                int bit_off = i * {bits} - word_start;
                if (bit_off >= 0) {{
                    w_val |= idx_val << bit_off;
                }} else {{
                    w_val |= idx_val >> (-bit_off);
                }}
            }}
            out_packed[bh * PackedWidth + d] = w_val;
        }}
        """
        name = f"turboquant_fused_mse_quantize_rht_{bits}"
    else:
        source = f"""
        auto d = thread_position_in_threadgroup.x;  // 0..Dim-1, one dim per thread
        auto bh = threadgroup_position_in_grid.x;
        auto sg_id = simdgroup_index_in_threadgroup;
        auto sg_lid = thread_index_in_simdgroup;
        auto vec_ptr = vectors + bh * Dim;

        // Step 1: Load & compute norm
        float val = (d < Dim) ? static_cast<float>(vec_ptr[d]) : 0.0f;
        float sq = val * val;
        float sg_sum = simd_sum(sq);

        threadgroup float sg_norms[8];
        if (d < 8) sg_norms[d] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_lid == 0 && sg_id < 8) sg_norms[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float total_sq = (sg_id == 0 && sg_lid < 8) ? sg_norms[sg_lid] : 0.0f;
        total_sq = simd_sum(total_sq);
        if (sg_id == 0 && sg_lid == 0) sg_norms[0] = total_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float norm = sqrt(sg_norms[0]);
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        if (d == 0) out_norms[bh] = half(norm);

        // Step 2: Unit vector -> shared for rotation
        threadgroup float shared[Dim];
        if (d < Dim) shared[d] = val * inv_norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Rotate — 1 dim per thread, 256 FMAs (was 2048 at TG=32)
        float rotated = 0.0f;
        if (d < Dim) {{
            auto row = rotation + d * Dim;
            for (int j = 0; j < (int)Dim; j++)
                rotated += shared[j] * row[j];
        }}

        // Step 4: Comparison-based quantize
        threadgroup uint shared_idx[Dim];
        uint idx = 0;
        if (d < Dim) {{
            for (int m = 0; m < {num_midpoints}; m++)
                idx += (rotated > midpoints[m]) ? 1u : 0u;
            shared_idx[d] = idx;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Pack indices
        if (d < PackedWidth) {{
            uint w_val = 0u;
            int word_start = (int)d * 32;
            int i_min = word_start / {bits};
            int i_max = (word_start + 31) / {bits};
            if (i_max >= (int)Dim) i_max = (int)Dim - 1;
            for (int i = i_min; i <= i_max; i++) {{
                uint idx_val = shared_idx[i] & {mask}u;
                int bit_off = i * {bits} - word_start;
                if (bit_off >= 0) {{
                    w_val |= idx_val << bit_off;
                }} else {{
                    w_val |= idx_val >> (-bit_off);
                }}
            }}
            out_packed[bh * PackedWidth + d] = w_val;
        }}
        """
        name = f"turboquant_fused_mse_quantize_v2_{bits}"

    return mx.fast.metal_kernel(
        name=name,
        input_names=["vectors", "rotation", "midpoints"],
        output_names=["out_norms", "out_packed"],
        source=source,
    )

def _fused_prod_quantize_kernel(mse_bits: int, use_rht: bool = False):
    """Fused Prod quantize: norm + rotate + MSE quantize + residual + QJL in 1 dispatch."""
    if not _metal_available() or mse_bits <= 0:
        return None
    num_entries = 1 << mse_bits
    num_midpoints = num_entries - 1
    mse_mask = num_entries - 1

    lines = [
        "        auto lane = thread_position_in_threadgroup.x;",
        "        auto bh = thread_position_in_grid.z;",
        "        auto vec_ptr = vectors + bh * Dim;",
        "",
        "        // Step 1: Load & compute norm",
        "        float v[DimsPerLane];",
        "        float partial_sq = 0.0f;",
        "        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {",
        "            float x = (d < Dim) ? static_cast<float>(vec_ptr[d]) : 0.0f;",
        "            v[i] = x;",
        "            partial_sq += x * x;",
        "        }",
        "        float norm = sqrt(simd_sum(partial_sq));",
        "        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;",
        "        if (lane == 0) out_norms[bh] = half(norm);",
        "",
    ]

    if use_rht:
        lines += [
            "        threadgroup float shared[DimPadded];",
            "        for (int i = 0, d = lane; i < DimsPerLanePadded; d += 32, i++)",
            "            shared[d < Dim ? d : 0] = (d < Dim) ? v[i] * inv_norm : 0.0f;",
            "        if (lane < (DimPadded - Dim) && (Dim + lane) < DimPadded)",
            "            shared[Dim + lane] = 0.0f;",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "",
        ]
        lines += _metal_butterfly_wht_forward(
            "shared", "sign_vec", "v", "DimsPerLanePadded"
        )
        lines += [
            "",
            "        float rotated[DimsPerLane];",
            "        for (int i = 0, d = lane; i < DimsPerLane; d += 32, i++)",
            "            rotated[i] = (d < Dim) ? shared[d] * rht_scale : 0.0f;",
        ]
    else:
        lines += [
            "        threadgroup float shared[Dim];",
            "        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32)",
            "            if (d < Dim) shared[d] = v[i] * inv_norm;",
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "",
            "        float rotated[DimsPerLane];",
            "        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {",
            "            float sum = 0.0f;",
            "            if (d < Dim) {",
            "                auto row = mse_rotation + d * Dim;",
            "                for (int j = 0; j < (int)Dim; j++)",
            "                    sum += shared[j] * row[j];",
            "            }",
            "            rotated[i] = sum;",
            "        }",
        ]

    # Steps 4-8 are the same for both paths
    lines += [
        "",
        "        // Step 4: MSE quantize + compute rotated residual",
        "        float rot_residual[DimsPerLane];",
        "        threadgroup uint shared_mse_idx[Dim];",
        "        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {",
        "            uint idx = 0;",
        f"            if (d < Dim) {{",
        f"                for (int m = 0; m < {num_midpoints}; m++)",
        f"                    idx += (rotated[i] > midpoints[m]) ? 1u : 0u;",
        f"            }}",
        "            float estimate = codebook[idx];",
        "            rot_residual[i] = rotated[i] - estimate;",
        "            if (d < Dim) shared_mse_idx[d] = idx;",
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        // Step 5: Pack MSE indices",
        "        for (int w = lane; w < MsePackedWidth; w += 32) {",
        "            uint word = 0;",
        f"            for (int b = 0; b < 32; b += {mse_bits}) {{",
        f"                int dim = (w * 32 + b) / {mse_bits};",
        f"                if (dim < Dim) word |= (shared_mse_idx[dim] & {mse_mask}u) << b;",
        f"            }}",
        "            out_mse_packed[bh * MsePackedWidth + w] = word;",
        "        }",
        "",
        "        // Step 6: Residual norm",
        "        float res_sq = 0.0f;",
        "        for (int i = 0; i < DimsPerLane; i++)",
        "            res_sq += rot_residual[i] * rot_residual[i];",
        "        float res_norm = sqrt(simd_sum(res_sq));",
        "        if (lane == 0) out_res_norms[bh] = half(res_norm);",
        "",
        "        // Step 7: QJL projection (rotated_residual @ combined_proj_t)",
        "        //   combined_proj_t = (mse_rotation @ projection_t).T",
        "        //   so row d gives the projection for output dim d",
        "        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32)",
        "            if (d < Dim) shared[d] = rot_residual[i];",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        threadgroup uint shared_signs[Dim];",
        "        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {",
        "            float proj = 0.0f;",
        "            if (d < Dim) {",
        "                auto row = combined_proj_t + d * Dim;",
        "                for (int j = 0; j < (int)Dim; j++)",
        "                    proj += shared[j] * row[j];",
        "            }",
        "            shared_signs[d < Dim ? d : 0] = (proj >= 0.0f) ? 1u : 0u;",
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        // Step 8: Pack sign bits (1 bit per dim)",
        "        for (int w = lane; w < SignPackedWidth; w += 32) {",
        "            uint word = 0;",
        "            for (int b = 0; b < 32; b++) {",
        "                int dim = w * 32 + b;",
        "                if (dim < Dim) word |= shared_signs[dim] << b;",
        "            }",
        "            out_signs[bh * SignPackedWidth + w] = word;",
        "        }",
    ]

    name_suffix = "_rht" if use_rht else ""
    rotation_input = "sign_vec" if use_rht else "mse_rotation"
    return mx.fast.metal_kernel(
        name=f"turboquant_fused_prod_quantize_{mse_bits}{name_suffix}",
        input_names=[
            "vectors",
            rotation_input,
            "midpoints",
            "codebook",
            "combined_proj_t",
        ],
        output_names=[
            "out_norms",
            "out_mse_packed",
            "out_res_norms",
            "out_signs",
        ],
        source="\n".join(lines),
    )

