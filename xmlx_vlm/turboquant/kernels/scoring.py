from __future__ import annotations
import math
from functools import lru_cache
import mlx.core as mx
from .base import _metal_available

def _mse_score_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = packed + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = packed_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= packed_ptr[word_idx + 1] << (Bits - spill);
            }
            value &= ((1u << Bits) - 1u);
            acc += static_cast<float>(q_ptr[d]) * codebook[value];
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc * static_cast<float>(norms[(b * kv_heads + h) * token_count + t]);
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_mse_score",
        input_names=["q_rot", "norms", "packed", "codebook"],
        output_names=["out"],
        source=source,
    )

def _mse_score_tiled_kernel(repeat_count: int):
    """Tiled MSE score: preload R queries into registers, loop over tokens.
    Processes TokTileSize tokens per threadgroup instead of 1.
    Reduces threadgroup count by TokTileSize×, amortizing dispatch overhead."""
    if not _metal_available() or repeat_count < 1:
        return None

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto q_base = q_rot + (bh * RepeatCount) * Dim;",
        "        auto pk = packed + bh * token_count * PackedWidth;",
        "        auto nm = norms + bh * token_count;",
        "",
        "        // Preload query values into registers (constant across all tokens)",
    ]

    for r in range(repeat_count):
        lines += [
            f"        float qr_{r}[DimsPerLane];",
            f"        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32)",
            f"            qr_{r}[i] = (d < Dim) ? static_cast<float>(q_base[{r} * Dim + d]) : 0.0f;",
        ]

    lines += [
        "",
        "        // Precompute bit offsets for key unpacking",
        "        int k_words[DimsPerLane], k_offs[DimsPerLane];",
        "        bool k_spills[DimsPerLane];",
        "        for (int i = 0, d = lane; i < DimsPerLane; i++, d += 32) {",
        "            int bo = d * Bits;",
        "            k_words[i] = bo / 32;",
        "            k_offs[i] = bo % 32;",
        "            k_spills[i] = (bo % 32 + Bits) > 32;",
        "        }",
        "        constexpr uint mask = (1u << Bits) - 1u;",
        "",
        "        // Token loop: key data changes, queries stay in registers",
        "        for (int t = t_start; t < t_end; t++) {",
        "            auto pt = pk + t * PackedWidth;",
        "            float kn = static_cast<float>(nm[t]);",
        "",
    ]

    for r in range(repeat_count):
        lines += [f"            float ps_{r} = 0.0f;"]

    lines += [
        "            for (int i = 0; i < DimsPerLane; i++) {",
        "                uint vv = (pt[k_words[i]] >> k_offs[i]);",
        "                if (k_spills[i]) vv |= pt[k_words[i]+1] << (Bits - (k_offs[i]+Bits-32));",
        "                float code = codebook[vv & mask];",
    ]

    for r in range(repeat_count):
        lines += [f"                ps_{r} += qr_{r}[i] * code;"]

    lines += [
        "            }",
        "",
    ]

    for r in range(repeat_count):
        lines += [f"            float s_{r} = simd_sum(ps_{r});"]

    lines += [
        "            if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines += [
            f"                out[(bh * RepeatCount + {r}) * token_count + t] = kn * s_{r};",
        ]
    lines += [
        "            }",
        "        }",
    ]

    return mx.fast.metal_kernel(
        name=f"turboquant_mse_score_tiled_{repeat_count}",
        input_names=["q_rot", "norms", "packed", "codebook"],
        output_names=["out"],
        source="\n".join(lines),
    )

def _qjl_score_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_proj_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto packed_ptr = signs + ((b * kv_heads + h) * token_count + t) * PackedWidth;

        float acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int word_idx = d / 32;
            int offset = d % 32;
            uint bit = (packed_ptr[word_idx] >> offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            acc += static_cast<float>(q_ptr[d]) * sign;
        }

        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                acc
                * static_cast<float>(norms[idx])
                * static_cast<float>(residual_norms[idx])
                * scale[0];
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_qjl_score",
        input_names=["q_proj", "norms", "residual_norms", "signs", "scale"],
        output_names=["out"],
        source=source,
    )

def _prod_score_kernel():
    if not _metal_available():
        return None

    source = r"""
        auto lane = thread_position_in_grid.x;
        auto repeat_idx = thread_position_in_grid.y;
        auto n = thread_position_in_grid.z;

        auto token_count = norms_shape[2];
        auto kv_heads = norms_shape[1];
        auto repeat_count = q_rot_shape[2];
        if (repeat_idx >= repeat_count) {
            return;
        }

        auto b = n / (kv_heads * token_count);
        auto rem = n % (kv_heads * token_count);
        auto h = rem / token_count;
        auto t = rem % token_count;

        auto q_rot_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto q_proj_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

        float mse_acc = 0.0f;
        float qjl_acc = 0.0f;
        for (int d = lane; d < Dim; d += 32) {
            int bit_offset = d * MseBits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = mse_ptr[word_idx] >> offset;
            int spill = offset + MseBits - 32;
            if (spill > 0) {
                value |= mse_ptr[word_idx + 1] << (MseBits - spill);
            }
            value &= ((1u << MseBits) - 1u);
            mse_acc += static_cast<float>(q_rot_ptr[d]) * codebook[value];

            int sign_word = d / 32;
            int sign_offset = d % 32;
            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            qjl_acc += static_cast<float>(q_proj_ptr[d]) * sign;
        }

        mse_acc = simd_sum(mse_acc);
        qjl_acc = simd_sum(qjl_acc);
        if (thread_index_in_simdgroup == 0) {
            auto idx = (b * kv_heads + h) * token_count + t;
            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                static_cast<float>(norms[idx]) * (
                    mse_acc
                    + scale[0] * static_cast<float>(residual_norms[idx]) * qjl_acc
                );
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_prod_score",
        input_names=[
            "q_rot",
            "q_proj",
            "norms",
            "residual_norms",
            "mse_packed",
            "signs",
            "codebook",
            "scale",
        ],
        output_names=["out"],
        source=source,
    )

def _prod_score_repeat_kernel(repeat_count: int):
    if not _metal_available() or repeat_count <= 1:
        return None

    # Group-by-codebook-entry scoring: accumulate q_rot by index, then
    # dot with codebook after the dim loop. Removes codebook multiply from
    # the inner loop (D additions instead of D FMAs, then 2^b FMAs).
    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = q_rot_shape[2];",
        "        constexpr uint mse_mask = (1u << MseBits) - 1u;",
        "        constexpr int num_entries = 1 << MseBits;",
        "",
        "        auto b = n / (kv_heads * token_count);",
        "        auto rem = n % (kv_heads * token_count);",
        "        auto h = rem / token_count;",
        "        auto t = rem % token_count;",
        "",
        "        auto q_rot_base = q_rot + ((b * kv_heads + h) * repeat_count) * Dim;",
        "        auto q_proj_base = q_proj + ((b * kv_heads + h) * repeat_count) * Dim;",
        "        auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;",
        "        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;",
        "",
        "        auto idx = (b * kv_heads + h) * token_count + t;",
        "        float norm = static_cast<float>(norms[idx]);",
        "        float residual_norm = static_cast<float>(residual_norms[idx]);",
        "",
    ]
    # Per-repeat group accumulators and QJL accumulators
    for r in range(repeat_count):
        lines.append(f"        float grp_{r}[num_entries] = {{}};")
        lines.append(f"        float qjl_acc_{r} = 0.0f;")
    lines += [
        "",
        "        for (int d = lane; d < Dim; d += 32) {",
        "            int bit_offset = d * MseBits;",
        "            int word_idx = bit_offset / 32;",
        "            int offset = bit_offset % 32;",
        "            uint value = mse_ptr[word_idx] >> offset;",
        "            int spill = offset + MseBits - 32;",
        "            if (spill > 0) {",
        "                value |= mse_ptr[word_idx + 1] << (MseBits - spill);",
        "            }",
        "            value &= mse_mask;",
        "",
        "            int sign_word = d / 32;",
        "            int sign_offset = d % 32;",
        "            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;",
        "            float sign = bit ? 1.0f : -1.0f;",
        "",
    ]
    # Group q_rot by codebook index (no multiply in inner loop)
    for r in range(repeat_count):
        lines.append(
            f"            grp_{r}[value] += static_cast<float>(q_rot_base[{r} * Dim + d]);"
        )
        lines.append(
            f"            qjl_acc_{r} += static_cast<float>(q_proj_base[{r} * Dim + d]) * sign;"
        )
    lines += [
        "        }",
        "",
        "        // Dot product: group sums × codebook entries",
    ]
    for r in range(repeat_count):
        lines.append(f"        float mse_acc_{r} = 0.0f;")
        lines.append(f"        for (int k = 0; k < num_entries; k++)")
        lines.append(f"            mse_acc_{r} += grp_{r}[k] * codebook[k];")

    lines.append("")
    for r in range(repeat_count):
        lines.append(f"        float mse_sum_{r} = simd_sum(mse_acc_{r});")
        lines.append(f"        float qjl_sum_{r} = simd_sum(qjl_acc_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * repeat_count + {r}) * token_count + t] ="
        )
        lines.append(
            f"                norm * (mse_sum_{r} + scale[0] * residual_norm * qjl_sum_{r});"
        )
    lines += [
        "        }",
    ]

    source = "\n".join(lines)
    return mx.fast.metal_kernel(
        name=f"turboquant_prod_score_repeat_{repeat_count}",
        input_names=[
            "q_rot",
            "q_proj",
            "norms",
            "residual_norms",
            "mse_packed",
            "signs",
            "codebook",
            "scale",
        ],
        output_names=["out"],
        source=source,
    )

def _polar_prod_score_kernel(level_bits: tuple[int, ...]):
    if not _metal_available() or len(level_bits) == 0:
        return None

    input_names = ["q_rot", "norms", "radii"]
    for level in range(len(level_bits)):
        input_names.append(f"angles_{level + 1}")
    for level in range(len(level_bits)):
        input_names.append(f"cos_{level + 1}")
        input_names.append(f"sin_{level + 1}")

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto repeat_idx = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto repeat_count = q_rot_shape[2];",
        "        if (repeat_idx >= repeat_count) {",
        "            return;",
        "        }",
        "",
        "        auto b = n / (kv_heads * token_count);",
        "        auto rem = n % (kv_heads * token_count);",
        "        auto h = rem / token_count;",
        "        auto t = rem % token_count;",
        "",
        "        auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;",
        "        auto radii_ptr = radii + ((b * kv_heads + h) * token_count + t) * BlockCount;",
        "",
        "        float acc = 0.0f;",
        "        for (int d = lane; d < Dim; d += 32) {",
        "            int block_idx = d >> Levels;",
        "            float coeff = static_cast<float>(radii_ptr[block_idx]);",
        "",
    ]

    for level, bits in enumerate(level_bits, start=1):
        mask = (1 << bits) - 1
        lines += [
            f"            auto angle_ptr_{level} = angles_{level} + ((b * kv_heads + h) * token_count + t) * PackedWidth{level};",
            f"            int angle_idx_{level} = d >> {level};",
            f"            int bit_offset_{level} = angle_idx_{level} * {bits};",
            f"            int word_idx_{level} = bit_offset_{level} / 32;",
            f"            int offset_{level} = bit_offset_{level} % 32;",
            f"            uint value_{level} = angle_ptr_{level}[word_idx_{level}] >> offset_{level};",
            f"            int spill_{level} = offset_{level} + {bits} - 32;",
            f"            if (spill_{level} > 0) {{",
            f"                value_{level} |= angle_ptr_{level}[word_idx_{level} + 1] << ({bits} - spill_{level});",
            "            }",
            f"            value_{level} &= {mask}u;",
            f"            bool use_sin_{level} = ((d >> {level - 1}) & 1) != 0;",
            f"            coeff *= use_sin_{level} ? static_cast<float>(sin_{level}[value_{level}]) : static_cast<float>(cos_{level}[value_{level}]);",
            "",
        ]

    lines += [
        "            acc += static_cast<float>(q_ptr[d]) * coeff;",
        "        }",
        "",
        "        acc = simd_sum(acc);",
        "        if (thread_index_in_simdgroup == 0) {",
        "            auto idx = (b * kv_heads + h) * token_count + t;",
        "            out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =",
        "                acc * static_cast<float>(norms[idx]);",
        "        }",
    ]

    return mx.fast.metal_kernel(
        name="turboquant_polar_prod_score_" + "_".join(str(bit) for bit in level_bits),
        input_names=input_names,
        output_names=["out"],
        source="\n".join(lines),
    )

def _polar_turbo_score_repeat_kernel(level_bits: tuple[int, ...], repeat_count: int):
    if not _metal_available() or len(level_bits) != 4 or repeat_count <= 0:
        return None

    bits1, bits2, bits3, bits4 = level_bits
    mask1 = (1 << bits1) - 1
    mask2 = (1 << bits2) - 1
    mask3 = (1 << bits3) - 1
    mask4 = (1 << bits4) - 1

    input_names = ["q_rot", "q_proj", "norms", "radii"]
    for level in range(4):
        input_names.append(f"angles_{level + 1}")
    input_names.extend(["residual_norms", "signs", "scale"])
    for level in range(4):
        input_names.append(f"cos_{level + 1}")
        input_names.append(f"sin_{level + 1}")

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "",
        "        auto b = n / (kv_heads * token_count);",
        "        auto rem = n % (kv_heads * token_count);",
        "        auto h = rem / token_count;",
        "        auto t = rem % token_count;",
        "",
        "        auto q_rot_base = q_rot + ((b * kv_heads + h) * RepeatCount) * Dim;",
        "        auto q_proj_base = q_proj + ((b * kv_heads + h) * RepeatCount) * Dim;",
        "        auto radii_ptr = radii + ((b * kv_heads + h) * token_count + t) * BlockCount;",
        "        auto angle1_ptr = angles_1 + ((b * kv_heads + h) * token_count + t) * PackedWidth1;",
        "        auto angle2_ptr = angles_2 + ((b * kv_heads + h) * token_count + t) * PackedWidth2;",
        "        auto angle3_ptr = angles_3 + ((b * kv_heads + h) * token_count + t) * PackedWidth3;",
        "        auto angle4_ptr = angles_4 + ((b * kv_heads + h) * token_count + t) * PackedWidth4;",
        "        auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;",
        "",
        "        auto idx = (b * kv_heads + h) * token_count + t;",
        "        float norm = static_cast<float>(norms[idx]);",
        "        float residual_norm = static_cast<float>(residual_norms[idx]);",
        "",
    ]
    for r in range(repeat_count):
        lines.append(f"        threadgroup float level1_{r}[Dim / 2];")
        lines.append(f"        threadgroup float level2_{r}[Dim / 4];")
        lines.append(f"        threadgroup float level3_{r}[Dim / 8];")
        lines.append(f"        threadgroup float level4_{r}[BlockCount];")
    lines += [""]
    for r in range(repeat_count):
        lines.append(f"        float qjl_acc_{r} = 0.0f;")
    lines += [
        "",
        "        for (int pair_idx = lane; pair_idx < Dim / 2; pair_idx += 32) {",
        f"            int bit_offset_1 = pair_idx * {bits1};",
        "            int word_idx_1 = bit_offset_1 / 32;",
        "            int offset_1 = bit_offset_1 % 32;",
        "            uint value_1 = angle1_ptr[word_idx_1] >> offset_1;",
        f"            int spill_1 = offset_1 + {bits1} - 32;",
        "            if (spill_1 > 0) {",
        f"                value_1 |= angle1_ptr[word_idx_1 + 1] << ({bits1} - spill_1);",
        "            }",
        f"            value_1 &= {mask1}u;",
        "            float cos_1_val = static_cast<float>(cos_1[value_1]);",
        "            float sin_1_val = static_cast<float>(sin_1[value_1]);",
        "            int d0 = pair_idx << 1;",
        "            int d1 = d0 + 1;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            level1_{r}[pair_idx] = static_cast<float>(q_rot_base[{r} * Dim + d0]) * cos_1_val + static_cast<float>(q_rot_base[{r} * Dim + d1]) * sin_1_val;"
        )
    lines += [
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        for (int pair_idx = lane; pair_idx < Dim / 4; pair_idx += 32) {",
        f"            int bit_offset_2 = pair_idx * {bits2};",
        "            int word_idx_2 = bit_offset_2 / 32;",
        "            int offset_2 = bit_offset_2 % 32;",
        "            uint value_2 = angle2_ptr[word_idx_2] >> offset_2;",
        f"            int spill_2 = offset_2 + {bits2} - 32;",
        "            if (spill_2 > 0) {",
        f"                value_2 |= angle2_ptr[word_idx_2 + 1] << ({bits2} - spill_2);",
        "            }",
        f"            value_2 &= {mask2}u;",
        "            float cos_2_val = static_cast<float>(cos_2[value_2]);",
        "            float sin_2_val = static_cast<float>(sin_2[value_2]);",
        "            int child = pair_idx << 1;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            level2_{r}[pair_idx] = level1_{r}[child] * cos_2_val + level1_{r}[child + 1] * sin_2_val;"
        )
    lines += [
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        for (int pair_idx = lane; pair_idx < Dim / 8; pair_idx += 32) {",
        f"            int bit_offset_3 = pair_idx * {bits3};",
        "            int word_idx_3 = bit_offset_3 / 32;",
        "            int offset_3 = bit_offset_3 % 32;",
        "            uint value_3 = angle3_ptr[word_idx_3] >> offset_3;",
        f"            int spill_3 = offset_3 + {bits3} - 32;",
        "            if (spill_3 > 0) {",
        f"                value_3 |= angle3_ptr[word_idx_3 + 1] << ({bits3} - spill_3);",
        "            }",
        f"            value_3 &= {mask3}u;",
        "            float cos_3_val = static_cast<float>(cos_3[value_3]);",
        "            float sin_3_val = static_cast<float>(sin_3[value_3]);",
        "            int child = pair_idx << 1;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            level3_{r}[pair_idx] = level2_{r}[child] * cos_3_val + level2_{r}[child + 1] * sin_3_val;"
        )
    lines += [
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        for (int pair_idx = lane; pair_idx < BlockCount; pair_idx += 32) {",
        f"            int bit_offset_4 = pair_idx * {bits4};",
        "            int word_idx_4 = bit_offset_4 / 32;",
        "            int offset_4 = bit_offset_4 % 32;",
        "            uint value_4 = angle4_ptr[word_idx_4] >> offset_4;",
        f"            int spill_4 = offset_4 + {bits4} - 32;",
        "            if (spill_4 > 0) {",
        f"                value_4 |= angle4_ptr[word_idx_4 + 1] << ({bits4} - spill_4);",
        "            }",
        f"            value_4 &= {mask4}u;",
        "            float cos_4_val = static_cast<float>(cos_4[value_4]);",
        "            float sin_4_val = static_cast<float>(sin_4[value_4]);",
        "            int child = pair_idx << 1;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            level4_{r}[pair_idx] = level3_{r}[child] * cos_4_val + level3_{r}[child + 1] * sin_4_val;"
        )
    lines += [
        "        }",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "        for (int d = lane; d < Dim; d += 32) {",
        "            int sign_word = d / 32;",
        "            int sign_offset = d % 32;",
        "            uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;",
        "            float sign = bit ? 1.0f : -1.0f;",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            qjl_acc_{r} += static_cast<float>(q_proj_base[{r} * Dim + d]) * sign;"
        )
    lines += [
        "        }",
        "",
    ]
    for r in range(repeat_count):
        lines.append(
            f"        float polar_acc_{r} = lane < BlockCount ? static_cast<float>(radii_ptr[lane]) * level4_{r}[lane] : 0.0f;"
        )
        lines.append(f"        float polar_sum_{r} = simd_sum(polar_acc_{r});")
        lines.append(f"        float qjl_sum_{r} = simd_sum(qjl_acc_{r});")
    lines += [
        "",
        "        if (thread_index_in_simdgroup == 0) {",
    ]
    for r in range(repeat_count):
        lines.append(
            f"            out[((b * kv_heads + h) * RepeatCount + {r}) * token_count + t] ="
        )
        lines.append(
            f"                norm * (polar_sum_{r} + scale[0] * residual_norm * qjl_sum_{r});"
        )
    lines += [
        "        }",
    ]

    return mx.fast.metal_kernel(
        name="turboquant_polar_turbo_score_"
        + "_".join(str(bit) for bit in level_bits)
        + f"_repeat_{repeat_count}",
        input_names=input_names,
        output_names=["out"],
        source="\n".join(lines),
    )

def _multi_query_prod_score_kernel(
    key_mse_bits: int, repeat_count: int, num_queries: int, dims_per_lane: int
):
    """Multi-query score kernel: unpack key data ONCE per token, loop over L queries.
    Avoids R*L repeat explosion that causes register spill."""
    if not _metal_available() or repeat_count < 1 or num_queries < 1:
        return None

    mse_mask = (1 << key_mse_bits) - 1

    return mx.fast.metal_kernel(
        name=f"mq_prod_score_{key_mse_bits}_r{repeat_count}_l{num_queries}",
        input_names=[
            "q_rot",
            "q_proj",
            "key_norms",
            "key_mse",
            "key_res_norms",
            "key_signs",
            "key_codebook",
            "key_scale",
        ],
        output_names=["out"],
        source=f"""
            auto lane = thread_position_in_grid.x;
            auto ri = thread_position_in_grid.y;
            auto n = thread_position_in_grid.z;
            auto tc = key_norms_shape[2];
            auto kh = key_norms_shape[1];
            auto b = n / (kh * tc);
            auto rem = n % (kh * tc);
            auto h = rem / tc;
            auto t = rem % tc;
            if (ri >= {repeat_count}) return;

            auto mt = key_mse + ((b*kh+h)*tc+t) * KMsePackedWidth;
            auto st = key_signs + ((b*kh+h)*tc+t) * KSignPackedWidth;
            float kn = static_cast<float>(key_norms[(b*kh+h)*tc+t]);
            float ksr = kn * key_scale[0] * static_cast<float>(key_res_norms[(b*kh+h)*tc+t]);

            // Unpack key ONCE
            float kc[{dims_per_lane}], ksf[{dims_per_lane}];
            for (int i=0, d=lane; d < Dim; i++, d+=32) {{
                int bo = d * {key_mse_bits};
                uint idx = (mt[bo>>5] >> (bo&31));
                if (((bo&31)+{key_mse_bits}) > 32) idx |= mt[(bo>>5)+1] << ({key_mse_bits} - ((bo&31)+{key_mse_bits}-32));
                idx &= {mse_mask}u;
                kc[i] = key_codebook[idx];
                uint sb = (st[d>>5] >> (d&31)) & 1u;
                ksf[i] = sb ? 1.0f : -1.0f;
            }}

            // Loop over L queries, reusing unpacked key
            auto bq = (b*kh+h) * {repeat_count} + ri;
            for (int l = 0; l < {num_queries}; l++) {{
                float ps = 0.0f;
                for (int i=0, d=lane; d < Dim; i++, d+=32) {{
                    ps += kn * static_cast<float>(q_rot[(bq*{num_queries}+l)*Dim+d]) * kc[i]
                        + ksr * static_cast<float>(q_proj[(bq*{num_queries}+l)*Dim+d]) * ksf[i];
                }}
                float s = simd_sum(ps);
                if (lane == 0)
                    out[((b*kh+h)*{repeat_count}+ri)*{num_queries}*tc + l*tc + t] = s;
            }}
        """,
    )

