from __future__ import annotations
import math
from functools import lru_cache
import mlx.core as mx
from .base import _metal_available
from .rotation import _metal_butterfly_wht_inverse

def _compiled_integer_decode_kernel(bits: int):
    mse_bits = max(bits - 1, 0)

    @mx.compile
    def _decode(
        grouped_queries: mx.array,
        key_norms: mx.array,
        key_mse_indices: mx.array,
        key_residual_norms: mx.array,
        key_qjl_signs: mx.array,
        value_norms: mx.array,
        value_indices: mx.array,
        key_query_transform_t: mx.array,
        key_codebook: mx.array,
        key_scale: mx.array,
        value_codebook: mx.array,
        value_rotation: mx.array,
    ) -> mx.array:
        query_transformed = mx.matmul(grouped_queries, key_query_transform_t)
        dim = grouped_queries.shape[-1]
        q_rot = query_transformed[..., :dim]
        q_proj = query_transformed[..., dim:]
        scores = _metal_prod_score(
            q_rot.reshape(
                q_rot.shape[0], q_rot.shape[1], q_rot.shape[2], q_rot.shape[-1]
            ),
            q_proj.reshape(
                q_proj.shape[0], q_proj.shape[1], q_proj.shape[2], q_proj.shape[-1]
            ),
            TurboQuantProdState(
                key_norms,
                key_mse_indices,
                key_residual_norms,
                key_qjl_signs,
            ),
            mse_bits,
            key_codebook,
            key_scale,
        )
        return _metal_mse_weighted_sum_from_scores(
            scores,
            TurboQuantMSEState(value_norms, value_indices),
            bits,
            value_codebook,
            value_rotation,
        )

    return _decode

def _fused_integer_decode_kernel(bits: int, repeat_count: int, key_mse_bits: int = -1):
    """Fused integer decode: score + online-softmax + weighted-sum."""
    if not _metal_available() or repeat_count < 1:
        return None

    mse_bits = key_mse_bits if key_mse_bits >= 0 else max(bits - 1, 0)
    mse_mask = (1 << mse_bits) - 1
    val_mask = (1 << bits) - 1

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto val_tile = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        "        int val_dim = val_tile * 32 + lane;",
        "",
        "        auto token_count = key_norms_shape[2];",
        "        auto kv_heads = key_norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        auto b = bh / kv_heads;",
        "        auto h = bh % kv_heads;",
        "        auto base = (b * kv_heads + h);",
        "",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto k_norms = key_norms + base * token_count;",
        "        auto k_mse = key_mse + base * token_count * KMsePackedWidth;",
        "        auto k_res = key_res_norms + base * token_count;",
        "        auto k_signs = key_signs + base * token_count * KSignPackedWidth;",
        "        auto v_norms = val_norms + base * token_count;",
        "        auto v_packed = val_packed + base * token_count * VPackedWidth;",
        "",
        "        bool v_valid = val_dim < Dim;",
        "        int v_bo = val_dim * ValBits;",
        "        int v_word = v_bo / 32;",
        "        int v_off = v_bo % 32;",
        f"        bool v_spills = (v_off + ValBits > 32);",
        "",
    ]

    for r in range(repeat_count):
        lines += [
            f"        auto qr_{r} = q_rot + (base * RepeatCount + {r}) * Dim;",
            f"        auto qp_{r} = q_proj + (base * RepeatCount + {r}) * Dim;",
        ]

    for r in range(repeat_count):
        lines += [
            f"        float lmax_{r} = -INFINITY;",
            f"        float lsum_{r} = 0.0f;",
            f"        float lacc_{r} = 0.0f;",
        ]

    lines += [
        "",
        "        for (int t = t_start; t < t_end; t++) {",
        "            auto mse_t = k_mse + t * KMsePackedWidth;",
        "            auto sign_t = k_signs + t * KSignPackedWidth;",
        "            float kn = static_cast<float>(k_norms[t]);",
        "            float ksr = kn * key_scale[0] * static_cast<float>(k_res[t]);",
    ]

    for r in range(repeat_count):
        lines += [f"            float ps_{r} = 0.0f;"]

    lines += [
        f"            for (int d = lane; d < Dim; d += 32) {{",
        f"                int bo = d * {mse_bits};",
        f"                uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                if (((bo & 31) + {mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({mse_bits} - ((bo & 31) + {mse_bits} - 32));",
        f"                idx &= {mse_mask}u;",
        f"                float code = key_codebook[idx];",
        f"                uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [
            f"                ps_{r} += kn * static_cast<float>(qr_{r}[d]) * code + ksr * (sb ? static_cast<float>(qp_{r}[d]) : -static_cast<float>(qp_{r}[d]));"
        ]
    lines += [
        "            }",
    ]
    for r in range(repeat_count):
        lines += [f"            float s_{r} = simd_sum(ps_{r});"]

    # Value decode + online softmax
    lines += [
        "",
        "            float v_code = 0.0f;",
        "            if (v_valid) {",
        "                auto vt = v_packed + t * VPackedWidth;",
        "                uint vv = (vt[v_word] >> v_off);",
        f"                if (v_spills) vv |= vt[v_word + 1] << (ValBits - (v_off + ValBits - 32));",
        f"                v_code = val_codebook[vv & {val_mask}u] * static_cast<float>(v_norms[t]);",
        "            }",
    ]

    for r in range(repeat_count):
        lines += [
            f"            float om_{r} = lmax_{r};",
            f"            lmax_{r} = max(lmax_{r}, s_{r});",
            f"            float rs_{r} = exp(om_{r} - lmax_{r});",
            f"            float w_{r} = exp(s_{r} - lmax_{r});",
            f"            lsum_{r} = lsum_{r} * rs_{r} + w_{r};",
            f"            lacc_{r} = lacc_{r} * rs_{r} + w_{r} * v_code;",
        ]

    lines += ["        }", ""]

    lines += ["        int out_stride = Dim;"]
    for r in range(repeat_count):
        lines += [
            f"        if (v_valid) {{",
            f"            out_acc[((bh * num_tok_tiles + tok_tile) * RepeatCount + {r}) * out_stride + val_dim] = lacc_{r};",
            f"        }}",
            f"        if (val_dim == 0) {{",
            f"            int sm_base = (bh * num_tok_tiles + tok_tile) * RepeatCount + {r};",
            f"            out_sum[sm_base] = lsum_{r};",
            f"            out_max[sm_base] = lmax_{r};",
            f"        }}",
        ]

    return mx.fast.metal_kernel(
        name=f"turboquant_fused_integer_decode_{bits}_r{repeat_count}",
        input_names=[
            "q_rot",
            "q_proj",
            "key_norms",
            "key_mse",
            "key_res_norms",
            "key_signs",
            "val_norms",
            "val_packed",
            "key_codebook",
            "key_scale",
            "val_codebook",
        ],
        output_names=["out_acc", "out_sum", "out_max"],
        source="\n".join(lines),
    )

def _single_tile_value_weighted_sum_kernel(
    bits: int, repeat_count: int, dims_per_lane: int
):
    """Single-tile value weighted sum with precomputed softmax weights.
    TG=Dim: one thread per value dim, no exp() calls in inner loop.
    2x faster than online-softmax variant."""
    if not _metal_available() or repeat_count < 1:
        return None

    val_mask = (1 << bits) - 1

    lines = [
        "        auto dim = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "        auto token_count = norms_shape[2];",
        "        auto kv_heads = norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto wt = weights + bh * RepeatCount * token_count;",
        "        auto nm = norms + bh * token_count;",
        "        auto pk = packed + bh * token_count * PackedWidth;",
        "",
        f"        int bo = dim * {bits};",
        f"        int v_word = bo / 32;",
        f"        int v_shift = bo % 32;",
        f"        bool v_spill = (bo % 32 + {bits}) > 32;",
        "",
    ]

    for r in range(repeat_count):
        lines += [f"        float acc_{r} = 0.0f;"]

    lines += [
        "",
        "        for (int t = t_start; t < t_end; t++) {",
        "            auto pt = pk + t * PackedWidth;",
        "            uint vv = (pt[v_word] >> v_shift);",
        f"            if (v_spill) vv |= pt[v_word+1] << ({bits} - (v_shift+{bits}-32));",
        f"            float val = codebook[vv & {val_mask}u] * static_cast<float>(nm[t]);",
        "",
    ]

    for r in range(repeat_count):
        lines += [f"            acc_{r} += wt[{r}*token_count+t] * val;"]

    lines += ["        }", ""]

    for r in range(repeat_count):
        lines += [
            f"        if (dim < Dim) out[((bh*num_tok_tiles+tok_tile)*RepeatCount+{r})*Dim+dim] = acc_{r};",
        ]

    return mx.fast.metal_kernel(
        name=f"turboquant_single_tile_value_{bits}_r{repeat_count}",
        input_names=["weights", "norms", "packed", "codebook"],
        output_names=["out"],
        source="\n".join(lines),
    )

def _fused_integer_decode_single_tile_kernel(
    bits: int, repeat_count: int, dims_per_lane: int, key_mse_bits: int = -1
):
    """Single-tile fused kernel — each lane handles multiple value dims.
    Zero key read redundancy: keys are read once per token, not once per val_tile.
    Faster than multi-tile at long contexts (256k+) where memory bandwidth dominates.
    """
    if not _metal_available() or repeat_count < 1:
        return None

    mse_bits = key_mse_bits if key_mse_bits >= 0 else max(bits - 1, 0)
    mse_mask = (1 << mse_bits) - 1
    val_mask = (1 << bits) - 1

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto n = thread_position_in_grid.z;",
        "        auto token_count = key_norms_shape[2];",
        "        auto kv_heads = key_norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        auto b = bh / kv_heads;",
        "        auto h = bh % kv_heads;",
        "        auto base = (b * kv_heads + h);",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto k_norms = key_norms + base * token_count;",
        "        auto k_mse = key_mse + base * token_count * KMsePackedWidth;",
        "        auto k_res = key_res_norms + base * token_count;",
        "        auto k_signs = key_signs + base * token_count * KSignPackedWidth;",
        "        auto v_norms = val_norms + base * token_count;",
        "        auto v_packed = val_packed + base * token_count * VPackedWidth;",
        "",
        "        // Precompute value bit offsets for all dims this lane handles",
        "        int v_words[DimsPerLane], v_offs[DimsPerLane];",
        "        bool v_spills[DimsPerLane], v_valids[DimsPerLane];",
        "        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32) {",
        f"            v_valids[i] = vd < Dim;",
        f"            int vbo = vd * {bits};",
        f"            v_words[i] = vbo / 32;",
        f"            v_offs[i] = vbo % 32;",
        f"            v_spills[i] = (vbo % 32 + {bits}) > 32;",
        "        }",
        "",
    ]

    for r in range(repeat_count):
        lines += [
            f"        auto qr_{r} = q_rot + (base * RepeatCount + {r}) * Dim;",
            f"        auto qp_{r} = q_proj + (base * RepeatCount + {r}) * Dim;",
        ]

    for r in range(repeat_count):
        lines += [
            f"        float lmax_{r} = -INFINITY, lsum_{r} = 0.0f;",
            f"        float lacc_{r}[DimsPerLane] = {{}};",
        ]

    lines += [
        "",
        "        for (int t = t_start; t < t_end; t++) {",
        "            auto mse_t = k_mse + t * KMsePackedWidth;",
        "            auto sign_t = k_signs + t * KSignPackedWidth;",
        "            float kn = static_cast<float>(k_norms[t]);",
        "            float ksr = kn * key_scale[0] * static_cast<float>(k_res[t]);",
        "",
    ]

    # Score
    for r in range(repeat_count):
        lines += [f"            float ps_{r} = 0.0f;"]

    lines += [
        f"            for (int d = lane; d < Dim; d += 32) {{",
        f"                int bo = d * {mse_bits};",
        f"                uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                if (((bo & 31) + {mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({mse_bits} - ((bo & 31) + {mse_bits} - 32));",
        f"                idx &= {mse_mask}u;",
        f"                float code = key_codebook[idx];",
        f"                uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [
            f"                ps_{r} += kn * static_cast<float>(qr_{r}[d]) * code + ksr * (sb ? static_cast<float>(qp_{r}[d]) : -static_cast<float>(qp_{r}[d]));"
        ]
    lines += ["            }"]

    for r in range(repeat_count):
        lines += [f"            float s_{r} = simd_sum(ps_{r});"]

    # Online softmax + multi-dim value accumulation
    lines += [
        "",
        "            auto vt = v_packed + t * VPackedWidth;",
        "            float vnorm = static_cast<float>(v_norms[t]);",
    ]

    for r in range(repeat_count):
        lines += [
            f"            {{ float om = lmax_{r};",
            f"              lmax_{r} = max(lmax_{r}, s_{r});",
            f"              float rs = exp(om - lmax_{r});",
            f"              float w = exp(s_{r} - lmax_{r});",
            f"              lsum_{r} = lsum_{r} * rs + w;",
            f"              for (int i = 0; i < DimsPerLane; i++) {{",
            f"                  lacc_{r}[i] *= rs;",
            f"                  if (v_valids[i]) {{",
            f"                      uint vv = (vt[v_words[i]] >> v_offs[i]);",
            f"                      if (v_spills[i]) vv |= vt[v_words[i]+1] << ({bits} - (v_offs[i]+{bits}-32));",
            f"                      lacc_{r}[i] += w * val_codebook[vv & {val_mask}u] * vnorm;",
            f"                  }}",
            f"              }}",
            f"            }}",
        ]

    lines += ["        }", ""]

    # Write unnormalized acc + scalar sum/max for cross-tile reduction
    for r in range(repeat_count):
        lines += [
            f"        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32) {{",
            f"            if (vd < Dim) out_acc[((bh*num_tok_tiles+tok_tile)*RepeatCount+{r})*Dim+vd] = lacc_{r}[i];",
            f"        }}",
            f"        if (lane == 0) {{",
            f"            int sm_base = (bh*num_tok_tiles+tok_tile)*RepeatCount+{r};",
            f"            out_sum[sm_base] = lsum_{r};",
            f"            out_max[sm_base] = lmax_{r};",
            f"        }}",
        ]

    return mx.fast.metal_kernel(
        name=f"turboquant_fused_integer_single_tile_{bits}_r{repeat_count}",
        input_names=[
            "q_rot",
            "q_proj",
            "key_norms",
            "key_mse",
            "key_res_norms",
            "key_signs",
            "val_norms",
            "val_packed",
            "key_codebook",
            "key_scale",
            "val_codebook",
        ],
        output_names=["out_acc", "out_sum", "out_max"],
        source="\n".join(lines),
    )

def _fully_fused_decode_kernel(
    bits: int,
    repeat_count: int,
    dims_per_lane: int,
    key_mse_bits: int = -1,
    use_rht: bool = False,
):
    """Fully fused decode: score + online softmax + value + normalize + rotation
    in a single Metal dispatch. Processes ALL tokens (no tiling), outputs in
    original space. Reduces dispatch count from 7 to 1."""
    if not _metal_available() or repeat_count < 1:
        return None

    mse_bits = key_mse_bits if key_mse_bits >= 0 else max(bits - 1, 0)
    mse_mask = (1 << mse_bits) - 1
    val_mask = (1 << bits) - 1
    num_mse_entries = 1 << mse_bits

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto bh = thread_position_in_grid.z;",
        "        auto token_count = key_norms_shape[2];",
        "        auto kv_heads = key_norms_shape[1];",
        "        auto b = bh / kv_heads;",
        "        auto h = bh % kv_heads;",
        "        auto base = (b * kv_heads + h);",
        "",
        "        auto k_norms = key_norms + base * token_count;",
        "        auto k_mse = key_mse + base * token_count * KMsePackedWidth;",
        "        auto k_res = key_res_norms + base * token_count;",
        "        auto k_signs = key_signs + base * token_count * KSignPackedWidth;",
        "        auto v_norms = val_norms + base * token_count;",
        "        auto v_packed = val_packed + base * token_count * VPackedWidth;",
        "",
        "        // Precompute value bit offsets",
        "        int v_words[DimsPerLane], v_offs[DimsPerLane];",
        "        bool v_spills[DimsPerLane], v_valids[DimsPerLane];",
        "        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32) {",
        f"            v_valids[i] = vd < Dim;",
        f"            int vbo = vd * {bits};",
        f"            v_words[i] = vbo / 32;",
        f"            v_offs[i] = vbo % 32;",
        f"            v_spills[i] = (vbo % 32 + {bits}) > 32;",
        "        }",
        "",
    ]

    for r in range(repeat_count):
        lines += [
            f"        auto qr_{r} = q_rot + (base * RepeatCount + {r}) * Dim;",
            f"        auto qp_{r} = q_proj + (base * RepeatCount + {r}) * Dim;",
        ]

    for r in range(repeat_count):
        lines += [
            f"        float lmax_{r} = -INFINITY, lsum_{r} = 0.0f;",
            f"        float lacc_{r}[DimsPerLane] = {{}};",
        ]

    # Token loop: score + online softmax + value accumulation
    lines += [
        "",
        "        for (int t = 0; t < (int)token_count; t++) {",
        "            auto mse_t = k_mse + t * KMsePackedWidth;",
        "            auto sign_t = k_signs + t * KSignPackedWidth;",
        "            float kn = static_cast<float>(k_norms[t]);",
        "            float ksr = kn * key_scale[0] * static_cast<float>(k_res[t]);",
        "",
    ]

    # Score with grouped codebook optimization
    for r in range(repeat_count):
        lines += [f"            float ps_{r} = 0.0f;"]

    lines += [
        f"            for (int d = lane; d < Dim; d += 32) {{",
        f"                int bo = d * {mse_bits};",
        f"                uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                if (((bo & 31) + {mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({mse_bits} - ((bo & 31) + {mse_bits} - 32));",
        f"                idx &= {mse_mask}u;",
        f"                float code = key_codebook[idx];",
        f"                uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [
            f"                ps_{r} += kn * static_cast<float>(qr_{r}[d]) * code + ksr * (sb ? static_cast<float>(qp_{r}[d]) : -static_cast<float>(qp_{r}[d]));"
        ]
    lines += ["            }"]

    for r in range(repeat_count):
        lines += [f"            float s_{r} = simd_sum(ps_{r});"]

    # Online softmax + value accumulation
    lines += [
        "",
        "            auto vt = v_packed + t * VPackedWidth;",
        "            float vnorm = static_cast<float>(v_norms[t]);",
    ]

    for r in range(repeat_count):
        lines += [
            f"            {{ float om = lmax_{r};",
            f"              lmax_{r} = max(lmax_{r}, s_{r});",
            f"              float rs = exp(om - lmax_{r});",
            f"              float w = exp(s_{r} - lmax_{r});",
            f"              lsum_{r} = lsum_{r} * rs + w;",
            f"              for (int i = 0; i < DimsPerLane; i++) {{",
            f"                  lacc_{r}[i] *= rs;",
            f"                  if (v_valids[i]) {{",
            f"                      uint vv = (vt[v_words[i]] >> v_offs[i]);",
            f"                      if (v_spills[i]) vv |= vt[v_words[i]+1] << ({bits} - (v_offs[i]+{bits}-32));",
            f"                      lacc_{r}[i] += w * val_codebook[vv & {val_mask}u] * vnorm;",
            f"                  }}",
            f"              }}",
            f"            }}",
        ]

    lines += ["        }", ""]

    # Normalize by softmax sum
    for r in range(repeat_count):
        lines += [
            f"        float inv_sum_{r} = 1.0f / lsum_{r};",
            f"        for (int i = 0; i < DimsPerLane; i++)",
            f"            lacc_{r}[i] *= inv_sum_{r};",
        ]

    # Fuse rotation via shared memory
    if use_rht:
        # RHT inverse: butterfly WHT on each repeat's accumulated values
        # Process one repeat at a time to reuse shared memory
        lines += [
            "",
            f"        threadgroup float shared_rot[DimPadded];",
            f"        float wht_tmp[DimsPerLanePadded];",
        ]
        for r in range(repeat_count):
            lines += [
                f"        // RHT inverse for repeat {r}",
                f"        for (int i = 0, vd = lane; i < DimsPerLanePadded; vd += 32, i++)",
                f"            shared_rot[vd < Dim ? vd : 0] = (vd < Dim) ? lacc_{r}[i] : 0.0f;",
            ]
            if repeat_count > 1 or True:
                # Zero padding region
                lines += [
                    f"        if (lane < (DimPadded - Dim) && (Dim + lane) < DimPadded)",
                    f"            shared_rot[Dim + lane] = 0.0f;",
                    f"        threadgroup_barrier(mem_flags::mem_threadgroup);",
                ]
            lines += _metal_butterfly_wht_inverse(
                "shared_rot", "sign_vec", "wht_tmp", "DimsPerLanePadded"
            )
            lines += [
                f"        for (int i = 0, vd = lane; i < DimsPerLane; vd += 32, i++)",
                f"            if (vd < Dim) out[(bh * RepeatCount + {r}) * Dim + vd] = shared_rot[vd];",
                f"",
            ]
    else:
        # Dense rotation via shared memory matmul
        lines += [
            "",
            f"        threadgroup float shared_rot[{repeat_count} * Dim];",
        ]
        for r in range(repeat_count):
            lines += [
                f"        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32)",
                f"            if (vd < Dim) shared_rot[{r} * Dim + vd] = lacc_{r}[i];",
            ]
        lines += [
            "        threadgroup_barrier(mem_flags::mem_threadgroup);",
            "",
            "        // Apply dense rotation: output in original space",
        ]
        for r in range(repeat_count):
            lines += [
                f"        for (int i = 0, vd = lane; i < DimsPerLane; i++, vd += 32) {{",
                f"            if (vd < Dim) {{",
                f"                float rot_val = 0.0f;",
                f"                auto rot_row = rotation_t + vd * Dim;",
                f"                for (int j = 0; j < (int)Dim; j++)",
                f"                    rot_val += shared_rot[{r} * Dim + j] * rot_row[j];",
                f"                out[(bh * RepeatCount + {r}) * Dim + vd] = rot_val;",
                f"            }}",
                f"        }}",
            ]

    name_suffix = "_rht" if use_rht else ""
    rotation_input = "sign_vec" if use_rht else "rotation_t"
    return mx.fast.metal_kernel(
        name=f"turboquant_fully_fused_{bits}_r{repeat_count}{name_suffix}",
        input_names=[
            "q_rot",
            "q_proj",
            "key_norms",
            "key_mse",
            "key_res_norms",
            "key_signs",
            "val_norms",
            "val_packed",
            "key_codebook",
            "key_scale",
            "val_codebook",
            rotation_input,
        ],
        output_names=["out"],
        source="\n".join(lines),
    )

def _gen_unrolled_extract(
    bits: int, n_elems: int, codebook_name: str, bit_off_var: str = ""
) -> list:
    """Generate fully-unrolled byte extraction for n_elems packed at `bits` bits.

    When *bit_off_var* is empty the extraction assumes element 0 starts at bit 0
    of the byte pointer (fast path — compile-time constant offsets, works when
    ``n_elems * bits % 8 == 0``).

    When *bit_off_var* names a Metal ``int`` variable (e.g. ``"k_bit_off"``),
    each element's bit position is computed as ``bit_off_var + i * bits`` at
    runtime, still fully unrolled (one expression per element, no loops).
    """
    mask = (1 << bits) - 1
    exprs: list[str] = []

    if not bit_off_var:
        # --- Fast path: compile-time constant offsets ---
        for i in range(n_elems):
            bit_offset = i * bits
            byte_idx = bit_offset // 8
            bit_in_byte = bit_offset % 8

            if bit_in_byte + bits <= 8:
                if bit_in_byte == 0:
                    expr = f"{codebook_name}[kb[{byte_idx}] & {mask}u]"
                else:
                    expr = (
                        f"{codebook_name}[(kb[{byte_idx}] >> {bit_in_byte}) & {mask}u]"
                    )
            else:
                low_bits = 8 - bit_in_byte
                high_mask = (1 << (bits - low_bits)) - 1
                expr = (
                    f"{codebook_name}[((kb[{byte_idx}] >> {bit_in_byte}) & {(1 << low_bits) - 1}u)"
                    f" | ((kb[{byte_idx + 1}] & {high_mask}u) << {low_bits})]"
                )
            exprs.append(expr)
    else:
        # --- General path: runtime bit offset, still fully unrolled ---
        for i in range(n_elems):
            bo = f"({bit_off_var} + {i * bits})" if i else bit_off_var
            by = f"({bo} >> 3)"
            bi = f"({bo} & 7)"
            expr = (
                f"{codebook_name}["
                f"(({bi} + {bits} <= 8)"
                f" ? (kb[{by}] >> {bi})"
                f" : ((kb[{by}] >> {bi}) | (kb[{by} + 1] << (8 - {bi}))))"
                f" & {mask}u]"
            )
            exprs.append(expr)
    return exprs

def _gen_unrolled_score(bits: int, n_elems: int, bit_off_var: str = "") -> str:
    """Generate score accumulation with unrolled key extraction."""
    exprs = _gen_unrolled_extract(bits, n_elems, "key_codebook", bit_off_var)
    terms = [f"q[{i}] * {expr}" for i, expr in enumerate(exprs)]
    return "\n                + ".join(terms)

def _gen_unrolled_value(bits: int, n_elems: int, bit_off_var: str = "") -> str:
    """Generate value accumulation with unrolled extraction."""
    exprs = _gen_unrolled_extract(bits, n_elems, "val_codebook", bit_off_var)
    exprs = [e.replace("kb[", "vb[") for e in exprs]
    lines = []
    for i, expr in enumerate(exprs):
        lines.append(f"            o[{i}] = o[{i}] * factor + exp_score * {expr} * vn;")
    return "\n".join(lines)

def _fused_mse_decode_kernel(key_bits: int, val_bits: int, dim: int = 256):
    """Fused MSE decode: 32 simdgroups × 32 lanes, online softmax + weighted sum."""
    if not _metal_available() or key_bits <= 0 or val_bits <= 0:
        return None
    if dim < 32 or dim % 32 != 0:
        return None  # dim must be a multiple of 32 SIMD lanes

    k_mask = (1 << key_bits) - 1
    v_mask = (1 << val_bits) - 1
    elems_per_lane = dim // 32

    source = f"""
        constexpr int BN = 32;  // simdgroups per threadgroup
        constexpr int BD = 32;  // lanes per simdgroup
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        constexpr uint k_mask = {k_mask}u;
        constexpr uint v_mask = {v_mask}u;
        constexpr int k_bits = {key_bits};
        constexpr int v_bits = {val_bits};

        typedef float U;

        // Thread identity — grid is total threads, so use threadgroup position
        auto bqh = threadgroup_position_in_grid.x;  // batch*q_head index
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto bh = bqh / RepeatCount;  // map q_head -> kv_head

        auto k_nm = key_norms + bh * token_count;
        auto k_pk = key_packed + bh * token_count * KPackedWidth;
        auto v_nm = val_norms + bh * token_count;
        auto v_pk = val_packed + bh * token_count * VPackedWidth;

        // Shared memory for cross-simdgroup reduction
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];
        threadgroup U shared[BN * BD];

        // Preload pre-rotated query into registers
        thread U q[qk_per_thread];
        auto qr = queries + bqh * Dim + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++)
            q[i] = static_cast<U>(qr[i]);

        // Initialize accumulators
        thread U o[v_per_thread] = {{}};
        U max_score = -INFINITY;
        U sum_exp_score = 0;

        // Byte/bit offset for this lane's first element (constant across all tokens)
        int k_bit_start = simd_lid * qk_per_thread * k_bits;
        int v_bit_start = simd_lid * v_per_thread * v_bits;
        int k_byte_base = k_bit_start >> 3;
        int v_byte_base = v_bit_start >> 3;
        {"int k_bit_off = k_bit_start & 7;" if (elems_per_lane * key_bits) % 8 else ""}
        {"int v_bit_off = v_bit_start & 7;" if (elems_per_lane * val_bits) % 8 else ""}

        // KV loop: each simdgroup handles tokens simd_gid, simd_gid+32, ...
        for (int t = simd_gid; t < (int)token_count; t += BN) {{
            U kn = static_cast<U>(k_nm[t]);

            // Key score — unrolled byte extraction
            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth) + k_byte_base;
            U score = {_gen_unrolled_score(key_bits, elems_per_lane, "k_bit_off" if (elems_per_lane * key_bits) % 8 else "")};
            score = simd_sum(score) * kn;

            // Online softmax
            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth) + v_byte_base;
            U vn = static_cast<U>(v_nm[t]);

            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            // Value accumulation — unrolled byte extraction
            {_gen_unrolled_value(val_bits, elems_per_lane, "v_bit_off" if (elems_per_lane * val_bits) % 8 else "")}
        }}

        // Cross-simdgroup reduction (matches MLX SDPA pattern)
        // 1. Communicate max and sum_exp across simdgroups
        if (simd_lid == 0) {{
            max_scores[simd_gid] = max_score;
            sum_exp_scores[simd_gid] = sum_exp_score;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Reduce max and sum across all simdgroups
        U sg_max = max_scores[simd_lid];
        U new_max = simd_max(sg_max);
        U factor = fast::exp(sg_max - new_max);
        U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);

        // 3. Rescale this simdgroup's factor for the global max
        U my_factor = fast::exp(max_score - new_max);

        // 4. Transpose-reduce outputs through shared memory
        for (int i = 0; i < v_per_thread; i++) {{
            shared[simd_lid * BD + simd_gid] = o[i] * my_factor;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(shared[simd_gid * BD + simd_lid]);
            o[i] = total_sum > 0 ? o[i] / total_sum : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Write output (in rotated space — caller applies inverse rotation)
        if (simd_lid == 0) {{
            for (int i = 0; i < v_per_thread; i++) {{
                out[bqh * Dim + simd_gid * v_per_thread + i] = static_cast<U>(o[i]);
            }}
        }}
    """

    return mx.fast.metal_kernel(
        name=f"turboquant_fused_mse_sdpa_k{key_bits}_v{val_bits}_d{dim}",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
        ],
        output_names=["out"],
        source=source,
    )

def _fused_mse_decode_2pass_1_kernel(key_bits: int, val_bits: int, dim: int = 256):
    """2-pass decode pass 1: block-parallel quantized attention."""
    if not _metal_available() or key_bits <= 0 or val_bits <= 0:
        return None
    if dim < 32 or dim % 32 != 0:
        return None

    k_mask = (1 << key_bits) - 1
    v_mask = (1 << val_bits) - 1
    elems_per_lane = dim // 32

    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        constexpr uint k_mask = {k_mask}u;
        constexpr uint v_mask = {v_mask}u;
        constexpr int k_bits = {key_bits};
        constexpr int v_bits = {val_bits};
        typedef float U;

        // Thread identity — matches sdpa_vector_2pass_1 layout
        auto kv_head_idx = threadgroup_position_in_grid.x;
        auto batch_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto simd_lid = thread_index_in_simdgroup;
        auto gqa_idx = thread_position_in_threadgroup.y;  // which q_head within kv_head

        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto bh = batch_idx * kv_heads + kv_head_idx;
        auto bqh = batch_idx * kv_heads * RepeatCount + kv_head_idx * RepeatCount + gqa_idx;

        auto k_nm = key_norms + bh * token_count;
        auto k_pk = key_packed + bh * token_count * KPackedWidth;
        auto v_nm = val_norms + bh * token_count;
        auto v_pk = val_packed + bh * token_count * VPackedWidth;

        // Load pre-rotated query
        thread U q[qk_per_thread];
        auto qr = queries + bqh * Dim + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++)
            q[i] = static_cast<U>(qr[i]);

        thread U o[v_per_thread] = {{}};
        U max_score = -INFINITY;
        U sum_exp_score = 0;

        // Byte/bit offset for this lane's first element (constant across all tokens)
        int k_bit_start = simd_lid * qk_per_thread * k_bits;
        int v_bit_start = simd_lid * v_per_thread * v_bits;
        int k_byte_base = k_bit_start >> 3;
        int v_byte_base = v_bit_start >> 3;
        {"int k_bit_off = k_bit_start & 7;" if (elems_per_lane * key_bits) % 8 else ""}
        {"int v_bit_off = v_bit_start & 7;" if (elems_per_lane * val_bits) % 8 else ""}

        // KV loop: stride by blocks (each block handles a different subset)
        for (int t = block_idx; t < (int)token_count; t += Blocks) {{
            U kn = static_cast<U>(k_nm[t]);

            // Key score — unrolled byte extraction
            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth) + k_byte_base;
            U score = {_gen_unrolled_score(key_bits, elems_per_lane, "k_bit_off" if (elems_per_lane * key_bits) % 8 else "")};
            score = simd_sum(score) * kn;

            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth) + v_byte_base;
            U vn = static_cast<U>(v_nm[t]);

            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            {_gen_unrolled_value(val_bits, elems_per_lane, "v_bit_off" if (elems_per_lane * val_bits) % 8 else "")}
        }}

        // Write partial results for this block
        if (simd_lid == 0) {{
            out_sums[bqh * Blocks + block_idx] = sum_exp_score;
            out_maxs[bqh * Blocks + block_idx] = max_score;
        }}
        for (int i = 0; i < v_per_thread; i++)
            out_acc[(bqh * Blocks + block_idx) * Dim + simd_lid * v_per_thread + i] =
                static_cast<U>(o[i]);
    """

    return mx.fast.metal_kernel(
        name=f"turboquant_mse_sdpa_2pass1_k{key_bits}_v{val_bits}_d{dim}",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source=source,
    )

def _fused_mse_decode_2pass_2_kernel():
    """2-pass decode pass 2: reduce partial block results."""
    if not _metal_available():
        return None

    source = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = Dim / BD;
        typedef float U;

        thread U o[elem_per_thread] = {};
        threadgroup U outputs[BN * BD];

        auto head_idx = threadgroup_position_in_grid.x;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto p = partials + head_idx * Blocks * Dim + simd_gid * Dim +
                 simd_lid * elem_per_thread;
        auto s = sums + head_idx * Blocks;
        auto m = maxs + head_idx * Blocks;

        U max_score = -INFINITY;
        U sum_exp_score = 0;

        // Each simdgroup handles a subset of blocks (stride BN)
        for (int b = simd_gid; b < Blocks; b += BN) {
            U block_max = m[b];
            U block_sum = s[b];

            U new_max = max(max_score, block_max);
            U factor = fast::exp(max_score - new_max);
            U block_factor = fast::exp(block_max - new_max);

            sum_exp_score = sum_exp_score * factor + block_sum * block_factor;
            for (int i = 0; i < elem_per_thread; i++) {
                o[i] = o[i] * factor +
                    partials[(head_idx * Blocks + b) * Dim +
                             simd_lid * elem_per_thread + i] * block_factor;
            }
            max_score = new_max;
        }

        // Cross-simdgroup reduction (same as sdpa_vector_2pass_2)
        threadgroup U sg_maxs[BN];
        threadgroup U sg_sums[BN];
        if (simd_lid == 0) {
            sg_maxs[simd_gid] = max_score;
            sg_sums[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        U sg_max = sg_maxs[simd_lid];
        U new_max = simd_max(sg_max);
        U factor = fast::exp(sg_max - new_max);
        U total_sum = simd_sum(sg_sums[simd_lid] * factor);

        U my_factor = fast::exp(max_score - new_max);

        for (int i = 0; i < elem_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i] * my_factor;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = total_sum > 0 ? o[i] / total_sum : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; i++)
                out[head_idx * Dim + simd_gid * elem_per_thread + i] =
                    static_cast<U>(o[i]);
        }
    """

    return mx.fast.metal_kernel(
        name="turboquant_mse_sdpa_2pass2",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=source,
    )

def _fused_split_decode_kernel(low_bits: int, high_bits: int, repeat_count: int):
    """Fused SplitCodec decode: score + online-softmax + weighted-sum."""
    if not _metal_available() or repeat_count < 1:
        return None

    low_mse_bits = max(low_bits - 1, 0)
    high_mse_bits = max(high_bits - 1, 0)
    low_mse_mask = (1 << low_mse_bits) - 1
    high_mse_mask = (1 << high_mse_bits) - 1

    val_dim_total = "DimLow + DimHigh"

    lines = [
        "        auto lane = thread_position_in_grid.x;",
        "        auto val_tile = thread_position_in_grid.y;",
        "        auto n = thread_position_in_grid.z;",
        "",
        f"        int val_dim = val_tile * 32 + lane;",
        "",
        "        auto token_count = key_low_norms_shape[2];",
        "        auto kv_heads = key_low_norms_shape[1];",
        "        auto num_tok_tiles = (token_count + TokTileSize - 1) / TokTileSize;",
        "        auto bh = n / num_tok_tiles;",
        "        auto tok_tile = n % num_tok_tiles;",
        "        auto b = bh / kv_heads;",
        "        auto h = bh % kv_heads;",
        "        auto base = (b * kv_heads + h);",
        "",
        "        int t_start = tok_tile * TokTileSize;",
        "        int t_end = min(t_start + TokTileSize, (int)token_count);",
        "",
        "        auto kl_norms = key_low_norms + base * token_count;",
        "        auto kl_mse = key_low_mse + base * token_count * KLMsePackedWidth;",
        "        auto kl_res = key_low_res_norms + base * token_count;",
        "        auto kl_signs = key_low_signs + base * token_count * KLSignPackedWidth;",
        "        auto kh_norms = key_high_norms + base * token_count;",
        "        auto kh_mse = key_high_mse + base * token_count * KHMsePackedWidth;",
        "        auto kh_res = key_high_res_norms + base * token_count;",
        "        auto kh_signs = key_high_signs + base * token_count * KHSignPackedWidth;",
        "",
        "        auto vl_norms = val_low_norms + base * token_count;",
        "        auto vl_packed = val_low_packed + base * token_count * VLPackedWidth;",
        "        auto vh_norms = val_high_norms + base * token_count;",
        "        auto vh_packed = val_high_packed + base * token_count * VHPackedWidth;",
        "",
        "        // Value bit offset for this lane's dim",
        f"        bool is_low_val = val_dim < DimLow;",
        f"        int vd_local = is_low_val ? val_dim : (val_dim - DimLow);",
        f"        int v_bits = is_low_val ? VLBits : VHBits;",
        f"        int v_bo = vd_local * v_bits;",
        f"        int v_word = v_bo / 32;",
        f"        int v_off = v_bo % 32;",
        f"        uint v_mask = (1u << v_bits) - 1u;",
        f"        bool v_spills = (v_off + v_bits > 32);",
        f"        bool v_valid = val_dim < ({val_dim_total});",
        "",
    ]

    for r in range(repeat_count):
        lines += [
            f"        auto qrl_{r} = q_rot_low + (base * RepeatCount + {r}) * DimLow;",
            f"        auto qpl_{r} = q_proj_low + (base * RepeatCount + {r}) * DimLow;",
            f"        auto qrh_{r} = q_rot_high + (base * RepeatCount + {r}) * DimHigh;",
            f"        auto qph_{r} = q_proj_high + (base * RepeatCount + {r}) * DimHigh;",
        ]

    for r in range(repeat_count):
        lines += [
            f"        float lmax_{r} = -INFINITY;",
            f"        float lsum_{r} = 0.0f;",
            f"        float lacc_{r} = 0.0f;",
        ]

    lines += [
        "",
        "        for (int t = t_start; t < t_end; t++) {",
    ]

    for r in range(repeat_count):
        lines += [f"            float ps_{r} = 0.0f;"]

    # Low half scoring — hoisted ksr, conditional negate
    lines += [
        "            {",
        "                auto mse_t = kl_mse + t * KLMsePackedWidth;",
        "                auto sign_t = kl_signs + t * KLSignPackedWidth;",
        "                float kn = static_cast<float>(kl_norms[t]);",
        "                float ksr = kn * key_low_scale[0] * static_cast<float>(kl_res[t]);",
        f"                for (int d = lane; d < DimLow; d += 32) {{",
        f"                    int bo = d * {low_mse_bits};",
        f"                    uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                    if (((bo & 31) + {low_mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({low_mse_bits} - ((bo & 31) + {low_mse_bits} - 32));",
        f"                    idx &= {low_mse_mask}u;",
        f"                    float code = key_low_codebook[idx];",
        f"                    uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [
            f"                    ps_{r} += kn * static_cast<float>(qrl_{r}[d]) * code + ksr * (sb ? static_cast<float>(qpl_{r}[d]) : -static_cast<float>(qpl_{r}[d]));"
        ]
    lines += [
        "                }",
        "            }",
    ]
    # High half scoring
    lines += [
        "            {",
        "                auto mse_t = kh_mse + t * KHMsePackedWidth;",
        "                auto sign_t = kh_signs + t * KHSignPackedWidth;",
        "                float kn = static_cast<float>(kh_norms[t]);",
        "                float ksr = kn * key_high_scale[0] * static_cast<float>(kh_res[t]);",
        f"                for (int d = lane; d < DimHigh; d += 32) {{",
        f"                    int bo = d * {high_mse_bits};",
        f"                    uint idx = (mse_t[bo >> 5] >> (bo & 31));",
        f"                    if (((bo & 31) + {high_mse_bits}) > 32) idx |= mse_t[(bo >> 5) + 1] << ({high_mse_bits} - ((bo & 31) + {high_mse_bits} - 32));",
        f"                    idx &= {high_mse_mask}u;",
        f"                    float code = key_high_codebook[idx];",
        f"                    uint sb = (sign_t[d >> 5] >> (d & 31)) & 1u;",
    ]
    for r in range(repeat_count):
        lines += [
            f"                    ps_{r} += kn * static_cast<float>(qrh_{r}[d]) * code + ksr * (sb ? static_cast<float>(qph_{r}[d]) : -static_cast<float>(qph_{r}[d]));"
        ]
    lines += [
        "                }",
        "            }",
    ]

    for r in range(repeat_count):
        lines += [f"            float s_{r} = simd_sum(ps_{r});"]

    # Value decode + online softmax accumulation
    lines += [
        "",
        "            float v_code = 0.0f;",
        "            if (v_valid) {",
        "                if (is_low_val) {",
        "                    auto vt = vl_packed + t * VLPackedWidth;",
        "                    uint vv = (vt[v_word] >> v_off);",
        "                    if (v_spills) vv |= vt[v_word + 1] << (v_bits - (v_off + v_bits - 32));",
        "                    v_code = val_low_codebook[vv & v_mask] * static_cast<float>(vl_norms[t]);",
        "                } else {",
        "                    auto vt = vh_packed + t * VHPackedWidth;",
        "                    uint vv = (vt[v_word] >> v_off);",
        "                    if (v_spills) vv |= vt[v_word + 1] << (v_bits - (v_off + v_bits - 32));",
        "                    v_code = val_high_codebook[vv & v_mask] * static_cast<float>(vh_norms[t]);",
        "                }",
        "            }",
    ]

    for r in range(repeat_count):
        lines += [
            f"            float om_{r} = lmax_{r};",
            f"            lmax_{r} = max(lmax_{r}, s_{r});",
            f"            float rs_{r} = exp(om_{r} - lmax_{r});",
            f"            float w_{r} = exp(s_{r} - lmax_{r});",
            f"            lsum_{r} = lsum_{r} * rs_{r} + w_{r};",
            f"            lacc_{r} = lacc_{r} * rs_{r} + w_{r} * v_code;",
        ]

    lines += ["        }", ""]

    # Write acc per val_dim, but sum/max are identical across lanes —
    # only write once per (bh, tok_tile, repeat) to avoid redundant writes.
    lines += [f"        int out_stride = ({val_dim_total});"]
    for r in range(repeat_count):
        lines += [
            f"        if (v_valid) {{",
            f"            out_acc[((bh * num_tok_tiles + tok_tile) * RepeatCount + {r}) * out_stride + val_dim] = lacc_{r};",
            f"        }}",
            f"        if (val_dim == 0) {{",
            f"            int sm_base = (bh * num_tok_tiles + tok_tile) * RepeatCount + {r};",
            f"            out_sum[sm_base] = lsum_{r};",
            f"            out_max[sm_base] = lmax_{r};",
            f"        }}",
        ]

    source = "\n".join(lines)

    input_names = [
        "q_rot_low",
        "q_proj_low",
        "q_rot_high",
        "q_proj_high",
        "key_low_norms",
        "key_low_mse",
        "key_low_res_norms",
        "key_low_signs",
        "key_high_norms",
        "key_high_mse",
        "key_high_res_norms",
        "key_high_signs",
        "val_low_norms",
        "val_low_packed",
        "val_high_norms",
        "val_high_packed",
        "key_low_codebook",
        "key_high_codebook",
        "key_low_scale",
        "key_high_scale",
        "val_low_codebook",
        "val_high_codebook",
    ]

    return mx.fast.metal_kernel(
        name=f"turboquant_fused_split_decode_{low_bits}_{high_bits}_r{repeat_count}",
        input_names=input_names,
        output_names=["out_acc", "out_sum", "out_max"],
        source=source,
    )

def _compiled_split_decode_kernel(low_bits: int, high_bits: int):
    """Compiled SplitCodec decode handling both halves."""
    low_mse_bits = max(low_bits - 1, 0)
    high_mse_bits = max(high_bits - 1, 0)

    @mx.compile
    def _decode(
        grouped_queries: mx.array,
        # Low key state
        key_low_norms: mx.array,
        key_low_mse_indices: mx.array,
        key_low_residual_norms: mx.array,
        key_low_qjl_signs: mx.array,
        # High key state
        key_high_norms: mx.array,
        key_high_mse_indices: mx.array,
        key_high_residual_norms: mx.array,
        key_high_qjl_signs: mx.array,
        # Low value state
        value_low_norms: mx.array,
        value_low_indices: mx.array,
        # High value state
        value_high_norms: mx.array,
        value_high_indices: mx.array,
        # Codec params
        key_low_transform_t: mx.array,
        key_high_transform_t: mx.array,
        key_low_codebook: mx.array,
        key_high_codebook: mx.array,
        key_low_scale: mx.array,
        key_high_scale: mx.array,
        value_low_codebook: mx.array,
        value_high_codebook: mx.array,
        value_low_rotation: mx.array,
        value_high_rotation: mx.array,
        # Split indices
        low_idx: mx.array,
        high_idx: mx.array,
        restore_order: mx.array,
    ) -> mx.array:
        dim = grouped_queries.shape[-1]
        # Split queries by dimension
        q_low = mx.take(grouped_queries, low_idx, axis=-1)
        q_high = mx.take(grouped_queries, high_idx, axis=-1)

        # Score low half
        qt_low = mx.matmul(q_low, key_low_transform_t)
        d_low = q_low.shape[-1]
        scores_low = _metal_prod_score(
            qt_low[..., :d_low].reshape(
                qt_low.shape[0], qt_low.shape[1], qt_low.shape[2], d_low
            ),
            qt_low[..., d_low:].reshape(
                qt_low.shape[0], qt_low.shape[1], qt_low.shape[2], d_low
            ),
            TurboQuantProdState(
                key_low_norms,
                key_low_mse_indices,
                key_low_residual_norms,
                key_low_qjl_signs,
            ),
            low_mse_bits,
            key_low_codebook,
            key_low_scale,
        )

        # Score high half
        qt_high = mx.matmul(q_high, key_high_transform_t)
        d_high = q_high.shape[-1]
        scores_high = _metal_prod_score(
            qt_high[..., :d_high].reshape(
                qt_high.shape[0], qt_high.shape[1], qt_high.shape[2], d_high
            ),
            qt_high[..., d_high:].reshape(
                qt_high.shape[0], qt_high.shape[1], qt_high.shape[2], d_high
            ),
            TurboQuantProdState(
                key_high_norms,
                key_high_mse_indices,
                key_high_residual_norms,
                key_high_qjl_signs,
            ),
            high_mse_bits,
            key_high_codebook,
            key_high_scale,
        )

        # Combined scores
        scores = scores_low + scores_high

        # Weighted sum of low values
        out_low = _metal_mse_weighted_sum_from_scores(
            scores,
            TurboQuantMSEState(value_low_norms, value_low_indices),
            low_bits,
            value_low_codebook,
            value_low_rotation,
        )

        # Weighted sum of high values
        out_high = _metal_mse_weighted_sum_from_scores(
            scores,
            TurboQuantMSEState(value_high_norms, value_high_indices),
            high_bits,
            value_high_codebook,
            value_high_rotation,
        )

        # Merge and reorder
        merged = mx.concatenate([out_low, out_high], axis=-1)
        return mx.take(merged, restore_order, axis=-1)

    return _decode

