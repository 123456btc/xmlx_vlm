# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from xmlx_vlm.models.cache import make_prompt_cache
from xmlx_vlm.specprefill import (
    score_tokens,
    select_chunks,
    sparse_prefill,
    cleanup_rope,
    _PositionMappedRoPE,
    _OffsetAdjustedRoPE,
)
from xmlx_vlm.generate.single import generate_step


class MockAttention(nn.Module):
    def __init__(self, dims=8, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims // 2)
        self.v_proj = nn.Linear(dims, dims // 2)
        self.o_proj = nn.Linear(dims, dims)
        self.rope = nn.RoPE(dims // num_heads)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        if self.num_attention_heads != self.num_key_value_heads:
            heads_per_group = self.num_attention_heads // self.num_key_value_heads
            k = mx.repeat(k, heads_per_group, axis=1)
            v = mx.repeat(v, heads_per_group, axis=1)

        scale = q.shape[-1] ** -0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            scores = scores + mask
        w = mx.softmax(scores.astype(mx.float32), axis=-1)
        out = w @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.o_proj(out)


class MockLayer(nn.Module):
    def __init__(self, dims=8):
        super().__init__()
        self.self_attn = MockAttention(dims)
        self.input_layernorm = nn.RMSNorm(dims)
        self.post_attention_layernorm = nn.RMSNorm(dims)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h


class MockLanguageModel(nn.Module):
    def __init__(self, dims=8, vocab_size=100, num_layers=2):
        super().__init__()
        self.model = self
        self.embed_tokens = nn.Embedding(vocab_size, dims)
        self.layers = [MockLayer(dims) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(dims)
        self.lm_head = nn.Linear(dims, vocab_size)

    def __call__(self, x, cache=None, **kwargs):
        h = self.embed_tokens(x)
        L = h.shape[1]
        mask = None
        if L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        
        # In mock call, cache is a list of KVCache objects
        for i, layer in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            h = layer(h, mask=mask, cache=c)
        h = self.norm(h)
        return SimpleNamespace(
            logits=self.lm_head(h),
            cross_attention_states=None,
            encoder_outputs=None,
            shared_kv_states=None,
            hidden_states=None,
        )


class MockVLM(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm
        self.config = SimpleNamespace(
            model_type="llama",
            image_token_index=999
        )

    def get_input_embeddings(self, input_ids, pixel_values=None, mask=None, **kwargs):
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        return SimpleNamespace(
            inputs_embeds=inputs_embeds,
            to_dict=lambda: {}
        )


class TestSpecPrefill(unittest.TestCase):
    def setUp(self):
        self.dims = 8
        self.vocab_size = 100
        self.draft_model = MockLanguageModel(self.dims, self.vocab_size, num_layers=2)
        self.target_model_lm = MockLanguageModel(self.dims, self.vocab_size, num_layers=2)
        self.target_vlm = MockVLM(self.target_model_lm)

    def test_score_tokens(self):
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        importance = score_tokens(
            self.draft_model,
            tokens,
            n_lookahead=2,
            pool_kernel=3,
        )
        self.assertEqual(importance.shape[0], len(tokens))
        self.assertTrue(mx.all(importance >= 0.0).item())

    def test_select_chunks(self):
        importance = mx.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.7, 0.8])
        # keep 50% chunk size 2
        selected = select_chunks(importance, keep_pct=0.5, chunk_size=2)
        selected_list = selected.tolist()
        self.assertTrue(len(selected_list) > 0)
        # Verify it is sorted
        self.assertEqual(selected_list, sorted(selected_list))

    def test_sparse_prefill(self):
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        selected_indices = mx.array([0, 2, 3, 5, 7])
        cache = make_prompt_cache(self.target_model_lm)
        
        sparse_logits = sparse_prefill(
            self.target_model_lm,
            tokens,
            selected_indices,
            cache,
        )
        self.assertEqual(sparse_logits.shape, (1, 1, self.vocab_size))
        
        # Verify that RoPE wrapper _OffsetAdjustedRoPE is installed
        attn = self.target_model_lm.layers[0].self_attn
        self.assertTrue(isinstance(attn.rope, _OffsetAdjustedRoPE))
        
        # Verify cleanup restores original RoPE
        cleanup_rope(self.target_model_lm)
        self.assertTrue(isinstance(attn.rope, nn.RoPE))

    def test_generate_step_integration(self):
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        gen = generate_step(
            input_ids,
            self.target_vlm,
            pixel_values=None,
            mask=None,
            max_tokens=3,
            enable_specprefill=True,
            specprefill_draft_model=self.draft_model,
            specprefill_threshold=4,
            specprefill_keep_pct=0.5,
            specprefill_chunk_size=2,
        )
        
        tokens_gen = []
        for token, logprobs in gen:
            tokens_gen.append(token)
            
        self.assertEqual(len(tokens_gen), 3)
        # Verify original RoPE was cleaned up properly
        attn = self.target_model_lm.layers[0].self_attn
        self.assertTrue(isinstance(attn.rope, nn.RoPE))


if __name__ == "__main__":
    unittest.main()
