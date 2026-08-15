"""Tests for Per-Layer KV Cache Quantization."""

import os
import pytest
import mlx.core as mx
import mlx.nn as nn

from xmlx_vlm.config import get_kv_bits_per_layer
from xmlx_vlm.generate.batch.cache_helpers import _make_cache
from xmlx_vlm.generate.single.utils import maybe_quantize_kv_cache
from xmlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache, KVCache
from xmlx_vlm.turboquant import BatchTurboQuantKVCache, TurboQuantKVCache


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()


class DummyModel(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.layers = [DummyLayer() for _ in range(num_layers)]


def test_get_kv_bits_per_layer_defaults():
    # When no config and no default
    assert get_kv_bits_per_layer(8, None, raw_config="") is None

    # When no config but default_bits=4.0
    res = get_kv_bits_per_layer(8, 4.0, raw_config="")
    assert res == [4.0] * 8


def test_get_kv_bits_per_layer_comma_separated():
    res = get_kv_bits_per_layer(4, None, raw_config="8, 8, 4, 3.5")
    assert res == [8.0, 8.0, 4.0, 3.5]

    # Partial list with default pad
    res2 = get_kv_bits_per_layer(6, 4.0, raw_config="8, 8")
    assert res2 == [8.0, 8.0, 4.0, 4.0, 4.0, 4.0]

    # With unquantized "none"
    res3 = get_kv_bits_per_layer(4, None, raw_config="none, 8, 4, 0")
    assert res3 == [None, 8.0, 4.0, None]


def test_get_kv_bits_per_layer_mapping():
    res = get_kv_bits_per_layer(8, None, raw_config="0:8, 1:8, -1:8, default:4")
    assert res[0] == 8.0
    assert res[1] == 8.0
    assert res[-1] == 8.0
    assert res[2] == 4.0
    assert res[3] == 4.0

    # Range syntax
    res2 = get_kv_bits_per_layer(6, None, raw_config="0-1:8, 2-4:3.5, default:none")
    assert res2[0] == 8.0
    assert res2[1] == 8.0
    assert res2[2] == 3.5
    assert res2[3] == 3.5
    assert res2[4] == 3.5
    assert res2[5] is None


def test_get_kv_bits_per_layer_adaptive():
    res = get_kv_bits_per_layer(8, 4.0, raw_config="adaptive")
    assert len(res) == 8
    # Critical first/last layers are boosted to 4.5
    assert res[0] == 4.5
    assert res[1] == 4.5
    assert res[-1] == 4.5
    assert res[-2] == 4.5
    # Middle layers reduced to 3.5
    assert any(b == 3.5 for b in res[2:-2])


def test_make_cache_per_layer():
    model = DummyModel(num_layers=4)
    # Layer 0: FP16 (None), Layer 1: 8-bit uniform, Layer 2: 4-bit uniform, Layer 3: 3.5-bit TurboQuant
    caches = _make_cache(
        model,
        left_padding=[0],
        kv_quant_scheme="turboquant",
        kv_bits_per_layer=[None, 8.0, 4.0, 3.5],
    )

    assert len(caches) == 4
    assert isinstance(caches[0], BatchKVCache)
    assert isinstance(caches[1], BatchTurboQuantKVCache)
    assert caches[1].bits == 8.0
    assert isinstance(caches[2], BatchTurboQuantKVCache)
    assert caches[2].bits == 4.0
    assert isinstance(caches[3], BatchTurboQuantKVCache)
    assert caches[3].bits == 3.5


def test_maybe_quantize_kv_cache_per_layer():
    prompt_cache = [KVCache() for _ in range(4)]
    # Set per-layer bits: Layer 0 unquantized (None), Layer 1: 8-bit, Layer 2: 4-bit, Layer 3: 3.5-bit
    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=None,
        kv_quant_scheme="turboquant",
        kv_bits_per_layer="0:none, 1:8, 2:4, 3:3.5",
    )

    assert isinstance(prompt_cache[0], KVCache)
    assert isinstance(prompt_cache[1], TurboQuantKVCache)
    assert prompt_cache[1].bits == 8.0
    assert isinstance(prompt_cache[2], TurboQuantKVCache)
    assert prompt_cache[2].bits == 4.0
    assert isinstance(prompt_cache[3], TurboQuantKVCache)
    assert prompt_cache[3].bits == 3.5
