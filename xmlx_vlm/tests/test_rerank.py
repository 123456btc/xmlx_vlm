# SPDX-License-Identifier: Apache-2.0
"""Tests for the rerank engine and endpoint."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import xmlx_vlm.server as server
from xmlx_vlm.rerank_engine import (
    RerankEngine,
    SigmoidAdapter,
    _MLXClassifierWrapper,
    get_adapter,
)
from xmlx_vlm.rerank_forward import classifier_forward


# ── Helpers ─────────────────────────────────────────────────────────────────

def _random_bert_weights(hidden_size=8, num_heads=2, num_layers=1, vocab_size=16):
    """Generate minimal random BERT-family weights for testing."""
    head_dim = hidden_size // num_heads
    weights = {}
    p = "bert"

    # Embeddings
    weights[f"{p}.embeddings.word_embeddings.weight"] = mx.random.normal(
        (vocab_size, hidden_size)
    )
    weights[f"{p}.embeddings.position_embeddings.weight"] = mx.random.normal(
        (128, hidden_size)
    )
    weights[f"{p}.embeddings.token_type_embeddings.weight"] = mx.random.normal(
        (2, hidden_size)
    )
    weights[f"{p}.embeddings.LayerNorm.weight"] = mx.ones(hidden_size)
    weights[f"{p}.embeddings.LayerNorm.bias"] = mx.zeros(hidden_size)

    for i in range(num_layers):
        lp = f"{p}.encoder.layer.{i}"
        # Self-attention
        weights[f"{lp}.attention.self.query.weight"] = mx.random.normal(
            (hidden_size, hidden_size)
        )
        weights[f"{lp}.attention.self.query.bias"] = mx.zeros(hidden_size)
        weights[f"{lp}.attention.self.key.weight"] = mx.random.normal(
            (hidden_size, hidden_size)
        )
        weights[f"{lp}.attention.self.key.bias"] = mx.zeros(hidden_size)
        weights[f"{lp}.attention.self.value.weight"] = mx.random.normal(
            (hidden_size, hidden_size)
        )
        weights[f"{lp}.attention.self.value.bias"] = mx.zeros(hidden_size)
        # Attention output
        weights[f"{lp}.attention.output.dense.weight"] = mx.random.normal(
            (hidden_size, hidden_size)
        )
        weights[f"{lp}.attention.output.dense.bias"] = mx.zeros(hidden_size)
        weights[f"{lp}.attention.output.LayerNorm.weight"] = mx.ones(hidden_size)
        weights[f"{lp}.attention.output.LayerNorm.bias"] = mx.zeros(hidden_size)
        # FFN
        weights[f"{lp}.intermediate.dense.weight"] = mx.random.normal(
            (hidden_size * 4, hidden_size)
        )
        weights[f"{lp}.intermediate.dense.bias"] = mx.zeros(hidden_size * 4)
        weights[f"{lp}.output.dense.weight"] = mx.random.normal(
            (hidden_size, hidden_size * 4)
        )
        weights[f"{lp}.output.dense.bias"] = mx.zeros(hidden_size)
        weights[f"{lp}.output.LayerNorm.weight"] = mx.ones(hidden_size)
        weights[f"{lp}.output.LayerNorm.bias"] = mx.zeros(hidden_size)

    # Pooler
    weights[f"{p}.pooler.dense.weight"] = mx.random.normal((hidden_size, hidden_size))
    weights[f"{p}.pooler.dense.bias"] = mx.zeros(hidden_size)

    # Classifier head
    weights["classifier.weight"] = mx.random.normal((1, hidden_size))
    weights["classifier.bias"] = mx.zeros(1)

    return weights


def _random_distilbert_weights(hidden_size=8, num_heads=2, num_layers=1, vocab_size=16):
    """Generate minimal random DistilBERT weights for testing."""
    head_dim = hidden_size // num_heads
    weights = {}
    p = "distilbert"

    # Embeddings
    weights[f"{p}.embeddings.word_embeddings.weight"] = mx.random.normal(
        (vocab_size, hidden_size)
    )
    weights[f"{p}.embeddings.position_embeddings.weight"] = mx.random.normal(
        (128, hidden_size)
    )
    weights[f"{p}.embeddings.LayerNorm.weight"] = mx.ones(hidden_size)
    weights[f"{p}.embeddings.LayerNorm.bias"] = mx.zeros(hidden_size)

    for i in range(num_layers):
        lp = f"{p}.transformer.layer.{i}"
        # Self-attention
        weights[f"{lp}.attention.q_lin.weight"] = mx.random.normal(
            (hidden_size, hidden_size)
        )
        weights[f"{lp}.attention.q_lin.bias"] = mx.zeros(hidden_size)
        weights[f"{lp}.attention.k_lin.weight"] = mx.random.normal(
            (hidden_size, hidden_size)
        )
        weights[f"{lp}.attention.k_lin.bias"] = mx.zeros(hidden_size)
        weights[f"{lp}.attention.v_lin.weight"] = mx.random.normal(
            (hidden_size, hidden_size)
        )
        weights[f"{lp}.attention.v_lin.bias"] = mx.zeros(hidden_size)
        # Attention output
        weights[f"{lp}.attention.out_lin.weight"] = mx.random.normal(
            (hidden_size, hidden_size)
        )
        weights[f"{lp}.attention.out_lin.bias"] = mx.zeros(hidden_size)
        # LayerNorms
        weights[f"{lp}.sa_layer_norm.weight"] = mx.ones(hidden_size)
        weights[f"{lp}.sa_layer_norm.bias"] = mx.zeros(hidden_size)
        weights[f"{lp}.output_layer_norm.weight"] = mx.ones(hidden_size)
        weights[f"{lp}.output_layer_norm.bias"] = mx.zeros(hidden_size)
        # FFN
        weights[f"{lp}.ffn.lin1.weight"] = mx.random.normal((hidden_size * 4, hidden_size))
        weights[f"{lp}.ffn.lin1.bias"] = mx.zeros(hidden_size * 4)
        weights[f"{lp}.ffn.lin2.weight"] = mx.random.normal((hidden_size, hidden_size * 4))
        weights[f"{lp}.ffn.lin2.bias"] = mx.zeros(hidden_size)

    # Classifier head
    weights["classifier.weight"] = mx.random.normal((1, hidden_size))
    weights["classifier.bias"] = mx.zeros(1)

    return weights


# ── Adapter tests ───────────────────────────────────────────────────────────


def test_sigmoid_adapter_normalize():
    adapter = SigmoidAdapter()
    assert adapter.normalize(0.0) == pytest.approx(0.5, abs=1e-6)
    assert adapter.normalize(float("inf")) == pytest.approx(1.0, abs=1e-6)
    assert adapter.normalize(-float("inf")) == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_adapter_extract_score():
    adapter = SigmoidAdapter()
    assert adapter.extract_score([2.0]) == 2.0
    assert adapter.extract_score([-1.5]) == -1.5


def test_get_adapter_returns_sigmoid_by_default():
    adapter = get_adapter("anything")
    assert isinstance(adapter, SigmoidAdapter)


# ── classifier_forward tests ────────────────────────────────────────────────


def test_classifier_forward_runs_without_error():
    hidden_size, num_heads, num_layers = 8, 2, 1
    weights = _random_bert_weights(hidden_size, num_heads, num_layers)
    config = {
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "num_labels": 1,
        "layer_norm_eps": 1e-12,
    }
    batch_size, seq_len = 2, 4
    input_ids = mx.array([[1, 2, 3, 0], [4, 5, 6, 7]])
    attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 1, 1]])

    logits = classifier_forward(input_ids, attention_mask, weights, config)

    assert logits.shape == (batch_size, 1)
    assert logits.dtype == mx.float32


def test_mlx_classifier_wrapper_callable():
    weights = _random_bert_weights()
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_labels": 1,
        "layer_norm_eps": 1e-12,
    }
    wrapper = _MLXClassifierWrapper(config, weights, num_labels=1)
    input_ids = mx.array([[1, 2, 3, 0]])
    attention_mask = mx.array([[1, 1, 1, 0]])

    out = wrapper(input_ids, attention_mask=attention_mask)

    assert hasattr(out, "logits")
    assert out.logits.shape == (1, 1)


# ── RerankEngine tests ──────────────────────────────────────────────────────


def test_rerank_engine_score_pairs():
    """Test that RerankEngine.score_pairs returns correct-length scores."""
    weights = _random_bert_weights()
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_labels": 1,
        "layer_norm_eps": 1e-12,
    }
    model = _MLXClassifierWrapper(config, weights, num_labels=1)

    engine = RerankEngine("dummy-model")
    engine._tokenizer = MagicMock()
    engine._model = model
    engine._adapter = SigmoidAdapter()

    # Mock tokenizer to return fixed-size arrays
    def mock_tokenize_pair(tokenizer, query, document):
        return {
            "input_ids": [[1, 2, 3, 0]],
            "attention_mask": [[1, 1, 1, 0]],
        }

    engine._adapter.tokenize_pair = lambda tokenizer, q, d: mock_tokenize_pair(
        tokenizer, q, d
    )

    scores, total_tokens = engine.score_pairs("query", ["doc1", "doc2", "doc3"])

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert total_tokens > 0


# ── Server endpoint tests ───────────────────────────────────────────────────


@pytest.fixture
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def test_rerank_endpoint_returns_expected_structure(client):
    """Mock the entire engine to avoid needing a real tokenizer."""
    mock_engine = MagicMock()
    mock_engine.model_name = "demo-model"
    mock_engine.score_pairs.return_value = (
        [0.9, 0.5, 0.1],
        12,
    )

    mock_store = MagicMock()
    mock_store.get_rerank_engine.return_value = mock_engine
    with patch("xmlx_vlm.routes.admin.get_store", return_value=mock_store):
        response = client.post(
            "/v1/rerank",
            json={
                "model": "demo-model",
                "query": "machine learning",
                "documents": ["doc A", "doc B", "doc C"],
                "top_n": 2,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["model"] == "demo-model"
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    # Verify top 2 sorted by score descending
    scores = [r["relevance_score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_endpoint_creates_new_engine_for_different_model(client):
    """Verify that the endpoint instantiates a new engine when model changes."""
    mock_engine_a = MagicMock()
    mock_engine_a.model_name = "model-a"
    mock_engine_a.score_pairs.return_value = ([0.8], 4)

    mock_store = MagicMock()
    mock_store.get_rerank_engine.return_value = mock_engine_a
    with patch("xmlx_vlm.routes.admin.get_store", return_value=mock_store):
        # Request with same model
        response = client.post(
            "/v1/rerank",
            json={
                "model": "model-a",
                "query": "q",
                "documents": ["d"],
            },
        )
        assert response.status_code == 200
        mock_engine_a.score_pairs.assert_called_once()

    # Request with different model should hit the store for a new engine.
    mock_store = MagicMock()
    mock_store.get_rerank_engine.return_value = mock_engine_a
    with patch("xmlx_vlm.routes.admin.get_store", return_value=mock_store):
        response = client.post(
            "/v1/rerank",
            json={
                "model": "model-b",
                "query": "q",
                "documents": ["d"],
            },
        )
        assert response.status_code == 200
        mock_store.get_rerank_engine.assert_called_once_with("model-b")


# ── DistilBERT architecture tests ───────────────────────────────────────────


def test_classifier_forward_distilbert_runs_without_error():
    hidden_size, num_heads, num_layers = 8, 2, 1
    weights = _random_distilbert_weights(hidden_size, num_heads, num_layers)
    config = {
        "hidden_size": hidden_size,
        "n_heads": num_heads,
        "n_layers": num_layers,
        "num_labels": 1,
        "layer_norm_eps": 1e-12,
    }
    batch_size, seq_len = 2, 4
    input_ids = mx.array([[1, 2, 3, 0], [4, 5, 6, 7]])
    attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 1, 1]])

    logits = classifier_forward(input_ids, attention_mask, weights, config)

    assert logits.shape == (batch_size, 1)
    assert logits.dtype == mx.float32


def test_mlx_classifier_wrapper_with_distilbert():
    weights = _random_distilbert_weights()
    config = {
        "hidden_size": 8,
        "n_heads": 2,
        "n_layers": 1,
        "num_labels": 1,
        "layer_norm_eps": 1e-12,
    }
    wrapper = _MLXClassifierWrapper(config, weights, num_labels=1)
    input_ids = mx.array([[1, 2, 3, 0]])
    attention_mask = mx.array([[1, 1, 1, 0]])

    out = wrapper(input_ids, attention_mask=attention_mask)

    assert hasattr(out, "logits")
    assert out.logits.shape == (1, 1)


def test_rerank_engine_score_pairs_with_distilbert():
    """Test RerankEngine works with DistilBERT-style weights."""
    weights = _random_distilbert_weights()
    config = {
        "hidden_size": 8,
        "n_heads": 2,
        "n_layers": 1,
        "num_labels": 1,
        "layer_norm_eps": 1e-12,
    }
    model = _MLXClassifierWrapper(config, weights, num_labels=1)

    engine = RerankEngine("dummy-distilbert")
    engine._tokenizer = MagicMock()
    engine._model = model
    engine._adapter = SigmoidAdapter()

    def mock_tokenize_pair(tokenizer, query, document):
        return {
            "input_ids": [[1, 2, 3, 0]],
            "attention_mask": [[1, 1, 1, 0]],
        }

    engine._adapter.tokenize_pair = lambda tokenizer, q, d: mock_tokenize_pair(
        tokenizer, q, d
    )

    scores, total_tokens = engine.score_pairs("query", ["doc1", "doc2", "doc3"])

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert total_tokens > 0
