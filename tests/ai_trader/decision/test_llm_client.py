"""测试本地 LLM 客户端."""

import os
from unittest.mock import MagicMock, patch

import pytest

from xmlx_vlm.ai_trader.decision.llm_client import (
    AutoLLMClient,
    LocalServiceLLMClient,
    _resolve_server_url,
)


def _mock_response(json_data=None, status_code=200, text="", iter_lines=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.iter_lines.return_value = iter_lines or []
    resp.raise_for_status.return_value = None
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@pytest.mark.anyio
async def test_local_service_llm_client_probe_and_complete():
    client = LocalServiceLLMClient(server_url="http://localhost:9999", api_key="test")

    health_resp = _mock_response(json_data={"loaded_model": "test-model"})
    completion_lines = [
        b'data: {"choices": [{"delta": {"content": "[{\\"action\\": \\"wait\\"}]"}}]}',
        b"data: [DONE]",
    ]
    complete_resp = _mock_response(iter_lines=completion_lines)

    with patch("requests.get", return_value=health_resp) as mock_get:
        assert client.probe() is True
        mock_get.assert_called_once()

    with patch("requests.post", return_value=complete_resp) as mock_post:
        text = await client.complete("system", "user")
        assert "wait" in text
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"


@pytest.mark.anyio
async def test_local_service_llm_client_auth_header():
    client = LocalServiceLLMClient(server_url="http://localhost:9999", api_key="secret")
    health_resp = _mock_response(json_data={"loaded_model": "m"})
    complete_resp = _mock_response(iter_lines=[b"data: [DONE]"])

    with patch("requests.get", return_value=health_resp):
        client.probe()

    with patch("requests.post", return_value=complete_resp) as mock_post:
        await client.complete("s", "u")
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer secret"


def test_resolve_server_url_uses_env_port(monkeypatch):
    monkeypatch.setenv("XMLX_VLM_PORT", "9999")
    assert _resolve_server_url(None) == "http://localhost:9999"
    assert _resolve_server_url("http://127.0.0.1:8081") == "http://127.0.0.1:8081"


@pytest.mark.anyio
async def test_auto_llm_client_no_fallback_raises(monkeypatch):
    monkeypatch.setenv("XMLX_VLM_PORT", "19999")
    client = AutoLLMClient(allow_mlx_fallback=False)
    with pytest.raises(RuntimeError, match="本地推理服务未在"):
        await client.complete("system", "user")
