# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for enhanced web search and extraction pipeline.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from xmlx_vlm.ai_trader.tools.web_search import (
    WebExtractTool,
    WebSearchTool,
    distill_excerpts,
    is_safe_url,
    normalize_query,
)


# ─── 1. SSRF Safety Protection Tests ──────────────────────────────────────────

def test_ssrf_safety_url_filtering():
    # Block internal IPs and localhost
    assert is_safe_url("http://localhost:5118/v1/chat") is False
    assert is_safe_url("http://127.0.0.1:8080") is False
    assert is_safe_url("http://10.0.0.5:9000") is False
    assert is_safe_url("http://192.168.1.100/admin") is False
    assert is_safe_url("http://169.254.169.254/latest/meta-data") is False
    assert is_safe_url("ftp://example.com/file") is False
    assert is_safe_url("javascript:alert(1)") is False

    # Allow public URLs
    assert is_safe_url("https://www.coindesk.com/markets") is True
    assert is_safe_url("https://github.com/123456btc/xmlx_vlm") is True


# ─── 2. Query Normalization Tests ──────────────────────────────────────────────

def test_query_normalization():
    assert normalize_query("帮我查一下以太坊ETF流入情况？") == "以太坊ETF流入情况"
    assert normalize_query("请问 2026 年比特币减半时间！") == "2026 年比特币减半时间"
    assert normalize_query("please find latest crypto market sentiment") == "latest crypto market sentiment"
    assert normalize_query("Bitcoin Price") == "Bitcoin Price"


# ─── 3. Chunk Distillation Tests ───────────────────────────────────────────────

def test_chunk_distillation():
    raw_doc = (
        "Welcome to the homepage.\n\n"
        "Cookie policy and navigation header.\n\n"
        "Ethereum ETF recorded a massive net inflow of $500M today led by BlackRock and Fidelity.\n\n"
        "The total cryptocurrency market capitalization rose by 4.2%.\n\n"
        "Terms of service and footer copyright 2026."
    )
    
    # Distill with query focus
    distilled = distill_excerpts(raw_doc, query="Ethereum ETF inflow", max_chars=300)
    assert "Ethereum ETF recorded a massive net inflow" in distilled
    assert len(distilled) <= 300


# ─── 4. WebExtractTool & BeautifulSoup Bug Fix Verification ───────────────────

def test_web_extract_local_scraper():
    tool = WebExtractTool()
    
    html_content = """
    <html>
      <head><title>Crypto News Daily</title></head>
      <body>
        <nav><a href="/home">Home</a></nav>
        <h1>Bitcoin Crosses Resistance</h1>
        <p>Bitcoin has broken through the key 65,000 USD level with high volume.</p>
        <script>alert("tracker");</script>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html_content
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        res_json = tool.run(url="https://example.com/crypto-news")
        data = json.loads(res_json)
        assert data["success"] is True
        assert "Bitcoin Crosses Resistance" in data["data"]["content"]
        assert "alert(" not in data["data"]["content"]  # Script tag removed
        assert "Copyright" not in data["data"]["content"]  # Footer tag removed


# ─── 5. WebSearchTool Mock Fallback Flow ───────────────────────────────────────

def test_web_search_jina_fallback():
    tool = WebSearchTool()

    mock_jina_resp = MagicMock()
    mock_jina_resp.status_code = 200
    mock_jina_resp.json.return_value = {
        "data": [
            {
                "title": "Hyperliquid DEX Volume Hits Record",
                "url": "https://example.com/hl-volume",
                "description": "Hyperliquid 24h volume exceeded $5 Billion in perpetual contracts.",
            }
        ]
    }

    with patch("requests.get", return_value=mock_jina_resp):
        res_json = tool.run(query="Hyperliquid volume")
        data = json.loads(res_json)
        assert data["success"] is True
        assert data["provider"] == "jina"
        assert len(data["data"]["web"]) == 1
        assert "Hyperliquid DEX Volume" in data["data"]["web"][0]["title"]
