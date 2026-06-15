from __future__ import annotations

import os
import logging
import json
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class WebSearchTool:
    """网页检索工具 —— 支持 Tavily, Brave Search 和 DuckDuckGo 检索."""

    name = "web_search"
    description = "Search the web for real-time information (news, prices, market sentiment). Returns a list of search results with titles, URLs, and descriptions."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def run(self, **kwargs) -> str:
        query = kwargs.get("query")
        if not query:
            return "错误：必须提供 query 参数"
        limit = int(kwargs.get("limit", 5))

        # 1. Try Tavily Search
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if tavily_key:
            try:
                import requests
                base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").rstrip("/")
                payload = {
                    "api_key": tavily_key,
                    "query": query,
                    "max_results": min(limit, 20),
                    "include_raw_content": False,
                    "include_images": False,
                }
                resp = requests.post(f"{base_url}/search", json=payload, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                web_results = []
                for i, r in enumerate(data.get("results", [])):
                    web_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "description": r.get("content", ""),
                        "position": i + 1
                    })
                return json.dumps({"success": True, "data": {"web": web_results}}, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("Tavily search failed: %s. Falling back.", e)

        # 2. Try Brave Search
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if brave_key:
            try:
                import requests
                endpoint = "https://api.search.brave.com/res/v1/web/search"
                headers = {
                    "X-Subscription-Token": brave_key,
                    "Accept": "application/json",
                }
                resp = requests.get(endpoint, params={"q": query, "count": min(limit, 20)}, headers=headers, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                raw_results = (data.get("web") or {}).get("results", []) or []
                web_results = [
                    {
                        "title": str(r.get("title", "")),
                        "url": str(r.get("url", "")),
                        "description": str(r.get("description", "")),
                        "position": i + 1,
                    }
                    for i, r in enumerate(raw_results[:limit])
                ]
                return json.dumps({"success": True, "data": {"web": web_results}}, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("Brave search failed: %s. Falling back.", e)

        # 3. Try DuckDuckGo
        try:
            from duckduckgo_search import DDGS
            web_results = []
            with DDGS() as client:
                hits = list(client.text(query, max_results=limit))
                for i, hit in enumerate(hits):
                    url = hit.get("href") or hit.get("url") or ""
                    web_results.append({
                        "title": hit.get("title", ""),
                        "url": url,
                        "description": hit.get("body") or hit.get("description") or "",
                        "position": i + 1
                    })
            return json.dumps({"success": True, "data": {"web": web_results}}, ensure_ascii=False, indent=2)
        except ImportError:
            return json.dumps({
                "success": False,
                "error": "未配置 TAVILY_API_KEY / BRAVE_SEARCH_API_KEY 且本地未安装 duckduckgo-search。请运行 `pip install duckduckgo-search` 或配置 API 密钥。"
            }, ensure_ascii=False)
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return json.dumps({"success": False, "error": f"DuckDuckGo search failed: {e}"}, ensure_ascii=False)


class WebExtractTool:
    """网页提取工具 —— 提取网页正文为纯文本/Markdown."""

    name = "web_extract"
    description = "Extract main text content from a web page URL. Returns clean text representation of the webpage."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The webpage URL to extract content from.",
            }
        },
        "required": ["url"],
    }

    def run(self, **kwargs) -> str:
        url = kwargs.get("url")
        if not url:
            return "错误：必须提供 url 参数"

        # 1. Try Tavily Extract
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if tavily_key:
            try:
                import requests
                base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").rstrip("/")
                payload = {
                    "api_key": tavily_key,
                    "urls": [url],
                    "include_images": False,
                }
                resp = requests.post(f"{base_url}/extract", json=payload, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if results:
                    content = results[0].get("raw_content") or results[0].get("content") or ""
                    title = results[0].get("title", "")
                    return json.dumps({
                        "success": True,
                        "data": {
                            "url": url,
                            "title": title,
                            "content": content
                        }
                    }, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("Tavily extract failed: %s. Falling back to scraper.", e)

        # 2. Scraper Fallback
        try:
            import requests
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            html = resp.text

            # Try BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                for s in soup(["script", "style", "noscript", "iframe"]):
                    s.decompose()
                title = soup.title.string.strip() if soup.title else ""
                # Get structured body text
                content = soup.get_text(separator="\n")
            except ImportError:
                # Regex fallback
                title_match = re.search(r"<title>(.*?)</title>", html, re.I)
                title = title_match.group(1).strip() if title_match else ""
                content = re.sub(r'<(script|style|noscript|iframe)\b[^>]*>([\s\S]*?)<\/\1>', '', html, flags=re.I)
                content = re.sub(r'<[^>]+>', ' ', content)
            
            # Post-process content whitespace
            lines = [line.strip() for line in content.splitlines()]
            chunks = [phrase.strip() for line in lines for phrase in line.split("  ")]
            content = "\n".join(chunk for chunk in chunks if chunk)

            # Cap content length to avoid token overflow (e.g. 8000 chars)
            if len(content) > 8000:
                content = content[:8000] + "\n\n[Content truncated...]"

            return json.dumps({
                "success": True,
                "data": {
                    "url": url,
                    "title": title,
                    "content": content
                }
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Scraper extract failed: {e}"
            }, ensure_ascii=False)
