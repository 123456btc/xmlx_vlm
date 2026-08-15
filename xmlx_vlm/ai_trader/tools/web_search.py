from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# SSRF Blocked IP networks
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def robust_http_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    timeout: int = 10,
) -> Optional[requests.Response]:
    """
    Execute HTTP request with automatic direct fallback if local proxy throws ProxyError/SSLError.
    """
    req_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }
    if headers:
        req_headers.update(headers)

    # 1. Try with default environment configuration
    try:
        if method.upper() == "POST":
            return requests.post(url, headers=req_headers, params=params, json=json_data, data=data, timeout=timeout)
        else:
            return requests.get(url, headers=req_headers, params=params, timeout=timeout)
    except (requests.exceptions.ProxyError, requests.exceptions.SSLError, requests.exceptions.Timeout) as exc:
        logger.debug("Default env HTTP request failed for %s (%s). Retrying direct connection...", url, exc)

    # 2. Fallback: Force direct connection (bypassing broken system/local proxies)
    try:
        session = requests.Session()
        session.trust_env = False
        if method.upper() == "POST":
            return session.post(url, headers=req_headers, params=params, json=json_data, data=data, timeout=timeout)
        else:
            return session.get(url, headers=req_headers, params=params, timeout=timeout)
    except Exception as exc:
        logger.debug("Direct HTTP request also failed for %s: %s", url, exc)
        return None


def is_safe_url(url: str) -> bool:
    """
    Validate that the URL does not target localhost or private internal networks (SSRF protection).
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            return False

        try:
            ip_str = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip_str)
            for net in _PRIVATE_NETWORKS:
                if ip_obj in net:
                    return False
        except Exception:
            pass

        return True
    except Exception:
        return False


def normalize_query(query: str) -> str:
    """
    Clean conversational noise from the query to maximize search engine hit rates.
    """
    q = query.strip()
    pattern = r"^(帮我查一下|请帮我查一下|帮我搜索|请帮我搜索|帮我看看|请问一下|请问|帮我|查一下|搜索一下|查询|想知道|can you search for|can you search|please find|search for)\s*"
    while re.search(pattern, q, flags=re.I):
        q = re.sub(pattern, "", q, flags=re.I).strip()
    q = re.sub(r"[？?！!。.\s]+$", "", q).strip()
    return q if q else query.strip()


def distill_excerpts(text: str, query: Optional[str] = None, max_chars: int = 3500) -> str:
    """
    Distill and extract the most relevant paragraphs from raw webpage text to preserve LLM token context.
    """
    if len(text) <= max_chars:
        return text

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return text[:max_chars]

    if not query:
        result = []
        current_len = 0
        for p in paragraphs:
            if current_len + len(p) + 2 > max_chars:
                break
            result.append(p)
            current_len += len(p) + 2
        return "\n\n".join(result) if result else text[:max_chars]

    terms = set(re.findall(r"\w+", query.lower()))
    scored_paragraphs: List[tuple[float, int, str]] = []

    for idx, p in enumerate(paragraphs):
        p_lower = p.lower()
        score = 0.0
        for term in terms:
            if len(term) >= 2:
                count = p_lower.count(term)
                score += count * (1.5 if len(term) > 3 else 1.0)
        if p.startswith("#") or p.startswith("**"):
            score += 1.0
        if idx < 3:
            score += 0.5
        scored_paragraphs.append((score, idx, p))

    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    
    selected_indices = set()
    current_len = 0
    for score, idx, p in scored_paragraphs:
        if current_len + len(p) + 2 <= max_chars:
            selected_indices.add(idx)
            current_len += len(p) + 2
        if len(selected_indices) >= 6 or current_len >= max_chars * 0.9:
            break

    final_paragraphs = [paragraphs[i] for i in sorted(selected_indices)]
    if not final_paragraphs:
        return text[:max_chars]

    distilled = "\n\n".join(final_paragraphs)
    if len(distilled) < len(text):
        distilled += "\n\n[... Remaining content condensed ...]"
    return distilled


def _search_baidu_native(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    High-reliability Baidu web search engine parser (100% direct, zero API key).
    """
    resp = robust_http_request("GET", "https://www.baidu.com/s", params={"wd": query}, timeout=8)
    if not resp or resp.status_code != 200:
        return []

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for div in soup.find_all("div", class_=re.compile(r"result|c-container")):
            h3 = div.find("h3")
            if not h3:
                continue
            a = h3.find("a")
            if not a:
                continue
            title = a.get_text().strip()
            url = a.get("href", "")
            
            # Find snippet
            p = div.find("span", class_=re.compile(r"content-right_")) or div.find("div", class_=re.compile(r"c-abstract")) or div.find("p")
            desc = p.get_text().strip() if p else ""

            if title and url and not any(r["url"] == url for r in results):
                results.append({
                    "title": title,
                    "url": url,
                    "description": desc[:300],
                    "position": len(results) + 1,
                })
            if len(results) >= limit:
                break
        return results
    except Exception as e:
        logger.debug("Baidu search parse error: %s", e)
        return []


def _search_ddg_html_native(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Pure Python zero-dependency DuckDuckGo HTML parser.
    """
    url = "https://html.duckduckgo.com/html/"
    resp = robust_http_request("POST", url, data={"q": query}, headers={"Referer": "https://html.duckduckgo.com/"}, timeout=8)
    if not resp or resp.status_code != 200:
        return []

    html = resp.text
    results = []
    titles = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
    snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
    for i in range(min(len(titles), len(snippets), limit)):
        u, t = titles[i]
        s = snippets[i]
        clean_url = u
        if "uddg=" in u:
            m = re.search(r"uddg=([^&]+)", u)
            if m:
                clean_url = urllib.parse.unquote(m.group(1))
        
        clean_title = re.sub(r"<[^>]+>", "", t).strip()
        clean_snippet = re.sub(r"<[^>]+>", "", s).strip()
        if clean_url.startswith("http"):
            results.append({
                "title": clean_title,
                "url": clean_url,
                "description": clean_snippet,
                "position": len(results) + 1,
            })
    return results


class WebSearchTool:
    """
    Multi-Provider Online Web Search Tool.
    Supports Tavily, Brave Search, Jina Search (No-Key Fast Path), Baidu Direct, and Native DuckDuckGo.
    """

    name = "web_search"
    description = "Search the web for real-time information (news, crypto metrics, market events, general knowledge). Returns structured titles, URLs, and summaries."
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
        raw_query = kwargs.get("query")
        if not raw_query:
            return "错误：必须提供 query 参数"
        limit = max(1, min(int(kwargs.get("limit", 5)), 15))
        query = normalize_query(raw_query)

        # 1. Tier 1: Tavily Search (if key provided)
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if tavily_key:
            try:
                base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").rstrip("/")
                payload = {
                    "api_key": tavily_key,
                    "query": query,
                    "max_results": limit,
                    "include_raw_content": False,
                    "include_images": False,
                }
                resp = robust_http_request("POST", f"{base_url}/search", json_data=payload, timeout=10)
                if resp and resp.status_code == 200:
                    data = resp.json()
                    web_results = [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "description": r.get("content", ""),
                            "position": i + 1,
                        }
                        for i, r in enumerate(data.get("results", []))
                    ]
                    if web_results:
                        return json.dumps({"success": True, "provider": "tavily", "data": {"web": web_results}}, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.warning("Tavily search failed: %s. Falling back.", exc)

        # 2. Tier 2: Brave Search (if key provided)
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if brave_key:
            try:
                endpoint = "https://api.search.brave.com/res/v1/web/search"
                headers = {"X-Subscription-Token": brave_key, "Accept": "application/json"}
                resp = robust_http_request("GET", endpoint, params={"q": query, "count": limit}, headers=headers, timeout=10)
                if resp and resp.status_code == 200:
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
                    if web_results:
                        return json.dumps({"success": True, "provider": "brave", "data": {"web": web_results}}, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.warning("Brave search failed: %s. Falling back.", exc)

        # 3. Tier 3: Jina AI Search (No-Key Fast Path)
        try:
            encoded_query = urllib.parse.quote(query)
            jina_url = f"https://s.jina.ai/{encoded_query}"
            resp = robust_http_request("GET", jina_url, headers={"Accept": "application/json", "X-No-Cache": "true"}, timeout=10)
            if resp and resp.status_code == 200:
                try:
                    data = resp.json()
                    results = data.get("data", [])
                    web_results = []
                    for i, item in enumerate(results[:limit]):
                        web_results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("description") or item.get("content", "")[:300],
                            "position": i + 1,
                        })
                    if web_results:
                        return json.dumps({"success": True, "provider": "jina", "data": {"web": web_results}}, ensure_ascii=False, indent=2)
                except Exception:
                    text_resp = resp.text
                    items = re.findall(r"\[\d+\]\s+Title:\s+(.*?)\nURL Source:\s+(.*?)\nMarkdown Content:\s+(.*?)(?=\n\[\d+\]\s+Title:|$)", text_resp, re.DOTALL)
                    if items:
                        web_results = []
                        for i, (t, u, c) in enumerate(items[:limit]):
                            web_results.append({
                                "title": t.strip(),
                                "url": u.strip(),
                                "description": c.strip()[:300],
                                "position": i + 1,
                            })
                        return json.dumps({"success": True, "provider": "jina_md", "data": {"web": web_results}}, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("Jina search fallback triggered: %s", exc)

        # 4. Tier 4: Direct Baidu Search (Instant connection for Chinese / Global market topics)
        try:
            baidu_results = _search_baidu_native(query, limit=limit)
            if baidu_results:
                return json.dumps({"success": True, "provider": "baidu_native", "data": {"web": baidu_results}}, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("Baidu search fallback triggered: %s", exc)

        # 5. Tier 5: Native DuckDuckGo Zero-Dependency Parser
        try:
            ddg_results = _search_ddg_html_native(query, limit=limit)
            if ddg_results:
                return json.dumps({"success": True, "provider": "duckduckgo_native", "data": {"web": ddg_results}}, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("Native DuckDuckGo failed: %s", exc)

        return json.dumps({
            "success": False,
            "error": f"All search providers failed for query '{query}'. Check network connection or configure TAVILY_API_KEY / BRAVE_SEARCH_API_KEY."
        }, ensure_ascii=False)


class WebExtractTool:
    """
    High-Fidelity Webpage Content Extraction Tool.
    Uses Jina Reader, Tavily Extract, and local BeautifulSoup with intelligent chunk distillation and SSRF protection.
    """

    name = "web_extract"
    description = "Extract and clean the main content from a webpage URL. Automatically eliminates navigation and ads, returning structured Markdown."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The webpage URL to extract content from.",
            },
            "query": {
                "type": "string",
                "description": "Optional search query to focus and rank relevant excerpts from long pages.",
            },
        },
        "required": ["url"],
    }

    def run(self, **kwargs) -> str:
        url = kwargs.get("url")
        if not url:
            return "错误：必须提供 url 参数"
        
        query = kwargs.get("query")

        # 0. SSRF Safety Check
        if not is_safe_url(url):
            return json.dumps({
                "success": False,
                "error": f"URL '{url}' is blocked for security reasons (private/internal network access prohibited)."
            }, ensure_ascii=False)

        # 1. Tier 1: Jina Reader (https://r.jina.ai/<url> - No-Key Fast Path & clean Markdown)
        try:
            reader_url = f"https://r.jina.ai/{url}"
            resp = robust_http_request("GET", reader_url, headers={"Accept": "text/markdown", "X-No-Cache": "true"}, timeout=12)
            if resp and resp.status_code == 200 and len(resp.text.strip()) > 100:
                raw_md = resp.text.strip()
                if not (raw_md.lower().startswith("<!doctype") or raw_md.lower().startswith("<html")):
                    title_match = re.search(r"^Title:\s*(.*)$", raw_md, re.M)
                    title = title_match.group(1).strip() if title_match else url
                    
                    content = distill_excerpts(raw_md, query=query, max_chars=3500)
                    return json.dumps({
                        "success": True,
                        "provider": "jina_reader",
                        "data": {
                            "url": url,
                            "title": title,
                            "content": content,
                        }
                    }, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("Jina Reader extract failed: %s. Falling back.", exc)

        # 2. Tier 2: Tavily Extract (if key provided)
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if tavily_key:
            try:
                base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").rstrip("/")
                payload = {"api_key": tavily_key, "urls": [url], "include_images": False}
                resp = robust_http_request("POST", f"{base_url}/extract", json_data=payload, timeout=12)
                if resp and resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        raw_content = results[0].get("raw_content") or results[0].get("content") or ""
                        title = results[0].get("title", url)
                        content = distill_excerpts(raw_content, query=query, max_chars=3500)
                        return json.dumps({
                            "success": True,
                            "provider": "tavily_extract",
                            "data": {
                                "url": url,
                                "title": title,
                                "content": content,
                            }
                        }, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.debug("Tavily extract failed: %s. Falling back to scraper.", exc)

        # 3. Tier 3: Local Scraper with BeautifulSoup
        try:
            resp = robust_http_request("GET", url, timeout=10)
            if not resp or resp.status_code != 200:
                raise RuntimeError(f"HTTP request returned status {resp.status_code if resp else 'No response'}")
            html = resp.text

            title = ""
            content = ""

            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")

                title_tag = soup.find("title")
                title = title_tag.get_text().strip() if title_tag else url

                for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
                    element.decompose()

                content = soup.get_text(separator="\n\n")
            except ImportError:
                title_match = re.search(r"<title>(.*?)</title>", html, re.I)
                title = title_match.group(1).strip() if title_match else url
                cleaned = re.sub(r"<(script|style|nav|footer|header|aside|noscript|iframe)\b[^>]*>([\s\S]*?)<\/\1>", "", html, flags=re.I)
                content = re.sub(r"<[^>]+>", " ", cleaned)

            lines = [line.strip() for line in content.splitlines() if line.strip()]
            cleaned_text = "\n\n".join(lines)
            final_content = distill_excerpts(cleaned_text, query=query, max_chars=3500)

            return json.dumps({
                "success": True,
                "provider": "local_scraper",
                "data": {
                    "url": url,
                    "title": title,
                    "content": final_content,
                }
            }, ensure_ascii=False, indent=2)

        except Exception as exc:
            return json.dumps({
                "success": False,
                "error": f"Failed to extract content from '{url}': {exc}"
            }, ensure_ascii=False)
