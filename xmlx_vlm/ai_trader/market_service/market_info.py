"""市场信息查询.

负责从 Hyperliquid REST API 获取币种元数据、成交量排名等静态/准静态信息。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

_HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def fetch_meta_and_ctxs(api_url: str = _HYPERLIQUID_API) -> tuple[dict, list]:
    """获取 Hyperliquid 的 meta 和所有资产上下文."""
    resp = requests.post(api_url, json={"type": "metaAndAssetCtxs"}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data[0], data[1]


def fetch_top_volume_coins(
    n: int = 30, api_url: str = _HYPERLIQUID_API
) -> List[str]:
    """按 24h 名义成交额（dayNtlVlm）排序，返回前 N 个币种代码."""
    try:
        meta, ctxs = fetch_meta_and_ctxs(api_url)
    except Exception as exc:
        logger.warning("Failed to fetch top volume coins: %s", exc)
        return []

    universe = meta.get("universe", [])
    if len(universe) != len(ctxs):
        logger.warning(
            "Universe size %d != ctxs size %d", len(universe), len(ctxs)
        )

    volumes: List[tuple[str, float]] = []
    for asset, ctx in zip(universe, ctxs):
        coin = asset.get("name")
        if not coin:
            continue
        # 只保留永续合约；Hyperliquid 现货/其他资产名称通常不是纯大写币对，也可以按 name 过滤
        day_vlm = _to_float(ctx.get("dayNtlVlm"))
        volumes.append((coin, day_vlm))

    volumes.sort(key=lambda x: x[1], reverse=True)
    top = [coin for coin, _ in volumes[:n]]
    logger.info("Top %d volume coins: %s", len(top), top)
    return top
