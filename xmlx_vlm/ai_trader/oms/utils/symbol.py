"""Symbol SSOT: 统一交易标的格式化与解析工具.

定义系统的唯一数据源规范（SSOT）：
1. Canonical Symbol (内部标准交易对): {Base}/{Quote}，例如 BTC/USDC, kSHIB/USDC
2. Base Coin (基础币种 / Wire 币种): 例如 BTC, kSHIB, kBONK (严格保留小写 k 前缀)
3. Quote Coin (计价币种): 默认 USDC
"""

from __future__ import annotations

import re
from typing import Any, Tuple


# 已知 Hyperliquid 上的千倍/万倍聚合前缀币种 (保留小写 'k' 前缀)
_KNOWN_K_PREFIX_COINS = {
    "KSHIB": "kSHIB",
    "KBONK": "kBONK",
    "KLUNC": "kLUNC",
    "KFLOKI": "kFLOKI",
    "KPEPE": "kPEPE",
    "KNEIRO": "kNEIRO",
}

# 常见报价币种后缀
_KNOWN_QUOTES = ("USDC", "USDT", "USD", "BTC", "ETH", "EUR", "GBP", "JPY")


def _format_base_coin(raw_base: str) -> str:
    """标准化 Base 币种名称，智能处理 k-prefix 标的."""
    b = raw_base.strip()
    if not b:
        return ""

    upper = b.upper()
    # 检查是否为已知 k-prefix 币种
    if upper in _KNOWN_K_PREFIX_COINS:
        return _KNOWN_K_PREFIX_COINS[upper]

    # 通用规则：如果以 k/K 开头且后面跟 3 个以上大写字母（如 kDOGE, KSHIB），保留小写 k
    if len(b) > 2 and (b.startswith("k") or b.startswith("K")) and b[1:].isupper():
        return f"k{b[1:].upper()}"

    # 默认转全大写
    return upper


def parse_symbol_parts(symbol: Any, default_quote: str = "USDC") -> Tuple[str, str]:
    """解析 symbol 为 (base_coin, quote_coin).
    
    支持格式:
    - 'BTC/USDC' -> ('BTC', 'USDC')
    - 'BTC-USDC' -> ('BTC', 'USDC')
    - 'BTC_USDC' -> ('BTC', 'USDC')
    - 'kSHIB/USDC' -> ('kSHIB', 'USDC')
    - 'kshibusdc' -> ('kSHIB', 'USDC')
    - 'BTC' -> ('BTC', 'USDC')
    - 'kSHIB' -> ('kSHIB', 'USDC')
    """
    if not symbol:
        raise ValueError("symbol cannot be empty")

    s = str(symbol).strip()
    if not s:
        raise ValueError("symbol cannot be empty")

    # 1. 检查常见分隔符 (/, -, _)
    for sep in ("/", "-", "_"):
        if sep in s:
            parts = s.split(sep, 1)
            raw_base = parts[0].strip()
            raw_quote = parts[1].strip() or default_quote
            base = _format_base_coin(raw_base)
            quote = raw_quote.upper()
            return base, quote

    # 2. 检查末尾是否含有已知 quote 后缀
    upper_s = s.upper()
    for quote_candidate in _KNOWN_QUOTES:
        if upper_s.endswith(quote_candidate) and len(upper_s) > len(quote_candidate):
            raw_base = s[: -len(quote_candidate)]
            base = _format_base_coin(raw_base)
            return base, quote_candidate

    # 3. 裸 coin，使用默认 quote
    base = _format_base_coin(s)
    return base, default_quote.upper()


def normalize_symbol(symbol: Any, default_quote: str = "USDC") -> str:
    """标准化为内部 Canonical 交易对格式: {Base}/{Quote}.
    
    例如:
    - 'BTC' -> 'BTC/USDC'
    - 'btc/usdc' -> 'BTC/USDC'
    - 'kSHIB' -> 'kSHIB/USDC'
    - 'KSHIB/USDC' -> 'kSHIB/USDC'
    - 'kshibusdt' -> 'kSHIB/USDT'
    """
    base, quote = parse_symbol_parts(symbol, default_quote=default_quote)
    return f"{base}/{quote}"


def extract_base_coin(symbol: Any) -> str:
    """从任意格式的交易对或币种名称中提取标准 Base 币种.
    
    例如:
    - 'BTC/USDC' -> 'BTC'
    - 'kSHIB/USDC' -> 'kSHIB'
    - 'KSHIB/USDC' -> 'kSHIB'
    - 'kshib' -> 'kSHIB'
    - 'BTCUSDT' -> 'BTC'
    """
    base, _ = parse_symbol_parts(symbol)
    return base


def symbol_matches(sym1: Any, sym2: Any) -> bool:
    """判断两个 symbol 是否代表同一个标的.
    
    支持:
    - 'BTC' vs 'BTC/USDC' -> True
    - 'kSHIB' vs 'KSHIB/USDC' -> True
    - 'BTC/USDC' vs 'BTC/USDT' -> False (不同 quote)
    - 'BTC' vs 'BTC' -> True
    """
    if not sym1 or not sym2:
        return False
    try:
        b1, q1 = parse_symbol_parts(sym1)
        b2, q2 = parse_symbol_parts(sym2)
        s1_raw = str(sym1).strip()
        s2_raw = str(sym2).strip()
        has_quote1 = any(sep in s1_raw for sep in ("/", "-", "_")) or any(s1_raw.upper().endswith(q) for q in _KNOWN_QUOTES)
        has_quote2 = any(sep in s2_raw for sep in ("/", "-", "_")) or any(s2_raw.upper().endswith(q) for q in _KNOWN_QUOTES)
        
        if b1 != b2:
            return False
        if has_quote1 and has_quote2:
            return q1 == q2
        return True
    except Exception:
        return False
