"""参数校验工具."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


def validate_symbol(symbol: Any) -> str:
    """校验并标准化交易对格式."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    return symbol.strip().upper()


def validate_positive(value: Any, name: str = "value") -> Decimal:
    """校验数值为正."""
    d = to_decimal(value)
    if d <= ZERO:
        raise ValueError(f"{name} must be positive, got {value}")
    return d


def validate_non_negative(value: Any, name: str = "value") -> Decimal:
    """校验数值非负."""
    d = to_decimal(value)
    if d < ZERO:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return d
