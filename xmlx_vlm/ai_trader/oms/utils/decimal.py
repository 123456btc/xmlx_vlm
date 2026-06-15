"""Decimal 计算辅助，避免浮点误差."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any


ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")


def to_decimal(value: Any, default: Decimal = ZERO) -> Decimal:
    """把任意值转为 Decimal，失败返回 default."""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (TypeError, ValueError, ArithmeticError):
        return default


def quantize_price(value: Decimal, decimals: int = 2) -> Decimal:
    """按价格精度四舍五入."""
    quant = Decimal("1").scaleb(-decimals)
    return value.quantize(quant, rounding=ROUND_HALF_UP)


def quantize_qty(value: Decimal, decimals: int = 6) -> Decimal:
    """按数量精度四舍五入."""
    quant = Decimal("1").scaleb(-decimals)
    return value.quantize(quant, rounding=ROUND_HALF_UP)


def pct_change(current: Decimal, base: Decimal) -> Decimal:
    """百分比变化，base 为 0 返回 0."""
    if base == ZERO:
        return ZERO
    return (current - base) / base * HUNDRED
