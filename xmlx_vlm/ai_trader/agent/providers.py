"""MarketDataService 价格与 ATR 提供者."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal

if TYPE_CHECKING:
    from xmlx_vlm.ai_trader.market_service.service import MarketDataService

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """基于 MarketDataService 的同步价格/ATR 查询.

    MarketDataService 运行在独立线程，其内部状态机使用 RLock，
    因此同步调用 get_summary 是线程安全的。
    """

    def __init__(self, service: "MarketDataService") -> None:
        self.service = service

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """获取最新 mark price."""
        try:
            summary = self.service.get_summary(symbol)
            if summary is None:
                return None
            return to_decimal(summary.mark_price)
        except Exception:
            logger.exception("Failed to get price for %s", symbol)
            return None

    def get_atr(self, symbol: str) -> Optional[Decimal]:
        """获取 1h ATR(14)."""
        try:
            summary = self.service.get_summary(symbol)
            if summary is None or summary.atr14 is None:
                return None
            return to_decimal(summary.atr14)
        except Exception:
            logger.exception("Failed to get ATR for %s", symbol)
            return None
