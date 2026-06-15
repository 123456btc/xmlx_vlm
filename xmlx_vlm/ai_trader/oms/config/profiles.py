"""风控配置模板."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict


RISK_PROFILES: Dict[str, Dict[str, object]] = {
    "conservative": {
        "max_daily_loss_pct": Decimal("1.0"),
        "max_single_position_pct": Decimal("10.0"),
        "max_total_position_pct": Decimal("20.0"),
        "max_single_order_notional": Decimal("1000"),
        "min_order_notional": Decimal("10"),
        "max_price_deviation_pct": Decimal("0.5"),
        "max_orders_per_minute": 6,
        "max_orders_per_second": 2,
        "min_available_margin_pct": Decimal("40.0"),
        "max_slippage_pct": Decimal("0.3"),
    },
    "moderate": {
        "max_daily_loss_pct": Decimal("3.0"),
        "max_single_position_pct": Decimal("20.0"),
        "max_total_position_pct": Decimal("40.0"),
        "max_single_order_notional": Decimal("5000"),
        "min_order_notional": Decimal("10"),
        "max_price_deviation_pct": Decimal("1.0"),
        "max_orders_per_minute": 12,
        "max_orders_per_second": 3,
        "min_available_margin_pct": Decimal("20.0"),
        "max_slippage_pct": Decimal("0.5"),
    },
    "aggressive": {
        "max_daily_loss_pct": Decimal("5.0"),
        "max_single_position_pct": Decimal("35.0"),
        "max_total_position_pct": Decimal("70.0"),
        "max_single_order_notional": Decimal("20000"),
        "min_order_notional": Decimal("10"),
        "max_price_deviation_pct": Decimal("2.0"),
        "max_orders_per_minute": 30,
        "max_orders_per_second": 5,
        "min_available_margin_pct": Decimal("10.0"),
        "max_slippage_pct": Decimal("1.0"),
    },
}
