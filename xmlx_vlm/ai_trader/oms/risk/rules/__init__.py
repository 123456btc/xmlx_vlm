"""风控规则集合."""

from xmlx_vlm.ai_trader.oms.risk.rules.base import RiskRule
from xmlx_vlm.ai_trader.oms.risk.rules.daily_loss_rule import DailyLossRule
from xmlx_vlm.ai_trader.oms.risk.rules.position_limit_rule import PositionLimitRule
from xmlx_vlm.ai_trader.oms.risk.rules.order_size_rule import OrderSizeRule
from xmlx_vlm.ai_trader.oms.risk.rules.price_deviation_rule import PriceDeviationRule
from xmlx_vlm.ai_trader.oms.risk.rules.rate_limit_rule import RateLimitRule
from xmlx_vlm.ai_trader.oms.risk.rules.margin_rule import MarginRule

__all__ = [
    "RiskRule",
    "DailyLossRule",
    "PositionLimitRule",
    "OrderSizeRule",
    "PriceDeviationRule",
    "RateLimitRule",
    "MarginRule",
]
