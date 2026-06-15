"""市场冲击与滑点模型.

基于 Almgren-Chriss 框架的简化实现：
- temporary_impact：与订单量占 ADV 比例、spread 相关
- permanent_impact：与订单量占 ADV 比例、波动率相关
- expected_slippage：temporary + permanent + spread 综合
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, HUNDRED


@dataclass
class ImpactEstimate:
    """冲击估算结果."""

    temporary_impact_pct: Decimal = ZERO
    permanent_impact_pct: Decimal = ZERO
    expected_slippage_pct: Decimal = ZERO
    expected_slippage_abs: Decimal = ZERO
    confidence: str = "low"  # high / medium / low
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.temporary_impact_pct = to_decimal(self.temporary_impact_pct)
        self.permanent_impact_pct = to_decimal(self.permanent_impact_pct)
        self.expected_slippage_pct = to_decimal(self.expected_slippage_pct)
        self.expected_slippage_abs = to_decimal(self.expected_slippage_abs)


class MarketImpactModel:
    """市场冲击模型抽象基类."""

    def estimate(
        self,
        order_qty: Decimal,
        side: OrderSide,
        price: Decimal,
        adv: Optional[Decimal] = None,
        spread_pct: Optional[Decimal] = None,
        volatility: Optional[Decimal] = None,
        urgency: str = "normal",
        **kwargs: Any,
    ) -> ImpactEstimate:
        """估算单次订单的预期冲击."""
        raise NotImplementedError


class AlmgrenChrissImpactModel(MarketImpactModel):
    """Almgren-Chriss 风格冲击模型.

    参数说明：
    - eta: 临时冲击系数（对 ADV 的弹性）
    - gamma: 永久冲击系数
    - spread_weight: spread 在预期滑点中的权重
    - urgency_multiplier:  urgency 对临时冲击的放大/缩小
    """

    def __init__(
        self,
        eta: Decimal = Decimal("0.1"),
        gamma: Decimal = Decimal("0.05"),
        spread_weight: Decimal = Decimal("0.5"),
        min_slippage_pct: Decimal = Decimal("0.01"),
    ):
        self.eta = to_decimal(eta)
        self.gamma = to_decimal(gamma)
        self.spread_weight = to_decimal(spread_weight)
        self.min_slippage_pct = to_decimal(min_slippage_pct)

    def estimate(
        self,
        order_qty: Decimal,
        side: OrderSide,
        price: Decimal,
        adv: Optional[Decimal] = None,
        spread_pct: Optional[Decimal] = None,
        volatility: Optional[Decimal] = None,
        urgency: str = "normal",
        **kwargs: Any,
    ) -> ImpactEstimate:
        order_qty = abs(to_decimal(order_qty))
        price = to_decimal(price)
        if price <= ZERO:
            return ImpactEstimate(confidence="low")

        adv = to_decimal(adv) if adv is not None else ZERO
        spread_pct = to_decimal(spread_pct) if spread_pct is not None else ZERO
        volatility = to_decimal(volatility) if volatility is not None else ZERO

        confidence = "high" if adv > ZERO and spread_pct > ZERO and volatility > ZERO else "medium"
        if adv <= ZERO or spread_pct <= ZERO:
            confidence = "low"

        participation = (order_qty / adv) if adv > ZERO else ZERO
        urgency_mult = self._urgency_multiplier(urgency)

        # 临时冲击：与参与率、spread 正相关
        temporary_pct = self.eta * participation.sqrt() * HUNDRED * urgency_mult
        if spread_pct > ZERO:
            temporary_pct += spread_pct * self.spread_weight

        # 永久冲击：与参与率、波动率正相关
        permanent_pct = self.gamma * participation * HUNDRED
        if volatility > ZERO:
            permanent_pct += volatility * participation * Decimal("0.1")

        expected_slippage_pct = temporary_pct + permanent_pct
        if expected_slippage_pct < self.min_slippage_pct:
            expected_slippage_pct = self.min_slippage_pct

        expected_slippage_abs = price * expected_slippage_pct / HUNDRED

        return ImpactEstimate(
            temporary_impact_pct=temporary_pct,
            permanent_impact_pct=permanent_pct,
            expected_slippage_pct=expected_slippage_pct,
            expected_slippage_abs=expected_slippage_abs,
            confidence=confidence,
            metadata={
                "participation_rate": str(participation),
                "urgency": urgency,
                "urgency_multiplier": str(urgency_mult),
                "adv": str(adv),
                "spread_pct": str(spread_pct),
                "volatility": str(volatility),
            },
        )

    def _urgency_multiplier(self, urgency: str) -> Decimal:
        mapping = {
            "passive": Decimal("0.7"),
            "normal": Decimal("1.0"),
            "aggressive": Decimal("1.5"),
        }
        return mapping.get(urgency.lower(), Decimal("1.0"))
