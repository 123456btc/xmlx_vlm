# SPDX-License-Identifier: Apache-2.0
"""
Dynamic Position Sizing Engine: ATR Volatility-Parity and Fractional Kelly Sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


@dataclass
class SizingRecommendation:
    """仓位计算推荐结果."""

    recommended_qty: Decimal
    recommended_notional_usd: Decimal
    stop_distance: Decimal
    suggested_stop_loss: Decimal
    suggested_take_profit: Decimal
    max_risk_usd: Decimal
    kelly_multiplier: Decimal
    is_clamped: bool
    notes: str


class PositionSizer:
    """基于波动率 (ATR) 与分数凯利公式的动态头寸计算器."""

    def __init__(
        self,
        default_risk_pct: float = 0.015,  # 单笔默认风险 1.5% 账户权益
        max_notional_ratio: float = 0.25,  # 单笔名义价值上限 25% 账户权益
        default_atr_mult: float = 2.0,    # 止损距离倍数 (2.0 * ATR)
        reward_risk_ratio: float = 2.0,   # 默认盈亏比 (2.0:1)
        kelly_fraction: float = 0.25,     # 四分之一凯利
    ):
        self.default_risk_pct = to_decimal(default_risk_pct)
        self.max_notional_ratio = to_decimal(max_notional_ratio)
        self.default_atr_mult = to_decimal(default_atr_mult)
        self.reward_risk_ratio = to_decimal(reward_risk_ratio)
        self.kelly_fraction = to_decimal(kelly_fraction)

    def compute_kelly_fraction(
        self,
        win_rate: float,
        win_loss_ratio: float,
    ) -> Decimal:
        """计算分数凯利系数 f* = (p * (b + 1) - 1) / b * fraction."""
        if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
            return Decimal("1.0")

        p = to_decimal(win_rate)
        b = to_decimal(win_loss_ratio)
        # f = (p * (b + 1) - 1) / b
        f_raw = (p * (b + Decimal("1.0")) - Decimal("1.0")) / b
        if f_raw <= ZERO:
            return Decimal("0.5")  # 负期望时降为最小防守比例

        kelly_adj = f_raw * self.kelly_fraction
        # 限制在 [0.5, 1.5] 之间作为动态调节系数
        return max(Decimal("0.5"), min(Decimal("1.5"), Decimal("1.0") + kelly_adj))

    def calculate(
        self,
        account_equity: Decimal | float | str,
        mark_price: Decimal | float | str,
        atr: Decimal | float | str,
        is_long: bool = True,
        risk_pct: Optional[float] = None,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> SizingRecommendation:
        """根据 ATR 和账户资金计算严格风控下的建议头寸大小."""
        equity = to_decimal(account_equity)
        price = to_decimal(mark_price)
        atr_val = to_decimal(atr)

        if equity <= ZERO or price <= ZERO:
            return SizingRecommendation(
                recommended_qty=ZERO,
                recommended_notional_usd=ZERO,
                stop_distance=ZERO,
                suggested_stop_loss=ZERO,
                suggested_take_profit=ZERO,
                max_risk_usd=ZERO,
                kelly_multiplier=Decimal("1.0"),
                is_clamped=False,
                notes="Invalid equity or price",
            )

        # 1. 确定单笔最大可承受美元风险
        active_risk_pct = to_decimal(risk_pct) if risk_pct is not None else self.default_risk_pct
        max_risk_usd = equity * active_risk_pct

        # 2. 确定止损点距离（若 ATR 缺失则保底使用 1.5% 价格）
        if atr_val > ZERO:
            stop_distance = atr_val * self.default_atr_mult
        else:
            stop_distance = price * Decimal("0.015")

        # 3. 凯利调节
        kelly_mult = Decimal("1.0")
        if win_rate is not None and win_loss_ratio is not None:
            kelly_mult = self.compute_kelly_fraction(win_rate, win_loss_ratio)

        adjusted_risk_usd = max_risk_usd * kelly_mult

        # 4. 基于风险距离反推数量：Qty = Risk$ / StopDistance
        raw_qty = adjusted_risk_usd / stop_distance if stop_distance > ZERO else ZERO
        raw_notional_usd = raw_qty * price

        # 5. 上限硬约束：单笔名义价值不能超过账户权益的设定上限（如 25%）
        max_notional_cap = equity * self.max_notional_ratio
        is_clamped = False
        if raw_notional_usd > max_notional_cap:
            raw_notional_usd = max_notional_cap
            raw_qty = raw_notional_usd / price
            is_clamped = True

        # 6. 计算推荐的止损价和止盈价
        take_profit_distance = stop_distance * self.reward_risk_ratio
        if is_long:
            suggested_stop = max(ZERO, price - stop_distance)
            suggested_tp = price + take_profit_distance
        else:
            suggested_stop = price + stop_distance
            suggested_tp = max(ZERO, price - take_profit_distance)

        notes = (
            f"ATR Risk-Parity size calculated (ATR={float(atr_val):.4f}, "
            f"StopDist={float(stop_distance):.4f}, Kelly={float(kelly_mult):.2f})"
        )

        return SizingRecommendation(
            recommended_qty=raw_qty,
            recommended_notional_usd=raw_notional_usd,
            stop_distance=stop_distance,
            suggested_stop_loss=suggested_stop,
            suggested_take_profit=suggested_tp,
            max_risk_usd=adjusted_risk_usd,
            kelly_multiplier=kelly_mult,
            is_clamped=is_clamped,
            notes=notes,
        )
