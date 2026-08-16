# SPDX-License-Identifier: Apache-2.0
"""
Automated Trade Post-Mortem and Quantitative Reflection Journal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, PositionSide
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO

logger = logging.getLogger(__name__)


@dataclass
class PostMortemReport:
    """平仓交易复盘与归因分析报告."""

    symbol: str
    side: str
    entry_price: Decimal
    exit_price: Decimal
    qty: Decimal
    pnl_usd: Decimal
    return_pct: Decimal
    holding_duration_min: float
    mfe_pct: float  # 最大有利偏移 (Max Favorable Excursion)
    mae_pct: float  # 最大不利偏移 (Max Adverse Excursion)
    entry_reason: str
    exit_reason: str
    category: str
    lessons: str

    def to_markdown(self) -> str:
        """格式化为 Markdown 复盘记录."""
        pnl_symbol = "+" if self.pnl_usd >= ZERO else ""
        return (
            f"### 📊 交易复盘: {self.symbol} ({self.side})\n"
            f"- **盈亏表现**: `{pnl_symbol}${float(self.pnl_usd):.2f}` (`{pnl_symbol}{float(self.return_pct):.2f}%`)\n"
            f"- **持仓时间**: `{self.holding_duration_min:.1f}` 分钟\n"
            f"- **入场/出场价**: `${float(self.entry_price):.4f}` ➔ `${float(self.exit_price):.4f}`\n"
            f"- **MFE (最大浮盈)**: `+{self.mfe_pct:.2f}%` | **MAE (最大浮亏)**: `-{self.mae_pct:.2f}%`\n"
            f"- **入场依据**: {self.entry_reason or 'AI 策略模型自动开仓'}\n"
            f"- **出场归因**: {self.exit_reason or '触发止盈/止损平仓'}\n"
            f"- **复盘反思 [{self.category}]**: {self.lessons}\n"
        )


class TradePostMortemGenerator:
    """自动化量化交易复盘与反思生成器."""

    @staticmethod
    def generate(
        symbol: str,
        side: PositionSide | OrderSide | str,
        entry_price: Decimal | float | str,
        exit_price: Decimal | float | str,
        qty: Decimal | float | str,
        entry_time_ms: int,
        exit_time_ms: int,
        entry_reason: str = "",
        exit_reason: str = "",
        highest_price: Optional[Decimal | float | str] = None,
        lowest_price: Optional[Decimal | float | str] = None,
    ) -> PostMortemReport:
        """生成一笔已平仓交易的深度复盘分析."""
        entry = to_decimal(entry_price)
        exit_p = to_decimal(exit_price)
        quantity = to_decimal(qty)

        side_str = side.value if hasattr(side, "value") else str(side)
        is_long = "long" in side_str.lower() or "buy" in side_str.lower()

        # 1. 盈亏与收益率计算
        if is_long:
            pnl = (exit_p - entry) * quantity
            ret_pct = ((exit_p - entry) / entry * Decimal("100")) if entry > ZERO else ZERO
        else:
            pnl = (entry - exit_p) * quantity
            ret_pct = ((entry - exit_p) / entry * Decimal("100")) if entry > ZERO else ZERO

        # 2. 持仓时长计算
        duration_ms = max(0, exit_time_ms - entry_time_ms)
        duration_min = round(duration_ms / 60000.0, 1)

        # 3. MFE (最大有利偏移) & MAE (最大不利偏移) 计算
        high_p = to_decimal(highest_price) if highest_price else max(entry, exit_p)
        low_p = to_decimal(lowest_price) if lowest_price else min(entry, exit_p)

        if is_long:
            mfe_pct = float((high_p - entry) / entry * 100) if entry > ZERO else 0.0
            mae_pct = float((entry - low_p) / entry * 100) if entry > ZERO else 0.0
        else:
            mfe_pct = float((entry - low_p) / entry * 100) if entry > ZERO else 0.0
            mae_pct = float((high_p - entry) / entry * 100) if entry > ZERO else 0.0

        mfe_pct = max(0.0, mfe_pct)
        mae_pct = max(0.0, mae_pct)

        # 4. 交易画像分类与经验反思生成
        if pnl > ZERO:
            if mfe_pct > float(ret_pct) * 2.0 and mfe_pct > 3.0:
                category = "PROFIT_GIVEBACK"
                lessons = f"虽然实现盈利，但最大浮盈达 +{mfe_pct:.2f}%，最终仅收获 +{float(ret_pct):.2f}%。建议启用更紧密的吊灯动态追踪止盈，防止大趋势反转时利润过度回撤。"
            elif duration_min < 5.0:
                category = "QUICK_SCALPING_WIN"
                lessons = "短线动量迅速兑现目标位，入场时机与盘口动量极其精准。"
            else:
                category = "SOLID_TREND_CAPTURE"
                lessons = "标准顺势波段交易，耐心持有并让利润充分奔跑，盈亏比与纪律性优异。"
        else:
            if mfe_pct > 2.0:
                category = "WINNER_TURNED_LOSER"
                lessons = f"严重风控瑕疵：持仓期间曾有 +{mfe_pct:.2f}% 的显著浮盈，但未及时拉升保本损（Break-Even Stop），导致盈利单变为亏损单。必须严格遵守 +1R 自动保本规则！"
            elif duration_min < 3.0:
                category = "FALSE_BREAKOUT_NOISE"
                lessons = "入场后迅速触及硬止损，可能遭遇假突破或盘口流动性扫单。建议多周期共振（5M+1H）确认后再入场。"
            else:
                category = "DISCIPLINED_STOP_LOSS"
                lessons = "严格执行预设止损，截断亏损，风险完全处于受控预期范围内。"

        report = PostMortemReport(
            symbol=symbol,
            side="LONG" if is_long else "SHORT",
            entry_price=entry,
            exit_price=exit_p,
            qty=quantity,
            pnl_usd=pnl,
            return_pct=ret_pct,
            holding_duration_min=duration_min,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            entry_reason=entry_reason,
            exit_reason=exit_reason,
            category=category,
            lessons=lessons,
        )

        logger.info("Trade Post-Mortem generated for %s: PnL=$%.2f, Category=%s", symbol, float(pnl), category)
        return report
