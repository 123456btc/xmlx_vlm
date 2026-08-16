"""把 TradingContext 渲染为 LLM prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from xmlx_vlm.ai_trader.decision.context import TradingContext


DEFAULT_SYSTEM_PROMPT = """你是 AI Trader，一位专业的合约高频与量化交易专家。

你的任务：根据提供的账户、持仓、市场数据，每周期生成一组可执行交易决策。

在输出最终的 JSON 决策之前，你必须先进行深度分析。请在回复的最开始使用 `<think>` 和 `</think>` 标签，在标签内输出你的分析过程（Chain of Thought），包括趋势判断、仓位计算和风控检查。

输出格式（在 </think> 标签之后，必须是合法 JSON 数组，不要 Markdown 代码块）：
[
  {
    "action": "open_long",
    "symbol": "BTC/USDC",
    "position_size_usd": 15,
    "leverage": 3,
    "price": null,
    "stop_loss": 62000,
    "take_profit": 72000,
    "confidence": 75,
    "reasoning": "基于账户比例计算仓位，1h EMA金叉，CVD 1h流入，止损设在2倍ATR"
  }
]

action 可选值：
- open_long / open_short：开新仓或加仓（同向时为加仓）
- close_long / close_short：平仓或减仓
- hold：维持当前仓位
- wait：不操作

核心策略管理与风控约束（必须严格遵守）：

1. 账户比例化仓位管理原则（Account Percentage Sizing）：
   - 全资金比例换算：所有开仓、平仓、止盈、止损均严格按当前【账户总权益】的比例（百分比）动态计算。无论账户规模是 $30、$300 还是 $30,000，均按账户比例换算并执行，绝对不要因账户本金小而拒绝开仓！
   - 开仓名义价值计算：
     * 单笔开仓占用保证金比例：通常为账户总权益的 15% ~ 35%（结合当前行情置信度与可用保证金）。
     * 仓位名义价值（position_size_usd） = 占用保证金 * 杠杆倍数（leverage 推荐 2x~5x）。
     * 小微账户（如 $30~$100 USDC）兼容：交易所单笔最低名义价值门槛通常约为 $10~$12 USD。对于小额账户（如 $30），按 20%~30% 保证金配合 2x~3x 杠杆，名义价值即为 $12~$25 USD，完全满足且应正常开仓。
   - 止损止盈比例与价格计算：
     * 止损比例（Stop Loss）：按价格的 1.5%~4.0% 或 1.5~2.5 倍 ATR(14) 动态计算。多单 stop_loss = mark_price * (1 - 止损比例)；空单 stop_loss = mark_price * (1 + 止损比例)。
     * 止盈比例（Take Profit）：按照至少 1:1.5 到 1:3 的盈亏比计算目标价。多单 take_profit = mark_price * (1 + 止盈比例)；空单 take_profit = mark_price * (1 - 止盈比例)。

2. 风控与持仓主动治理（Risk Control & Position Pruning）：
   - 浮亏持仓果断止损：若当前持仓处于大幅浮亏（如浮亏 > 10%~20% 且趋势已破位），必须果断下达 close_long / close_short 止损平仓指令，释放保证金，防止回撤扩大。
   - 浮盈保本与止盈：当持仓浮盈达到目标的一半以上时，提示移动止损至入场均价（保本损）或分批止盈减仓。
   - 保证金安全边界：整体已用保证金使用率上限为 50%。一旦超过 50%，禁止新增开仓（仅允许 wait/hold/close）。

3. 加减仓与分批管理（Scaling In/Out）：
   - 盈利金字塔加仓（Pyramid Adding）：仅在当前持仓盈利且趋势增强时允许同向加仓；亏损持仓绝对不许补仓（逆势不加仓）。
   - 平仓与减仓执行：
     - 若要完全平仓，action 设为 close_long / close_short，将 position_size_usd 设为 null 或等于/大于当前持仓名义价值。
     - 若要部分减仓，action 设为 close_long / close_short，将 position_size_usd 设为希望减仓的名义价值（USD）。

4. 交易执行自律：
   - 当技术指标与盘口（如 RSI 处于动量区间、CVD 明显流入/流出、量价共振）出现较好胜率机会时，果断按账户比例下发开仓指令，严禁因账户本金规模较小而过度畏缩观望。

价格口径：以 mark_price 为基准。"""


GRID_SYSTEM_PROMPT = """你是 AI Grid Trader，一位专业的合约网格与高频交易专家。

你的任务：在指定交易对上管理一个区间网格策略，输出网格操作。

在输出最终的 JSON 决策之前，你必须先进行深度分析。请在回复的最开始使用 `<think>` 和 `</think>` 标签，在标签内输出你的分析过程（Chain of Thought），包括网格相对位置分析、波动率评估和风控检查。

输出格式（在 </think> 标签之后，必须是合法 JSON 数组）：
[
  {
    "action": "place_buy_limit",
    "symbol": "BTC/USDC",
    "price": 65000,
    "position_size_usd": 500,
    "confidence": 80,
    "reasoning": "价格接近网格下沿且ADX<20，挂买单"
  }
]

网格 action 可选值：
- place_buy_limit / place_sell_limit：在指定价格挂限价单
- cancel_order：取消某订单
- cancel_all_orders：取消所有网格单
- pause_grid / resume_grid：暂停/恢复网格
- adjust_grid：调整网格区间（需提供 price 作为参考）
- wait：不操作

合约网格核心管理与风控约束（必须严格遵守）：

1. 网格区间与间距管理（Grid & ATR-based Spacing）：
   - 基于 ATR(14) 动态间距：单格间距应根据当前波动率动态设置，一般设定为 0.5 倍到 1.5 倍 ATR(14)。
   - 趋势识别防单边（Trend Filter）：参考 ADX 指标。当 ADX > 25，表明当前处于强趋势行情，必须调用 pause_grid 暂停网格，或者 adjust_grid 顺势调整区间，防止在单边行情中被动接刀。当 ADX < 20 震荡市时，方可全力运行网格。

2. 仓位与保证金风控（Margin & Position Control）：
   - 低杠杆运行：网格交易一般面临多头或空头堆积，杠杆需控制在 3 倍以下。
   - 保证金预警：已用保证金占比不得超过 40%。若超过，需逐步调用 cancel_order 减少未成交挂单，回笼保证金。
   - 最大回撤硬止损：若当前网格累积未实现亏损超过账户总权益 of 5%，必须执行 cancel_all_orders 并平仓止损。

3. 分批建仓与调整算法（Grid Scaling & Adjustments）：
   - 区间突破处理：当价格突破网格预设的上沿（VAL）或下沿（VAH）时，如果确认趋势突破，需执行 adjust_grid 调整网格区间，或暂停网格，切忌盲目逆势死扛。
   - reasoning 必须说明当前价格所处的网格相对位置（如中轨、上轨、下轨）以及 ADX/ATR 的匹配逻辑。"""


VARIANT_PROMPTS = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "conservative": DEFAULT_SYSTEM_PROMPT + "\n\n风险偏好评级：保守。请降低开仓频率，提高 confidence 阈值，严格止损。",
    "aggressive": DEFAULT_SYSTEM_PROMPT + "\n\n风险偏好评级：激进。允许更高杠杆和更密集交易，但仍需遵守单笔与总体仓位限制。",
    "grid": GRID_SYSTEM_PROMPT,
}


@dataclass
class PromptSet:
    """一次决策所需的 system + user prompt."""

    system_prompt: str
    user_prompt: str
    variant: str = "default"


class PromptBuilder:
    """把 TradingContext 渲染为 LLM prompt."""

    def __init__(self, variant: str = "default"):
        self.variant = variant

    def build(self, context: TradingContext) -> PromptSet:
        system_prompt = VARIANT_PROMPTS.get(self.variant, DEFAULT_SYSTEM_PROMPT)
        user_prompt = self._render_user_prompt(context)
        return PromptSet(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            variant=self.variant,
        )

    def _render_user_prompt(self, context: TradingContext) -> str:
        sections: List[str] = []
        sections.append(f"当前时间: {context.current_time}")
        sections.append(f"运行时长: {context.runtime_minutes} 分钟")
        sections.append(f"周期编号: {context.cycle_number}")
        if context.trader_id:
            sections.append(f"策略 ID: {context.trader_id}")

        sections.append(self._account_section(context))
        sections.append(self._positions_section(context))
        sections.append(self._candidate_symbols_section(context))
        sections.append(self._market_section(context))
        if context.trading_stats:
            sections.append(self._stats_section(context))
        if context.recent_orders:
            sections.append(self._recent_orders_section(context))

        sections.append("\n请根据以上信息生成本周期交易决策（JSON 数组）：")
        return "\n\n".join(sections)

    def _account_section(self, context: TradingContext) -> str:
        account = context.account
        return (
            "【账户状态】\n"
            f"总权益: {account.equity}\n"
            f"可用保证金: {account.available_margin}\n"
            f"已用保证金: {account.used_margin}\n"
            f"保证金使用率: {account.margin_utilization_pct():.2f}%\n"
            f"账户模式: {account.mode or 'standard'}"
        )

    def _positions_section(self, context: TradingContext) -> str:
        if not context.positions:
            return "【当前持仓】\n无持仓"
        lines = ["【当前持仓】"]
        for pos in context.positions:
            lines.append(
                f"- {pos.symbol} {pos.side.value}: qty={pos.qty}, "
                f"entry={pos.avg_entry_price}, mark={pos.mark_price}, "
                f"unrealized={pos.unrealized_pnl}, realized={pos.realized_pnl}"
            )
        return "\n".join(lines)

    def _candidate_symbols_section(self, context: TradingContext) -> str:
        if not context.candidate_symbols:
            return "【候选交易对】\n无"
        return "【候选交易对】\n" + ", ".join(context.candidate_symbols)

    def _market_section(self, context: TradingContext) -> str:
        if not context.market_data:
            return "【市场数据】\n暂无"
        lines = ["【市场数据】"]
        for symbol, summary in context.market_data.items():
            lines.append(f"- {symbol}:")
            lines.append(f"  mark={summary.mark_price}, oracle={summary.oracle_price}, basis={summary.basis_pct:+.4f}%")
            lines.append(
                f"  24h_change={summary.change_24h_pct:+.2f}%, "
                f"volume={summary.volume_24h:,.2f}, "
                f"spread={summary.spread:.4f}"
            )
            if summary.atr14 is not None:
                lines.append(f"  ATR14={summary.atr14:.2f}, RSI14={summary.rsi14}")
            if summary.oi_change_1h_pct is not None:
                lines.append(
                    f"  OI_1h={summary.oi_change_1h_pct:+.2f}%, "
                    f"OI_24h={summary.oi_change_24h_pct:+.2f}%"
                )
            if summary.cvd_1h is not None:
                lines.append(f"  CVD_1h={summary.cvd_1h:,.2f}, CVD_4h={summary.cvd_4h}")
            # 6 大高阶币圈量化因子输出
            factor_tags = []
            if getattr(summary, "is_squeezed", False):
                factor_tags.append("布林带极度挤压(准备单边爆发)")
            if getattr(summary, "pinbar_type", "none") != "none":
                factor_tags.append(f"清算插针({summary.pinbar_type})")
            if getattr(summary, "cvd_divergence", "neutral") != "neutral":
                factor_tags.append(f"CVD背离({summary.cvd_divergence})")
            if getattr(summary, "oi_regime", "neutral") != "neutral":
                factor_tags.append(f"OI共振({summary.oi_regime})")
            if getattr(summary, "funding_zscore", None) is not None and abs(summary.funding_zscore) >= 2.0:
                factor_tags.append(f"费率极端偏离(Z={summary.funding_zscore:.2f})")
            if getattr(summary, "candle_efficiency", None) is not None:
                factor_tags.append(f"K线推进效率={summary.candle_efficiency:.2f}")

            if factor_tags:
                lines.append(f"  高阶量化因子: {', '.join(factor_tags)}")
        return "\n".join(lines)

    def _stats_section(self, context: TradingContext) -> str:
        stats = context.trading_stats
        return (
            "【历史统计】\n"
            f"总交易次数: {stats.total_trades}\n"
            f"胜率: {stats.win_rate}%\n"
            f"盈亏比: {stats.profit_factor}\n"
            f"总盈亏: {stats.total_pnl}\n"
            f"最大回撤: {stats.max_drawdown_pct}%"
        )

    def _recent_orders_section(self, context: TradingContext) -> str:
        lines = ["【最近订单】"]
        for order in context.recent_orders[-5:]:
            lines.append(
                f"- {order.symbol} {order.side}: entry={order.entry_price}, "
                f"exit={order.exit_price}, pnl={order.realized_pnl} ({order.pnl_pct}%), "
                f"hold={order.hold_duration}"
            )
        return "\n".join(lines)
