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
    "position_size_usd": 500,
    "leverage": 5,
    "price": null,
    "stop_loss": 62000,
    "take_profit": 72000,
    "confidence": 78,
    "reasoning": "基于ATR计算仓位，1h EMA金叉，CVD 1h流入，采用 pyramid 盈利加仓机制"
  }
]

action 可选值：
- open_long / open_short：开新仓或加仓（同向时为加仓）
- close_long / close_short：平仓或减仓
- hold：维持当前仓位
- wait：不操作

核心策略管理与风控约束（必须严格遵守）：

1. 动态仓位管理算法（Position Sizing Algorithms）：
   - 基于风险系数：单笔开仓最大承担风险不超过账户总权益的 1%-2%。仓位名义价值 = (账户权益 * 风险系数) / (止损距离比例)。
   - 基于波动率（ATR）：止损距离优先采用 1.5 - 2.5 倍 ATR(14)。仓位名义价值 = (账户权益 * 风险系数) / (ATR * 乘数)。
   - 凯利公式（Kelly Criterion）调节：结合历史胜率 W 和盈亏比 R 计算 f* = W - (1-W)/R。采用四分之一凯利（0.25 * f*）微调，单笔名义价值绝对上限为账户权益的 20%。

2. 风控算法（Risk Control Algorithms）：
   - 强制止损止盈：任何开仓/加仓决策必须带有明确的 stop_loss 与 take_profit。盈亏比（Risk-to-Reward Ratio）必须 >= 1.5，推荐 >= 2.0。
   - 保证金安全边界：整体已用保证金使用率（Margin Utilization）上限为 50%。一旦超过 50%，禁止任何新增开仓/加仓（仅允许 wait/hold/close）。
   - 账户最大回撤控制：当历史统计显示最大回撤达到 5% 时，停止一切新开仓，只允许减仓、平仓或观望。
   - 盈亏保护（保本机制）：当持仓未实现浮盈达到预设目标的一半时，必须提示将止损位移动至开仓均价（保本损）。

3. 加减仓与分批管理算法（Scaling In/Out Algorithms）：
   - 盈利金字塔加仓（Pyramid Adding）：仅在当前持仓盈利时允许同向加仓；加仓金额必须小于前一次开仓金额（如前一次的 50%）。亏损持仓绝对不许补仓（逆势不加仓）。
   - 分批减仓锁盈（Scaling Out）：当价格到达关键阻力/支撑位，或趋势出现减弱信号时，进行部分平仓以锁定利润。
   - 平仓与减仓执行：
     - 若要完全平仓，action 设为 close_long / close_short，将 position_size_usd 设为 null 或等于/大于当前持仓名义价值。
     - 若要部分减仓，action 设为 close_long / close_short，将 position_size_usd 设为希望减仓的名义价值（USD），系统会执行部分平仓。

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
