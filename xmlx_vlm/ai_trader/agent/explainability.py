"""决策可解释性：为每笔交易生成人类可读的决策依据."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.agent.config import AgentObjective
from xmlx_vlm.ai_trader.agent.decision import SignalEvaluation, TradeProposal


@dataclass
class DecisionRationale:
    """决策理由报告.

    回答：信号是什么、置信度多少、风险/收益比、止损位、为什么做/不做。
    """

    symbol: str
    action: str
    should_execute: bool
    signal_summary: str
    confidence: int
    risk_reward_ratio: str
    expected_return_pct: str
    expected_risk_pct: str
    stop_loss: Optional[str]
    take_profit: Optional[str]
    position_size_usd: str
    reasoning: List[str] = field(default_factory=list)
    constraints_checked: List[str] = field(default_factory=list)
    rejected_reason: Optional[str] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "should_execute": self.should_execute,
            "signal_summary": self.signal_summary,
            "confidence": self.confidence,
            "risk_reward_ratio": self.risk_reward_ratio,
            "expected_return_pct": self.expected_return_pct,
            "expected_risk_pct": self.expected_risk_pct,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size_usd": self.position_size_usd,
            "reasoning": self.reasoning,
            "constraints_checked": self.constraints_checked,
            "rejected_reason": self.rejected_reason,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        lines = [
            f"## {self.symbol} {self.action.upper()}",
            "",
            f"- **执行建议**: {'是' if self.should_execute else '否'}",
            f"- **信号**: {self.signal_summary}",
            f"- **置信度**: {self.confidence}/100",
            f"- **风险收益比**: {self.risk_reward_ratio}",
            f"- **预期收益**: {self.expected_return_pct}%",
            f"- **预期风险**: {self.expected_risk_pct}%",
            f"- **止损**: {self.stop_loss or '未设置'}",
            f"- **止盈**: {self.take_profit or '未设置'}",
            f"- **仓位大小**: {self.position_size_usd} USD",
            "",
            "### 决策逻辑",
        ]
        for r in self.reasoning:
            lines.append(f"- {r}")
        if self.constraints_checked:
            lines.extend(["", "### 约束检查"])
            for c in self.constraints_checked:
                lines.append(f"- {c}")
        if self.rejected_reason:
            lines.extend(["", f"### 拒绝原因\n- {self.rejected_reason}"])
        return "\n".join(lines)


class ExplainabilityBuilder:
    """根据信号评估、提案和目标函数生成可解释报告."""

    def __init__(self, objective: AgentObjective):
        self.objective = objective

    def build(
        self,
        evaluation: SignalEvaluation,
        proposal: Optional[TradeProposal],
        should_execute: bool,
        rejected_reason: Optional[str] = None,
    ) -> DecisionRationale:
        reasoning: List[str] = []
        constraints_checked: List[str] = []

        reasoning.append(
            f"收到 {evaluation.signal_type} 信号，方向 {evaluation.metadata.get('direction', 'unknown')}"
        )
        reasoning.append(f"置信度评分 {evaluation.confidence}/100")
        reasoning.append(
            f"止损 {evaluation.stop_loss} / 止盈 {evaluation.take_profit}，风险收益比 {evaluation.risk_reward_ratio}"
        )

        constraint = self.objective.position_constraint
        constraints_checked.append(
            f"最低置信度要求: {evaluation.confidence} >= {constraint.min_confidence} -> "
            f"{'通过' if evaluation.confidence >= constraint.min_confidence else '未通过'}"
        )
        constraints_checked.append(
            f"最低风险收益比要求: {evaluation.risk_reward_ratio} >= {constraint.min_risk_reward_ratio} -> "
            f"{'通过' if evaluation.risk_reward_ratio >= constraint.min_risk_reward_ratio else '未通过'}"
        )

        if proposal is not None:
            constraints_checked.append(
                f"仓位上限: {proposal.size_usd} USD <= {constraint.max_position_size_usd} USD -> "
                f"{'通过' if proposal.size_usd <= constraint.max_position_size_usd else '未通过'}"
            )
            constraints_checked.append(
                f"杠杆上限: {proposal.leverage} <= {constraint.max_leverage} -> "
                f"{'通过' if proposal.leverage <= constraint.max_leverage else '未通过'}"
            )
            if should_execute:
                reasoning.append(
                    f"生成 {proposal.action.value} 提案，仓位 {proposal.size_usd} USD，"
                    f"杠杆 {proposal.leverage}，预计风险 {proposal.expected_risk_pct}%"
                )
            else:
                reasoning.append("提案已生成，但当前模式/风控决定不执行")
        else:
            reasoning.append("未生成交易提案，不满足约束或信号无效")

        if rejected_reason:
            reasoning.append(f"拒绝原因: {rejected_reason}")

        return DecisionRationale(
            symbol=evaluation.symbol,
            action=proposal.action.value if proposal else "wait",
            should_execute=should_execute,
            signal_summary=f"{evaluation.signal_type} ({evaluation.metadata.get('direction', 'unknown')})",
            confidence=evaluation.confidence,
            risk_reward_ratio=str(evaluation.risk_reward_ratio),
            expected_return_pct=str(evaluation.expected_return_pct),
            expected_risk_pct=str(evaluation.expected_risk_pct),
            stop_loss=str(evaluation.stop_loss) if evaluation.stop_loss else None,
            take_profit=str(evaluation.take_profit) if evaluation.take_profit else None,
            position_size_usd=str(proposal.size_usd) if proposal else "0",
            reasoning=reasoning,
            constraints_checked=constraints_checked,
            rejected_reason=rejected_reason,
        )
