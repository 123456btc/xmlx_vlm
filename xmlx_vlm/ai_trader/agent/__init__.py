"""AI Agent 自主交易层.

提供：
- 目标函数与约束配置
- 信号评估与决策生成
- 决策可解释性
- A/B 测试与模型治理
- 人机协同模式与急停
- 自主决策闭环
"""

from __future__ import annotations

from .config import AgentObjective, PositionConstraint, RiskBudget
from .decision import AgentDecision, SignalEvaluation, TradeProposal
from .evaluator import SignalEvaluator
from .explainability import DecisionRationale, ExplainabilityBuilder
from .governance import ModelGovernance, ShadowRecord, VariantRegistry
from .loop import AutonomousAgentLoop
from .modes import AgentMode, ModeController
from .providers import MarketDataProvider
from .runtime import AgentEngine

__all__ = [
    "AgentDecision",
    "AgentEngine",
    "AgentMode",
    "AgentObjective",
    "AutonomousAgentLoop",
    "DecisionRationale",
    "ExplainabilityBuilder",
    "MarketDataProvider",
    "ModeController",
    "ModelGovernance",
    "PositionConstraint",
    "RiskBudget",
    "ShadowRecord",
    "SignalEvaluation",
    "SignalEvaluator",
    "TradeProposal",
    "VariantRegistry",
]
