"""确定性量化规则验证器 (Deterministic Verifiers) & 策略优化引擎.

遵循量化策略专家与 DeepSeek R1 验证哲学：
1. 剥离对大模型主观置信度的盲信，进行端到端确定性数学校验。
2. 引入投资组合级相关性与总 Net Delta 敞口风控，防止山寨币多头杠杆共振踩踏。
3. 引入资金费率 (Funding Rate Carry) 磨损折算，杜绝高额资金费侵蚀 Alpha。
4. 引入结构化技术位锚定 (Structural Technical Anchoring)，彻底解决大模型价格幻觉。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.agent.decision import ActionType, TradeProposal
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """验证结果."""

    passed: bool
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjusted_proposal: Optional[TradeProposal] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.passed:
            warn_msg = f" (警告: {', '.join(self.warnings)})" if self.warnings else ""
            return f"[Verifier 通过]{warn_msg} 风险收益比={self.metrics.get('risk_reward_ratio')}, 推荐仓位={self.metrics.get('recommended_qty')}"
        return f"[Verifier 拦截拒绝]: {'; '.join(self.rejection_reasons)}"


class MathematicalRiskRewardVerifier:
    """数学盈亏比与 ATR 波动包络检验器."""

    def __init__(self, min_rr: float = 1.8, min_atr_mult: float = 0.5, max_atr_mult: float = 4.5):
        self.min_rr = min_rr
        self.min_atr_mult = min_atr_mult
        self.max_atr_mult = max_atr_mult

    def verify(
        self,
        proposal: TradeProposal,
        atr: Optional[Decimal] = None,
    ) -> VerificationResult:
        reasons = []
        warnings = []
        metrics = {}

        entry = to_decimal(proposal.entry_price or proposal.stop_loss or ZERO)
        stop_loss = to_decimal(proposal.stop_loss or ZERO)
        take_profit = to_decimal(proposal.take_profit or ZERO)

        if entry <= ZERO or stop_loss <= ZERO or take_profit <= ZERO:
            reasons.append(f"价格非正: entry={entry}, stop_loss={stop_loss}, take_profit={take_profit}")
            return VerificationResult(passed=False, rejection_reasons=reasons)

        is_long = proposal.action in (ActionType.OPEN_LONG, ActionType.CLOSE_SHORT)
        is_short = proposal.action in (ActionType.OPEN_SHORT, ActionType.CLOSE_LONG)

        # 1. 逻辑方向校验
        if is_long:
            if not (stop_loss < entry < take_profit):
                reasons.append(f"多头止损止盈顺序错误: 要求 SL({stop_loss}) < Entry({entry}) < TP({take_profit})")
        elif is_short:
            if not (take_profit < entry < stop_loss):
                reasons.append(f"空头止损止盈顺序错误: 要求 TP({take_profit}) < Entry({entry}) < SL({stop_loss})")

        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)

        if risk <= ZERO:
            reasons.append("止损距离为 0，无法计算风险敞口")
            return VerificationResult(passed=False, rejection_reasons=reasons)

        rr = float(reward / risk)
        metrics["risk_reward_ratio"] = round(rr, 2)
        metrics["risk_per_unit"] = float(risk)
        metrics["reward_per_unit"] = float(reward)

        # 2. 盈亏比硬性阈值
        if rr < self.min_rr:
            reasons.append(f"盈亏比过低: 当前 RR={rr:.2f} < 最低门槛 {self.min_rr:.2f}")

        # 3. ATR 波动范围校验（若有 ATR）
        if atr is not None and atr > ZERO:
            dist_atr = float(risk / atr)
            metrics["sl_distance_atr"] = round(dist_atr, 2)
            if dist_atr < self.min_atr_mult:
                reasons.append(f"止损过窄容易被噪音触发: 当前止损距离仅 {dist_atr:.2f} ATR (要求 ≥ {self.min_atr_mult} ATR)")
            elif dist_atr > self.max_atr_mult:
                warnings.append(f"止损距离偏宽 ({dist_atr:.2f} ATR)，请确保仓位足够轻")

        return VerificationResult(
            passed=len(reasons) == 0,
            rejection_reasons=reasons,
            warnings=warnings,
            metrics=metrics,
        )


class KellyCriterionSizer:
    """凯利公式 (Kelly Criterion) 仓位约束器."""

    def __init__(
        self,
        base_win_rate: float = 0.45,
        fraction: float = 0.25,  # Quarter-Kelly (四分之一凯利) 以保证稳健
        max_loss_budget_pct: float = 0.02,  # 单笔最大允许损失本金 2%
    ):
        self.base_win_rate = base_win_rate
        self.fraction = fraction
        self.max_loss_budget_pct = max_loss_budget_pct

    def compute_size(
        self,
        proposal: TradeProposal,
        equity: Decimal,
        rr: float,
    ) -> VerificationResult:
        reasons = []
        warnings = []
        metrics = {}

        if equity <= ZERO:
            reasons.append("账户可用权益 <= 0")
            return VerificationResult(passed=False, rejection_reasons=reasons)

        # Kelly: f* = (p * (b + 1) - 1) / b = p - (1 - p) / b
        p = self.base_win_rate
        b = max(rr, 0.5)
        raw_kelly = p - (1.0 - p) / b
        if raw_kelly <= 0:
            reasons.append(f"根据胜率 ({p*100:.0f}%) 与盈亏比 ({b:.2f})，凯利最优仓位为负 ({raw_kelly:.4f})，数学期望为负")
            return VerificationResult(passed=False, rejection_reasons=reasons)

        # 分数凯利
        applied_kelly = min(raw_kelly * self.fraction, self.max_loss_budget_pct)
        max_loss_dollar = equity * to_decimal(applied_kelly)

        entry = to_decimal(proposal.entry_price or proposal.stop_loss or ZERO)
        stop_loss = to_decimal(proposal.stop_loss or ZERO)
        risk_per_unit = abs(entry - stop_loss)

        if risk_per_unit <= ZERO:
            reasons.append("单笔风险距离为 0")
            return VerificationResult(passed=False, rejection_reasons=reasons)

        recommended_qty = max_loss_dollar / risk_per_unit
        proposed_qty = to_decimal(getattr(proposal, "size_usd", ZERO)) / entry if entry > ZERO else ZERO

        metrics["applied_kelly_fraction"] = round(applied_kelly, 4)
        metrics["max_loss_dollar"] = float(max_loss_dollar)
        metrics["recommended_qty"] = float(recommended_qty)
        metrics["proposed_qty"] = float(proposed_qty)

        # 若提案下单量超过凯利上限，自动调整或给出告警
        adjusted = None
        if proposed_qty > recommended_qty:
            warnings.append(
                f"提案数量 ({proposed_qty:.4f}) 超过凯利安全上限 ({recommended_qty:.4f})，建议下调仓位"
            )

        return VerificationResult(
            passed=len(reasons) == 0,
            rejection_reasons=reasons,
            warnings=warnings,
            metrics=metrics,
        )


class PortfolioCorrelationRiskVerifier:
    """投资组合级关联敞口与 Net Delta 检验器.
    
    防止同时做多/做空多个高 Beta 相关代币导致组合敞口放大。
    """

    def __init__(self, max_net_delta_multiplier: float = 1.5, default_beta: float = 1.0):
        self.max_net_delta_multiplier = max_net_delta_multiplier
        self.default_beta = default_beta

    def verify(
        self,
        proposal: TradeProposal,
        equity: Decimal,
        existing_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        reasons = []
        warnings = []
        metrics = {}

        if equity <= ZERO:
            return VerificationResult(passed=True)

        max_allowed_delta = float(equity) * self.max_net_delta_multiplier
        curr_net_delta = 0.0

        # 计算现有持仓的 Net Delta
        if existing_positions:
            for p in existing_positions:
                side = str(p.get("side", "long")).lower()
                size_usd = float(to_decimal(p.get("size_usd") or p.get("notional") or ZERO))
                beta = float(p.get("beta", self.default_beta))
                if side in ("long", "buy"):
                    curr_net_delta += size_usd * beta
                elif side in ("short", "sell"):
                    curr_net_delta -= size_usd * beta

        # 估算新提案带来的 Delta 变动
        proposal_size = float(to_decimal(proposal.size_usd))
        is_long = proposal.action in (ActionType.OPEN_LONG, ActionType.CLOSE_SHORT)
        new_net_delta = curr_net_delta + (proposal_size if is_long else -proposal_size)

        metrics["current_net_delta_usd"] = round(curr_net_delta, 2)
        metrics["projected_net_delta_usd"] = round(new_net_delta, 2)
        metrics["max_allowed_delta_usd"] = round(max_allowed_delta, 2)

        # 检查是否超过单向净敞口上限
        if abs(new_net_delta) > max_allowed_delta:
            direction_str = "多头" if new_net_delta > 0 else "空头"
            reasons.append(
                f"投资组合 {direction_str} 总净敞口将达到 ${abs(new_net_delta):.0f}，"
                f"超过账户允许上限 (${max_allowed_delta:.0f}，即 {self.max_net_delta_multiplier}x 权益)"
            )

        return VerificationResult(
            passed=len(reasons) == 0,
            rejection_reasons=reasons,
            warnings=warnings,
            metrics=metrics,
        )


class FundingRateCarryVerifier:
    """资金费率 (Funding Rate Carry) 磨损检验器."""

    def __init__(self, max_acceptable_daily_funding_loss_pct: float = 0.003):  # 0.3% / day
        self.max_acceptable_daily_funding_loss = max_acceptable_daily_funding_loss_pct

    def verify(
        self,
        proposal: TradeProposal,
        funding_rate: Optional[float] = None,
        estimated_holding_hours: float = 24.0,
    ) -> VerificationResult:
        reasons = []
        warnings = []
        metrics = {}

        if funding_rate is None:
            return VerificationResult(passed=True)

        is_long = proposal.action in (ActionType.OPEN_LONG, ActionType.CLOSE_SHORT)
        intervals = estimated_holding_hours / 8.0  # 8小时结算一次
        carry_pct = funding_rate * intervals

        # 多头支付正资金费，空头支付负资金费
        paying_funding = (is_long and funding_rate > 0) or (not is_long and funding_rate < 0)
        est_funding_loss_pct = abs(carry_pct) if paying_funding else -abs(carry_pct)

        metrics["funding_rate_8h"] = funding_rate
        metrics["estimated_funding_cost_pct"] = round(est_funding_loss_pct, 6)

        if paying_funding and est_funding_loss_pct > self.max_acceptable_daily_funding_loss:
            warnings.append(
                f"当前资金费率较高 (8h={funding_rate*100:.3f}%)，预估 {estimated_holding_hours:.0f}h "
                f"持仓资金费磨损达 {est_funding_loss_pct*100:.2f}%，需确保利润空间充足"
            )

        return VerificationResult(
            passed=True,
            rejection_reasons=reasons,
            warnings=warnings,
            metrics=metrics,
        )


class StructuralAnchorResolver:
    """结构化技术位锚定解析器 (防大模型价格幻觉)."""

    @staticmethod
    def resolve_anchor(
        anchor_name: str,
        mark_price: float,
        atr: float,
        volume_profile: Optional[Dict[str, Any]] = None,
        swing_low: Optional[float] = None,
        swing_high: Optional[float] = None,
        buffer_atr_mult: float = 0.5,
    ) -> Optional[float]:
        """将语义化技术位（如 'swing_low', 'vah', 'val', 'poc'）解析为真实价格."""
        anchor = anchor_name.lower().strip()
        buffer_dist = atr * buffer_atr_mult

        if "swing_low" in anchor and swing_low is not None:
            return round(swing_low - buffer_dist, 4)
        if "swing_high" in anchor and swing_high is not None:
            return round(swing_high + buffer_dist, 4)

        if volume_profile:
            if "val" in anchor and volume_profile.get("val") is not None:
                return round(float(volume_profile["val"]) - buffer_dist, 4)
            if "vah" in anchor and volume_profile.get("vah") is not None:
                return round(float(volume_profile["vah"]) + buffer_dist, 4)
            if "poc" in anchor and volume_profile.get("poc") is not None:
                return round(float(volume_profile["poc"]), 4)

        if "atr_sl" in anchor:
            return round(mark_price - atr * 1.5, 4)
        if "atr_tp" in anchor:
            return round(mark_price + atr * 3.0, 4)

        return None


class DeterministicProposalVerifier:
    """聚合式确定性规则终审验证器 (全流程)."""

    def __init__(
        self,
        min_rr: float = 1.8,
        min_atr_mult: float = 0.5,
        max_atr_mult: float = 4.5,
        base_win_rate: float = 0.45,
        fractional_kelly: float = 0.25,
        max_risk_pct: float = 0.02,
        max_net_delta_multiplier: float = 1.5,
    ):
        self.rr_verifier = MathematicalRiskRewardVerifier(
            min_rr=min_rr, min_atr_mult=min_atr_mult, max_atr_mult=max_atr_mult
        )
        self.kelly_sizer = KellyCriterionSizer(
            base_win_rate=base_win_rate,
            fraction=fractional_kelly,
            max_loss_budget_pct=max_risk_pct,
        )
        self.portfolio_verifier = PortfolioCorrelationRiskVerifier(
            max_net_delta_multiplier=max_net_delta_multiplier
        )
        self.funding_verifier = FundingRateCarryVerifier()

    def verify_proposal(
        self,
        proposal: TradeProposal,
        equity: Decimal,
        atr: Optional[Decimal] = None,
        existing_positions: Optional[List[Dict[str, Any]]] = None,
        funding_rate: Optional[float] = None,
    ) -> VerificationResult:
        """对提案进行端到端确定性数学、组合敞口与资金费率终审."""
        # 1. 检验盈亏比与 ATR 包络
        rr_res = self.rr_verifier.verify(proposal, atr=atr)
        if not rr_res.passed:
            return rr_res

        rr = rr_res.metrics.get("risk_reward_ratio", 2.0)

        # 2. 检验凯利安全仓位
        kelly_res = self.kelly_sizer.compute_size(proposal, equity, rr=rr)
        if not kelly_res.passed:
            return kelly_res

        # 3. 检验组合总 Delta 净敞口
        port_res = self.portfolio_verifier.verify(
            proposal, equity=equity, existing_positions=existing_positions
        )
        if not port_res.passed:
            return port_res

        # 4. 检验资金费率 Carry 磨损
        funding_res = self.funding_verifier.verify(proposal, funding_rate=funding_rate)

        # 合并所有警告与度量
        merged_warnings = (
            list(rr_res.warnings)
            + list(kelly_res.warnings)
            + list(port_res.warnings)
            + list(funding_res.warnings)
        )
        merged_metrics = {
            **rr_res.metrics,
            **kelly_res.metrics,
            **port_res.metrics,
            **funding_res.metrics,
        }

        return VerificationResult(
            passed=True,
            rejection_reasons=[],
            warnings=merged_warnings,
            metrics=merged_metrics,
        )
