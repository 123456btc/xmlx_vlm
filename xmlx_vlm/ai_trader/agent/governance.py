"""A/B 测试与模型治理.

支持多个 prompt/策略变体并行运行 shadow trading，并评估效果。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.agent.decision import AgentDecision, TradeProposal
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


@dataclass
class Variant:
    """一个策略/prompt 变体."""

    variant_id: str
    name: str
    description: str
    prompt_template: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "description": self.description,
            "prompt_template": self.prompt_template,
            "config_overrides": self.config_overrides,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ShadowRecord:
    """一次 shadow 决策记录，用于后续评估."""

    record_id: str
    variant_id: str
    symbol: str
    proposal: TradeProposal
    mark_price: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    realized_pnl_pct: Optional[Decimal] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "variant_id": self.variant_id,
            "symbol": self.symbol,
            "proposal": self.proposal.to_dict(),
            "mark_price": str(self.mark_price),
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "realized_pnl_pct": str(self.realized_pnl_pct) if self.realized_pnl_pct is not None else None,
            "notes": self.notes,
        }


class VariantRegistry:
    """变体注册表."""

    def __init__(self):
        self._variants: Dict[str, Variant] = {}
        self.register(
            Variant(
                variant_id="default",
                name="Default",
                description="默认策略变体",
                prompt_template="default",
            )
        )

    def register(self, variant: Variant) -> None:
        self._variants[variant.variant_id] = variant

    def get(self, variant_id: str) -> Optional[Variant]:
        return self._variants.get(variant_id)

    def list_variants(self) -> List[Variant]:
        return list(self._variants.values())

    def create_variant(
        self,
        name: str,
        description: str,
        prompt_template: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Variant:
        variant = Variant(
            variant_id=uuid.uuid4().hex[:12],
            name=name,
            description=description,
            prompt_template=prompt_template,
            config_overrides=config_overrides or {},
        )
        self.register(variant)
        return variant


class ModelGovernance:
    """模型治理：记录决策、shadow trading、评估变体表现."""

    def __init__(self, registry: Optional[VariantRegistry] = None):
        self.registry = registry or VariantRegistry()
        self._decisions: List[AgentDecision] = []
        self._shadow_records: List[ShadowRecord] = []

    # ── 决策记录 ──
    def record_decision(self, decision: AgentDecision) -> None:
        self._decisions.append(decision)

    def list_decisions(
        self, variant_id: Optional[str] = None, symbol: Optional[str] = None
    ) -> List[AgentDecision]:
        result = self._decisions
        if variant_id:
            result = [d for d in result if d.variant_id == variant_id]
        if symbol:
            result = [d for d in result if d.symbol.upper() == symbol.upper()]
        return result

    # ── Shadow trading ──
    def record_shadow(
        self,
        variant_id: str,
        symbol: str,
        proposal: TradeProposal,
        mark_price: Decimal,
    ) -> ShadowRecord:
        record = ShadowRecord(
            record_id=uuid.uuid4().hex[:16],
            variant_id=variant_id,
            symbol=symbol.upper(),
            proposal=proposal,
            mark_price=to_decimal(mark_price),
        )
        self._shadow_records.append(record)
        return record

    def resolve_shadow(
        self,
        record_id: str,
        exit_price: Decimal,
    ) -> Optional[ShadowRecord]:
        record = next((r for r in self._shadow_records if r.record_id == record_id), None)
        if record is None:
            return None
        entry = record.mark_price
        exit_p = to_decimal(exit_price)
        direction = 1 if record.proposal.action.value in ("open_long", "close_short") else -1
        if entry > ZERO:
            pnl_pct = direction * (exit_p - entry) / entry * Decimal("100")
        else:
            pnl_pct = ZERO
        record.realized_pnl_pct = pnl_pct
        record.resolved_at = datetime.now(timezone.utc)
        return record

    def list_shadow_records(self, variant_id: Optional[str] = None) -> List[ShadowRecord]:
        if variant_id is None:
            return list(self._shadow_records)
        return [r for r in self._shadow_records if r.variant_id == variant_id]

    # ── 变体评估 ──
    def evaluate_variant(self, variant_id: str) -> Dict[str, Any]:
        records = self.list_shadow_records(variant_id=variant_id)
        resolved = [r for r in records if r.realized_pnl_pct is not None]
        if not resolved:
            return {"variant_id": variant_id, "sample_count": 0, "win_rate": None, "avg_pnl_pct": None}
        wins = sum(1 for r in resolved if r.realized_pnl_pct > ZERO)
        total = len(resolved)
        avg_pnl = sum(r.realized_pnl_pct for r in resolved) / Decimal(total)
        return {
            "variant_id": variant_id,
            "sample_count": total,
            "win_rate": f"{wins / total * 100:.2f}%",
            "avg_pnl_pct": str(avg_pnl),
        }

    def best_variant(self) -> Optional[str]:
        """返回平均 PnL 最高的变体 ID."""
        best: Optional[str] = None
        best_pnl: Optional[Decimal] = None
        for variant in self.registry.list_variants():
            stats = self.evaluate_variant(variant.variant_id)
            if stats["avg_pnl_pct"] is None:
                continue
            pnl = Decimal(stats["avg_pnl_pct"])
            if best_pnl is None or pnl > best_pnl:
                best_pnl = pnl
                best = variant.variant_id
        return best
