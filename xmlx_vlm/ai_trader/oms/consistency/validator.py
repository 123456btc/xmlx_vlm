"""Paper-to-Live 一致性校验器."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List

from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, HUNDRED, pct_change


@dataclass
class ConsistencyReport:
    """一致性校验报告."""

    consistent: bool = False
    avg_price_diff_pct: Decimal = ZERO
    filled_qty_diff_pct: Decimal = ZERO
    state_path_diff: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    tolerance_pct: Decimal = Decimal("0.1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consistent": self.consistent,
            "avg_price_diff_pct": str(self.avg_price_diff_pct),
            "filled_qty_diff_pct": str(self.filled_qty_diff_pct),
            "state_path_diff": self.state_path_diff,
            "messages": self.messages,
            "tolerance_pct": str(self.tolerance_pct),
        }


class PaperLiveConsistencyValidator:
    """比较 paper 与 live 订单执行结果的一致性."""

    def __init__(self, tolerance_pct: Decimal = Decimal("0.1")):
        self.tolerance_pct = to_decimal(tolerance_pct)

    def compare(
        self,
        paper_order: Order,
        live_order: Order,
    ) -> ConsistencyReport:
        report = ConsistencyReport(tolerance_pct=self.tolerance_pct)

        # 1. 成交均价差异
        if paper_order.avg_fill_price > ZERO and live_order.avg_fill_price > ZERO:
            diff_pct = abs(pct_change(paper_order.avg_fill_price, live_order.avg_fill_price))
        elif paper_order.avg_fill_price == live_order.avg_fill_price:
            diff_pct = ZERO
        else:
            diff_pct = HUNDRED
        report.avg_price_diff_pct = diff_pct

        # 2. 成交量差异
        if paper_order.qty > ZERO:
            qty_diff_pct = abs(paper_order.filled_qty - live_order.filled_qty) / paper_order.qty * HUNDRED
        else:
            qty_diff_pct = ZERO if paper_order.filled_qty == live_order.filled_qty else HUNDRED
        report.filled_qty_diff_pct = qty_diff_pct

        # 3. 状态路径差异
        paper_states = [str(s) for s in paper_order.algo_params.get("state_path", [])]
        live_states = [str(s) for s in live_order.algo_params.get("state_path", [])]
        if paper_states != live_states:
            report.state_path_diff = [f"paper: {paper_states}", f"live: {live_states}"]
            report.messages.append("state path differs")

        # 4. 最终状态差异
        if paper_order.state != live_order.state:
            report.messages.append(
                f"final state differs: paper={paper_order.state.value}, live={live_order.state.value}"
            )

        # 5. 综合判定
        report.consistent = (
            diff_pct <= self.tolerance_pct
            and qty_diff_pct <= self.tolerance_pct
            and not report.state_path_diff
            and paper_order.state == live_order.state
        )

        if not report.consistent:
            if diff_pct > self.tolerance_pct:
                report.messages.append(
                    f"avg price diff {diff_pct:.4f}% exceeds tolerance {self.tolerance_pct:.4f}%"
                )
            if qty_diff_pct > self.tolerance_pct:
                report.messages.append(
                    f"filled qty diff {qty_diff_pct:.4f}% exceeds tolerance {self.tolerance_pct:.4f}%"
                )
        else:
            report.messages.append("paper and live execution are consistent")

        return report
