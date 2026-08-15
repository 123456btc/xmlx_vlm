"""交易工具 —— 统一对接 OMS.

默认纸盘模式；实盘交易需要 AI_TRADER_LIVE=1 环境变量 + CLI --live 参数。
所有真实资金操作都经过 OMS 风控、审计、熔断、急停机制。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.config import DEFAULT_RISK, LOGS_DIR
from xmlx_vlm.ai_trader.oms.config.settings import get_settings
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.exceptions import (
    CircuitTrippedError,
    LiveTradingNotEnabledError,
    RiskRejectedError,
)
from xmlx_vlm.ai_trader.tools.market import MarketDataTool

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """保留旧版纸盘仓位结构，用于兼容原有状态文件."""

    symbol: str
    side: str  # long / short
    qty: float
    entry_price: float
    entry_at: float = field(default_factory=lambda: time.time())
    unrealized_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "entry_at": self.entry_at,
            "unrealized_pnl": self.unrealized_pnl,
        }


class TradingTool:
    """交易执行工具：统一通过 OMS 下单，支持 paper / live 模式."""

    name = "trading"
    description = (
        "执行交易操作：查询持仓、模拟/真实下单、平仓、紧急停止。"
        "默认使用纸盘模式；实盘交易需要显式启用并配置 API 凭证。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get_positions", "place_order", "close_position", "emergency_stop"],
                "description": "交易操作类型",
            },
            "symbol": {
                "type": "string",
                "description": "交易对，例如 BTC/USDT",
            },
            "side": {
                "type": "string",
                "enum": ["buy", "sell"],
                "description": "下单方向",
            },
            "qty": {
                "type": "number",
                "description": "下单数量",
            },
            "mode": {
                "type": "string",
                "enum": ["paper", "live"],
                "description": "交易模式，默认 paper",
                "default": "paper",
            },
            "order_type": {
                "type": "string",
                "enum": ["market", "limit"],
                "description": "订单类型，默认 market",
                "default": "market",
            },
            "price": {
                "type": "number",
                "description": "限价单价格（order_type=limit 时必填）",
            },
            "post_only": {
                "type": "boolean",
                "description": "只做 Maker 挂单 (Post-Only)，若会立即成交则自动取消，避免产生 Taker 摩擦手续费",
                "default": False,
            },
            "reduce_only": {
                "type": "boolean",
                "description": "只减仓模式 (Reduce-Only)，仅用于平仓或减仓，绝不会开出反向仓位",
                "default": False,
            },
        },
        "required": ["action"],
    }

    def __init__(self, oms: Optional[OMSEngine] = None):
        # 兼容旧版本地纸盘状态
        self._legacy_positions: Dict[str, PaperPosition] = {}
        self._legacy_trades: List[Dict[str, Any]] = []
        self._legacy_daily_pnl = 0.0
        self._legacy_last_day = time.strftime("%Y-%m-%d")
        self._market = MarketDataTool()
        self._load_legacy_state()

        # OMS 引擎（懒加载）
        self._oms: Optional[OMSEngine] = oms

    @property
    def oms(self) -> OMSEngine:
        if self._oms is None:
            settings = get_settings()
            self._oms = OMSEngine(settings=settings, market_data_tool=self._market)
        return self._oms

    def _run_async(self, coro) -> Any:
        """同步入口中运行异步协程."""
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:
            # 如果已经在事件循环中（如某些测试环境），使用 nest_asyncio 风格
            if "already running" in str(exc):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(coro)
            raise

    def _current_price(self, symbol: str) -> float:
        import re

        text = self._market.get_ticker(symbol, "hyperliquid")
        # 匹配 mark=64,534.00 或 last=64,534.00，保留数字与逗号
        match = re.search(r"(?:mark|last)=([\d,]+\.?\d*)", text)
        if match:
            return float(match.group(1).replace(",", ""))
        raise RuntimeError("无法从 Hyperliquid 获取当前价格")

    def _legacy_state_path(self) -> Path:
        return LOGS_DIR / "paper_state.json"

    def _load_legacy_state(self):
        path = self._legacy_state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._legacy_positions = {
                    k: PaperPosition(**v) for k, v in data.get("positions", {}).items()
                }
                self._legacy_trades = data.get("trades", [])
                self._legacy_daily_pnl = data.get("daily_pnl", 0.0)
                self._legacy_last_day = data.get("last_day", time.strftime("%Y-%m-%d"))
            except Exception as exc:
                logger.warning("加载旧版纸盘状态失败: %s", exc)

    def _save_legacy_state(self):
        try:
            data = {
                "positions": {k: v.to_dict() for k, v in self._legacy_positions.items()},
                "trades": self._legacy_trades,
                "daily_pnl": self._legacy_daily_pnl,
                "last_day": self._legacy_last_day,
            }
            self._legacy_state_path().write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:
            logger.warning("保存旧版纸盘状态失败: %s", exc)

    # ── 公开方法 ──
    def get_positions(self) -> str:
        try:
            result = self._run_async(self.oms.sync())
            summary = self.oms.portfolio_summary()
            positions = summary.get("positions", [])
            if not positions:
                return "当前没有持仓"
            lines = []
            for p in positions:
                lines.append(
                    f"{p['symbol']} {p['side']}: qty={p['qty']}, "
                    f"entry={p['avg_entry_price']}, "
                    f"mark={p['mark_price']}, "
                    f"unrealized={p['unrealized_pnl']}, "
                    f"realized={p['realized_pnl']}"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.exception("get_positions failed")
            return f"查询持仓失败: {exc}"

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        mode: str = "paper",
        order_type: str = "market",
        price: Optional[float] = None,
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> str:
        settings = self.oms.settings

        # 如果请求 live 但当前系统未启用实盘（例如处于纸盘状态且没有激活的实盘 key），明确拒绝
        if mode == "live" and not self.oms.is_live:
            return (
                "[拒绝] 实盘模式当前未启用（系统处于纸盘模拟状态）。\n"
                "请先在 API 凭证管理页面中激活您的 API 密钥以启用实盘交易模式。"
            )

        # 获取参考价格用于风控与名义金额计算
        mark_price = oracle_price = None
        try:
            mark_price = Decimal(str(self._current_price(symbol)))
            oracle_price = mark_price
        except Exception as exc:
            logger.warning("failed to get mark price for risk check: %s", exc)

        # 如果启用 post_only，强制使用 limit 挂单价格
        actual_order_type = order_type
        order_price = Decimal(str(price)) if price is not None else None
        if post_only:
            actual_order_type = "limit"
            if order_price is None and mark_price is not None:
                order_price = mark_price

        # 构建订单：市价单无 price 时，用 mark_price 计算名义金额以通过风控
        if actual_order_type == "market" and order_price is None and mark_price is not None:
            order_price = mark_price

        order = self.oms.create_order(
            symbol=symbol,
            side=side,
            qty=Decimal(str(qty)),
            order_type=actual_order_type,
            price=order_price,
            reduce_only=reduce_only,
        )

        try:
            result = self._run_async(
                self.oms.submit_order(order, mark_price=mark_price, oracle_price=oracle_price)
            )
            status = result.get("status")
            if status == "dry_run":
                return (
                    f"[DRY-RUN] 订单已通过风控，不会提交交易所\n"
                    f"{result['order']}"
                )
            order_dict = result["order"]
            exchange_mode = "LIVE" if self.oms.is_live else "PAPER"
            return (
                f"[{exchange_mode}] 已提交 {side} {qty} {symbol} @ "
                f"{order_dict.get('avg_fill_price', 'pending')}\n"
                f"状态: {order_dict['state']}"
            )
        except RiskRejectedError as exc:
            return f"[风控拒绝] {exc.rule_name}: {exc.reason}"
        except CircuitTrippedError as exc:
            return f"[熔断] {exc.circuit_name}: {exc.reason}"
        except LiveTradingNotEnabledError as exc:
            return f"[实盘未启用] {exc}"
        except Exception as exc:
            logger.exception("place_order failed")
            return f"下单失败: {exc}"

    def close_position(self, symbol: str) -> str:
        try:
            order = self._run_async(self.oms.close_position(symbol))
            if order is None:
                return f"没有找到 {symbol} 的持仓"
            return f"已平仓 {order.symbol}: {order.side.value} {order.qty}"
        except Exception as exc:
            logger.exception("close_position failed")
            return f"平仓失败: {exc}"

    def emergency_stop(self) -> str:
        try:
            result = self._run_async(self.oms.emergency_stop(flatten=True))
            return f"急停已触发，已清空持仓并锁定新开仓。详情: {result}"
        except Exception as exc:
            logger.exception("emergency_stop failed")
            return f"急停失败: {exc}"

    def run(self, **kwargs) -> str:
        """工具统一入口."""
        action = kwargs.get("action")
        try:
            if action == "get_positions":
                return self.get_positions()
            if action == "place_order":
                return self.place_order(
                    symbol=kwargs.get("symbol", ""),
                    side=kwargs.get("side", ""),
                    qty=float(kwargs.get("qty", 0)),
                    mode=kwargs.get("mode", "paper"),
                    order_type=kwargs.get("order_type", "market"),
                    price=kwargs.get("price"),
                    post_only=bool(kwargs.get("post_only", False)),
                    reduce_only=bool(kwargs.get("reduce_only", False)),
                )
            if action == "close_position":
                return self.close_position(kwargs.get("symbol", ""))
            if action == "emergency_stop":
                return self.emergency_stop()
            return f"错误：未知的 action={action}"
        except Exception as exc:
            logger.exception("trading tool failed")
            return f"交易操作失败: {exc}"
