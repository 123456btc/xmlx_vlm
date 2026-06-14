"""交易工具 —— 默认纸盘模式，可配置为实盘."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.config import DEFAULT_RISK, LOGS_DIR
from xmlx_vlm.ai_trader.tools.market import MarketDataTool

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
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
    """交易执行工具：默认纸盘，可扩展实盘."""

    name = "trading"
    description = "执行交易操作：查询持仓、模拟/真实下单、平仓、紧急停止。默认使用纸盘模式。"
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
        },
        "required": ["action"],
    }

    def __init__(self):
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[Dict[str, Any]] = []
        self.risk = DEFAULT_RISK.copy()
        self.daily_pnl = 0.0
        self.last_day = time.strftime("%Y-%m-%d")
        self.market = MarketDataTool()
        self._load_state()

    def _current_price(self, symbol: str) -> float:
        text = self.market.get_ticker(symbol, "hyperliquid")
        # text 形如 "BTC/USDC: last=64,321.50, bid=..., ask=..."
        for part in text.split(","):
            if "last=" in part:
                value = part.split("=", 1)[1].strip()
                return float(value.replace(",", ""))
        raise RuntimeError("无法从 Hyperliquid 获取当前价格")

    def _state_path(self) -> Path:
        return LOGS_DIR / "paper_state.json"

    def _load_state(self):
        path = self._state_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.positions = {
                    k: PaperPosition(**v) for k, v in data.get("positions", {}).items()
                }
                self.trades = data.get("trades", [])
                self.daily_pnl = data.get("daily_pnl", 0.0)
                self.last_day = data.get("last_day", time.strftime("%Y-%m-%d"))
            except Exception as exc:
                logger.warning("加载纸盘状态失败: %s", exc)

    def _save_state(self):
        try:
            data = {
                "positions": {k: v.to_dict() for k, v in self.positions.items()},
                "trades": self.trades,
                "daily_pnl": self.daily_pnl,
                "last_day": self.last_day,
            }
            self._state_path().write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:
            logger.warning("保存纸盘状态失败: %s", exc)

    def _reset_daily_if_needed(self):
        today = time.strftime("%Y-%m-%d")
        if today != self.last_day:
            self.daily_pnl = 0.0
            self.last_day = today

    def _check_risk(self, action: str, symbol: str, qty: float, price: float) -> Optional[str]:
        """返回 None 表示通过，否则返回拒绝理由."""
        self._reset_daily_if_needed()

        if action == "place_order":
            # 简化的风控检查
            notion = qty * price
            if notion <= 0:
                return "订单金额必须大于 0"
            # 单日亏损检查（这里用当日已实现 pnl 近似）
            if self.daily_pnl < -self.risk["max_daily_loss_pct"]:
                return f"当日亏损已超过 {self.risk['max_daily_loss_pct']}% 限制，禁止新开仓"
        return None

    def get_positions(self) -> str:
        if not self.positions:
            return "当前没有持仓"
        lines = []
        for pos in self.positions.values():
            lines.append(
                f"{pos.symbol} {pos.side}: qty={pos.qty}, entry={pos.entry_price:.2f}, "
                f"pnl={pos.unrealized_pnl:.2f}"
            )
        return "\n".join(lines)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        mode: str = "paper",
        price: Optional[float] = None,
    ) -> str:
        if mode != "paper":
            return "实盘模式尚未启用，请先使用 paper 模式验证策略"

        if price is None:
            price = self._current_price(symbol)

        risk_msg = self._check_risk("place_order", symbol, qty, price)
        if risk_msg:
            return f"风控拒绝: {risk_msg}"

        # 纸盘成交
        key = f"{symbol}_{side}"
        if key in self.positions:
            pos = self.positions[key]
            total_qty = pos.qty + qty
            pos.entry_price = (pos.entry_price * pos.qty + price * qty) / total_qty
            pos.qty = total_qty
        else:
            self.positions[key] = PaperPosition(
                symbol=symbol, side="long" if side == "buy" else "short", qty=qty, entry_price=price
            )

        trade = {
            "timestamp": time.time(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "mode": mode,
        }
        self.trades.append(trade)
        self._save_state()
        return f"[{mode.upper()}] 已成交 {side} {qty} {symbol} @ {price:.2f}"

    def close_position(self, symbol: str) -> str:
        closed = []
        for key in list(self.positions.keys()):
            if key.startswith(f"{symbol}_"):
                pos = self.positions.pop(key)
                closed.append(f"{pos.side} {pos.qty} @ {pos.entry_price:.2f}")
        if not closed:
            return f"没有找到 {symbol} 的持仓"
        self._save_state()
        return f"已平仓 {symbol}: " + ", ".join(closed)

    def emergency_stop(self) -> str:
        count = len(self.positions)
        self.positions.clear()
        self._save_state()
        return f"急停已触发，已清空 {count} 个持仓，暂停所有新开仓"

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
                )
            if action == "close_position":
                return self.close_position(kwargs.get("symbol", ""))
            if action == "emergency_stop":
                return self.emergency_stop()
            return f"错误：未知的 action={action}"
        except Exception as exc:
            logger.exception("trading tool failed")
            return f"交易操作失败: {exc}"
