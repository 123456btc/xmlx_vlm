"""OMS 常量与枚举."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good till cancel
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill


class OrderState(str, Enum):
    DRAFT = "draft"
    PRE_TRADE_OK = "pre_trade_ok"
    SENT = "sent"                     # 已提交到适配器，等待交易所 ack
    SUBMITTED = "submitted"           # 交易所已收到（网络层）
    ACKNOWLEDGED = "acknowledged"     # 交易所已确认挂单/成交
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class RiskDecisionType(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    WARNING = "warning"


class EventType(str, Enum):
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_SENT = "order_sent"
    ORDER_ACKED = "order_acked"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCEL_REQUESTED = "order_cancel_requested"
    ORDER_CANCEL_ACKED = "order_cancel_acked"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXPIRED = "order_expired"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILLED = "order_partial_filled"

    RISK_PASSED = "risk_passed"
    RISK_REJECTED = "risk_rejected"
    RISK_WARNING = "risk_warning"

    CIRCUIT_TRIPPED = "circuit_tripped"
    CIRCUIT_RESET = "circuit_reset"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"

    PORTFOLIO_SYNCED = "portfolio_synced"
    ACCOUNT_SYNCED = "account_synced"


class AuditEventType(str, Enum):
    ORDER_INTENT = "order_intent"
    ORDER_SUBMIT = "order_submit"
    ORDER_UPDATE = "order_update"
    ORDER_FILL = "order_fill"
    RISK_DECISION = "risk_decision"
    CIRCUIT_EVENT = "circuit_event"
    KILL_SWITCH = "kill_switch"
    ACCOUNT_SNAPSHOT = "account_snapshot"
    POSITION_SNAPSHOT = "position_snapshot"
    ERROR = "error"


# Hyperliquid 固定配置
HL_MAINNET_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_MAINNET_EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"
HL_TESTNET_INFO_URL = "https://api.hyperliquid-testnet.xyz/info"
HL_TESTNET_EXCHANGE_URL = "https://api.hyperliquid-testnet.xyz/exchange"

HL_COIN_SIZES: Dict[str, float] = {
    # 币种最小下单数量，可在初始化时从 meta 动态加载
}

HL_PRICE_DECIMALS: Dict[str, int] = {
    # 币种价格精度
}
