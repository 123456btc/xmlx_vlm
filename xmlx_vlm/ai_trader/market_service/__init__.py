"""AI Trader 机构级行情服务.

提供常驻 WebSocket 连接、内存状态机、事件总线与技术指标计算，
供 AI Agent 以毫秒级延迟消费行情数据。
"""

from .alerts import AlertConfig, AlertEngine
from .columnar_store import ColumnarChunk, ColumnarMarketStore, SymbolPartition
from .events import EventBus, MarketEvent
from .service import MarketDataService
from .state import MarketState, SymbolState

__all__ = [
    "AlertConfig",
    "AlertEngine",
    "ColumnarChunk",
    "ColumnarMarketStore",
    "EventBus",
    "MarketDataService",
    "MarketEvent",
    "MarketState",
    "SymbolPartition",
    "SymbolState",
]
