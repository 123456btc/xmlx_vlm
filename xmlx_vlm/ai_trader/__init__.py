"""AI Trader — 基于 XMLX-VLM 的聊天即交易助手.

提供行情查询、K 线图生成、视觉分析、模拟/实盘交易等工具，
让用户通过自然语言与本地 VLM 对话即可完成交易决策.
"""

from .tools.market import MarketDataTool
from .tools.chart import ChartTool
from .tools.trading import TradingTool
from .tools.registry import ToolRegistry

__all__ = [
    "MarketDataTool",
    "ChartTool",
    "TradingTool",
    "ToolRegistry",
]
