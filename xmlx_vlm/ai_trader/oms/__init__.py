"""AI Trader Order Management System (OMS).

机构级、模块化、事件驱动的订单管理系统。
默认纸盘模式；实盘交易需要显式启用并配置 API 凭证。
"""

from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings, get_settings

__all__ = ["OMSEngine", "OMSSettings", "get_settings"]
