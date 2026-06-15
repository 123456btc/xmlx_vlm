"""执行适配器工厂."""

from __future__ import annotations

import logging
from typing import Any, Dict

from xmlx_vlm.ai_trader.oms.exceptions import ConfigurationError
from xmlx_vlm.ai_trader.oms.execution.hyperliquid.adapter import HyperliquidExecutionAdapter
from xmlx_vlm.ai_trader.oms.execution.paper.adapter import PaperExecutionAdapter
from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import ExecutionAdapter

logger = logging.getLogger(__name__)


class ExecutionAdapterFactory:
    """根据配置创建合适的执行适配器.

    本平台只有两种交易场所，地位相同：
    - paper / local_sim：本地仿真机构盘（零真实资金风险，机构级撮合）
    - hyperliquid：Hyperliquid 实盘
    """

    LOCAL_SIMULATION_EXCHANGES = {"local", "paper", "local_sim"}

    @staticmethod
    def create(
        exchange: str = "paper",
        market_data_tool=None,
        **kwargs: Any,
    ) -> ExecutionAdapter:
        exchange = exchange.lower()
        if exchange in ExecutionAdapterFactory.LOCAL_SIMULATION_EXCHANGES:
            return PaperExecutionAdapter(
                market_data_tool=market_data_tool,
                market_data_provider=kwargs.get("market_data_provider"),
                fill_slippage_pct=kwargs.get("fill_slippage_pct", 0),
                default_price=kwargs.get("default_price", 50000),
                latency_ms=kwargs.get("latency_ms", 0),
            )

        if exchange == "hyperliquid":
            return HyperliquidExecutionAdapter(
                wallet_address=kwargs.get("wallet_address"),
                private_key=kwargs.get("private_key"),
                signer_endpoint=kwargs.get("signer_endpoint"),
                testnet=kwargs.get("testnet", False),
                timeout=kwargs.get("timeout", 20),
            )

        raise ConfigurationError(f"unsupported exchange: {exchange}")
