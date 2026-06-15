"""OMS 执行适配器层."""

from xmlx_vlm.ai_trader.oms.execution.factory import ExecutionAdapterFactory
from xmlx_vlm.ai_trader.oms.execution.paper.adapter import PaperExecutionAdapter

__all__ = ["ExecutionAdapterFactory", "PaperExecutionAdapter"]
