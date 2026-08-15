"""AI Trader In-Memory SDK.

专为 Programmatic Tool Calling (PTC) / Code Mode 设计的轻量级交易 SDK。
Agent 生成的 Python 脚本可直接通过 import 或预注入的 `sdk` 变量访问高性能内存状态与 OMS。
"""

from xmlx_vlm.ai_trader.sdk.client import TraderSDK

__all__ = ["TraderSDK"]
