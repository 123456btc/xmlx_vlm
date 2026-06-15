"""OMS 自定义异常."""

from __future__ import annotations


class OMSError(Exception):
    """OMS 根异常."""


class RiskRejectedError(OMSError):
    """风控拒绝."""

    def __init__(self, rule_name: str, reason: str):
        self.rule_name = rule_name
        self.reason = reason
        super().__init__(f"Risk rejected by {rule_name}: {reason}")


class CircuitTrippedError(OMSError):
    """熔断器已触发."""

    def __init__(self, circuit_name: str, reason: str):
        self.circuit_name = circuit_name
        self.reason = reason
        super().__init__(f"Circuit tripped: {circuit_name} - {reason}")


class LiveTradingNotEnabledError(OMSError):
    """实盘交易未启用."""


class AdapterError(OMSError):
    """执行适配器异常."""


class OrderStateError(OMSError):
    """订单状态机非法转换."""


class ConfigurationError(OMSError):
    """OMS 配置错误."""
