"""信号数据模型."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class SignalSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Signal:
    """一个市场信号或情报事件."""

    type: str
    symbol: str
    severity: SignalSeverity
    title: str
    detail: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.severity, str):
            self.severity = SignalSeverity(self.severity.lower())
        self.symbol = self.symbol.upper()

    @property
    def debounce_key(self) -> str:
        return f"{self.type}:{self.symbol}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "symbol": self.symbol,
            "severity": self.severity.value,
            "title": self.title,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }
