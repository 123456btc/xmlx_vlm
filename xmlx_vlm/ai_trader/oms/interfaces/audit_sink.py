"""审计输出抽象接口."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AuditSink(ABC):
    """审计事件输出目标抽象基类."""

    @property
    @abstractmethod
    def name(self) -> str:
        """sink 名称."""
        ...

    @abstractmethod
    def write(self, event: "AuditEvent") -> None:
        """写入单条审计事件."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """刷写缓冲区."""
        ...

    def close(self) -> None:
        """释放资源，子类可重写."""
        pass
