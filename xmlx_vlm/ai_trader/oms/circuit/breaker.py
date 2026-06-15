"""熔断器基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class CircuitBreaker(ABC):
    """熔断器抽象基类."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def is_tripped(self) -> bool:
        """当前是否已熔断."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """手动重置熔断器."""
        ...

    def check(self) -> Optional[str]:
        """如果已熔断返回原因，否则返回 None."""
        if self.is_tripped():
            return f"circuit {self.name} is tripped"
        return None
