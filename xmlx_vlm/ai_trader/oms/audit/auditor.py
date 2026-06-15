"""审计事件分发器."""

from __future__ import annotations

import logging
from typing import List

from xmlx_vlm.ai_trader.oms.audit.events import AuditEvent
from xmlx_vlm.ai_trader.oms.interfaces.audit_sink import AuditSink

logger = logging.getLogger(__name__)


class Auditor:
    """把审计事件分发到多个 sink，失败时记录日志但不阻塞主流程."""

    def __init__(self, sinks: List[AuditSink]):
        self._sinks = list(sinks)

    def add_sink(self, sink: AuditSink) -> None:
        self._sinks.append(sink)

    def record(self, event: AuditEvent) -> None:
        for sink in self._sinks:
            try:
                sink.write(event)
            except Exception:
                logger.exception("audit sink %s failed", sink.name)

    def flush(self) -> None:
        for sink in self._sinks:
            try:
                sink.flush()
            except Exception:
                logger.exception("audit sink %s flush failed", sink.name)

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                logger.exception("audit sink %s close failed", sink.name)
