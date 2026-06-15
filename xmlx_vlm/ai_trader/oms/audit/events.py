"""审计事件定义."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.constants import AuditEventType
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


@dataclass
class AuditEvent:
    """审计事件."""

    event_type: AuditEventType
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp_ms: int = field(default_factory=utc_now_ms)
    client_order_id: Optional[str] = None
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Any] = None

    def __post_init__(self):
        if isinstance(self.event_type, str):
            self.event_type = AuditEventType(self.event_type)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["raw"] = self._serialize_raw(self.raw)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @staticmethod
    def _serialize_raw(raw: Any) -> Any:
        if raw is None:
            return None
        try:
            json.dumps(raw, ensure_ascii=False, default=str)
            return raw
        except (TypeError, ValueError):
            return str(raw)
