"""QuantTracer - Structured telemetry and tracing for AI Trader agent execution."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class QuantTracer:
    """Agent-level structured telemetry tracer.
    
    Logs LLM generation costs (tokens), latencies, tool execution times,
    and agent dialogue rounds to a local JSONL trace log.
    """

    def __init__(self, trace_file_path: Optional[str] = None):
        if trace_file_path is None:
            # Default to project root .logs directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(project_root, ".logs")
            os.makedirs(log_dir, exist_ok=True)
            trace_file_path = os.path.join(log_dir, "agent_traces.jsonl")
        
        self.trace_file_path = trace_file_path
        self._current_span: Dict[str, Any] = {}

    def start_span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start a new tracing span."""
        self._current_span = {
            "span_name": name,
            "start_time": time.perf_counter(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "steps": [],
            "tokens": {"prompt": 0, "completion": 0, "total": 0}
        }

    def log_step(self, step_name: str, duration: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log an intermediate step (e.g. tool execution, single LLM call) within the current span."""
        if not self._current_span:
            return
        self._current_span["steps"].append({
            "step_name": step_name,
            "duration_seconds": round(duration, 4),
            "metadata": metadata or {}
        })

    def add_tokens(self, prompt: int, completion: int) -> None:
        """Accumulate token counts for the current span."""
        if not self._current_span:
            return
        t = self._current_span["tokens"]
        t["prompt"] += prompt
        t["completion"] += completion
        t["total"] = t["prompt"] + t["completion"]

    def end_span(self, status: str = "success", error_message: Optional[str] = None) -> None:
        """Close the current span and write the trace record to file."""
        if not self._current_span:
            return
        
        duration = time.perf_counter() - self._current_span.pop("start_time")
        self._current_span["duration_seconds"] = round(duration, 4)
        self._current_span["status"] = status
        if error_message:
            self._current_span["error"] = error_message

        try:
            with open(self.trace_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._current_span, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to write tracer span: %s", e)
        finally:
            self._current_span = {}
