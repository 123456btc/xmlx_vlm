# SPDX-License-Identifier: Apache-2.0
"""
Prometheus metrics for xmlx_vlm server.

Simplified from vllm-mlx metrics — keeps only HTTP and inference
instrumentation that works with the synchronous xmlx_vlm architecture.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


@dataclass
class InferenceTracker:
    """Request-scoped inference timing and token accounting."""

    collector: "MetricsCollector | None"
    endpoint: str
    stream: bool
    start_time: float = field(default_factory=time.perf_counter)
    _finished: bool = False
    _ttft_observed: bool = False

    def observe_ttft(self) -> None:
        if self.collector is None or self._ttft_observed:
            return
        self.collector.observe_ttft(
            endpoint=self.endpoint,
            stream=self.stream,
            value=time.perf_counter() - self.start_time,
        )
        self._ttft_observed = True

    def finish(
        self,
        *,
        result: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        if self.collector is None or self._finished:
            return
        self.collector.observe_inference(
            endpoint=self.endpoint,
            stream=self.stream,
            result=result,
            duration=time.perf_counter() - self.start_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._finished = True


class MetricsCollector:
    """Lazy Prometheus-backed metrics collector."""

    def __init__(self) -> None:
        self._enabled = False
        self._lock = threading.Lock()
        self._prom = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(self, *, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled or self._prom is not None:
                return
            self._init_prometheus()

    def _init_prometheus(self) -> None:
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Counter,
            Gauge,
            Histogram,
            generate_latest,
        )

        registry = CollectorRegistry(auto_describe=True)
        self._prom = {
            "registry": registry,
            "generate_latest": generate_latest,
            "content_type": CONTENT_TYPE_LATEST,
            "http_requests_total": Counter(
                "xmlx_vlm_http_requests_total",
                "HTTP requests handled by the server.",
                ["method", "path", "status_code"],
                registry=registry,
            ),
            "http_request_duration_seconds": Histogram(
                "xmlx_vlm_http_request_duration_seconds",
                "HTTP request latency in seconds.",
                ["method", "path"],
                registry=registry,
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            ),
            "http_requests_in_flight": Gauge(
                "xmlx_vlm_http_requests_in_flight",
                "HTTP requests currently in flight.",
                ["method", "path"],
                registry=registry,
            ),
            "inference_requests_total": Counter(
                "xmlx_vlm_inference_requests_total",
                "Inference requests completed by endpoint.",
                ["endpoint", "stream", "result"],
                registry=registry,
            ),
            "inference_request_duration_seconds": Histogram(
                "xmlx_vlm_inference_request_duration_seconds",
                "End-to-end inference latency in seconds.",
                ["endpoint", "stream"],
                registry=registry,
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            ),
            "inference_ttft_seconds": Histogram(
                "xmlx_vlm_inference_ttft_seconds",
                "Time to first token for streaming endpoints.",
                ["endpoint", "stream"],
                registry=registry,
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            ),
            "prompt_tokens_total": Counter(
                "xmlx_vlm_prompt_tokens_total",
                "Prompt/input tokens processed by endpoint.",
                ["endpoint", "stream"],
                registry=registry,
            ),
            "completion_tokens_total": Counter(
                "xmlx_vlm_completion_tokens_total",
                "Generated output tokens produced by endpoint.",
                ["endpoint", "stream"],
                registry=registry,
            ),
            "model_loaded": Gauge(
                "xmlx_vlm_model_loaded",
                "Whether a generation model is currently loaded.",
                registry=registry,
            ),
            "metal_memory_bytes": Gauge(
                "xmlx_vlm_metal_memory_bytes",
                "Metal memory usage in bytes.",
                ["kind"],
                registry=registry,
            ),
        }

    def track_inference(self, endpoint: str, *, stream: bool) -> InferenceTracker:
        if not self._enabled:
            return InferenceTracker(None, endpoint, stream)
        return InferenceTracker(self, endpoint, stream)

    def observe_http_start(self, *, method: str, path: str) -> None:
        if not self._enabled or self._prom is None:
            return
        self._prom["http_requests_in_flight"].labels(method=method, path=path).inc()

    def observe_http_finish(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration: float,
    ) -> None:
        if not self._enabled or self._prom is None:
            return
        self._prom["http_requests_in_flight"].labels(method=method, path=path).dec()
        self._prom["http_requests_total"].labels(
            method=method,
            path=path,
            status_code=str(status_code),
        ).inc()
        self._prom["http_request_duration_seconds"].labels(
            method=method,
            path=path,
        ).observe(duration)

    def observe_inference(
        self,
        *,
        endpoint: str,
        stream: bool,
        result: str,
        duration: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        if not self._enabled or self._prom is None:
            return
        stream_label = _bool_str(stream)
        self._prom["inference_requests_total"].labels(
            endpoint=endpoint,
            stream=stream_label,
            result=result,
        ).inc()
        self._prom["inference_request_duration_seconds"].labels(
            endpoint=endpoint,
            stream=stream_label,
        ).observe(duration)
        if prompt_tokens > 0:
            self._prom["prompt_tokens_total"].labels(
                endpoint=endpoint,
                stream=stream_label,
            ).inc(prompt_tokens)
        if completion_tokens > 0:
            self._prom["completion_tokens_total"].labels(
                endpoint=endpoint,
                stream=stream_label,
            ).inc(completion_tokens)

    def observe_ttft(self, *, endpoint: str, stream: bool, value: float) -> None:
        if not self._enabled or self._prom is None:
            return
        self._prom["inference_ttft_seconds"].labels(
            endpoint=endpoint,
            stream=_bool_str(stream),
        ).observe(value)

    def render_metrics(self) -> tuple[bytes, str]:
        if not self._enabled:
            raise RuntimeError("metrics_disabled")
        if self._prom is None:
            self._init_prometheus()
        return (
            self._prom["generate_latest"](self._prom["registry"]),
            self._prom["content_type"],
        )


metrics = MetricsCollector()
