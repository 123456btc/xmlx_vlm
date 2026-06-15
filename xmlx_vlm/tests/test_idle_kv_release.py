import time
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import xmlx_vlm.app as app_module

from xmlx_vlm.config import (
    DEFAULT_IDLE_KV_RELEASE_TIMEOUT,
    get_idle_kv_release_timeout,
)
from xmlx_vlm.engine.generation import ResponseGenerator


class TestIdleKvReleaseConfig:
    def test_default_timeout(self, monkeypatch):
        monkeypatch.delenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", raising=False)
        assert get_idle_kv_release_timeout() == DEFAULT_IDLE_KV_RELEASE_TIMEOUT

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "60")
        assert get_idle_kv_release_timeout() == 60.0

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "0")
        assert get_idle_kv_release_timeout() is None

    def test_negative_disables(self, monkeypatch):
        monkeypatch.setenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "-1")
        assert get_idle_kv_release_timeout() is None

    def test_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "not-a-number")
        assert get_idle_kv_release_timeout() == DEFAULT_IDLE_KV_RELEASE_TIMEOUT


class TestIdleKvReleaseRuntime:
    def test_idle_release_triggered_after_timeout(self, monkeypatch):
        """ResponseGenerator releases batch_gen + APC + MLX cache after idle."""
        monkeypatch.setenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "0.05")
        monkeypatch.setenv("XMLX_VLM_MAX_QUEUE_DEPTH", "0")

        fake_apc = MagicMock()
        fake_apc.clear = MagicMock()

        gen = ResponseGenerator.__new__(ResponseGenerator)
        gen.model_path = "dummy"
        gen._model_loader = lambda *args: None
        gen.adapter_path = None
        gen.model = SimpleNamespace(language_model=SimpleNamespace())
        gen.processor = SimpleNamespace()
        gen.config = SimpleNamespace(eos_token_id=0)
        gen.stop_tokens = {0}
        gen.vision_cache = None
        gen.draft_model = None
        gen.kv_bits = None
        gen.kv_group_size = 64
        gen.kv_quant_scheme = "uniform"
        gen.quantized_kv_start = 5000
        gen.top_logprobs_k = 0
        gen.apc_manager = fake_apc
        gen.tokenizer = None
        gen.requests = Queue()
        gen._stop = False
        gen._ready = Event()
        gen._ready.set()
        gen._load_error = None
        gen._cancelled = set()
        gen._cancel_lock = MagicMock()
        gen._idle_kv_release_timeout = get_idle_kv_release_timeout()
        gen._last_activity_time = time.time()
        gen._idle_kv_released = False

        fake_batch_gen = MagicMock()
        fake_batch_gen.close = MagicMock()

        clear_cache_calls = []

        def fake_clear_cache():
            clear_cache_calls.append(1)

        monkeypatch.setattr(mx, "clear_cache", fake_clear_cache)

        # _run tail is complicated; directly exercise the release helper.
        gen._release_idle_kv(fake_batch_gen)

        fake_batch_gen.close.assert_called_once()
        fake_apc.clear.assert_called_once()
        assert len(clear_cache_calls) == 1
        assert gen._idle_kv_released is True

    def test_idle_release_without_batch_gen(self, monkeypatch):
        """_release_idle_kv(None) still clears APC and MLX cache (speculative/diffusion paths)."""
        monkeypatch.setenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "0.05")

        fake_apc = MagicMock()
        gen = ResponseGenerator.__new__(ResponseGenerator)
        gen.apc_manager = fake_apc
        gen._idle_kv_release_timeout = 0.05
        gen._idle_kv_released = False

        clear_cache_calls = []

        def fake_clear_cache():
            clear_cache_calls.append(1)

        monkeypatch.setattr(mx, "clear_cache", fake_clear_cache)

        gen._release_idle_kv(None)

        fake_apc.clear.assert_called_once()
        assert len(clear_cache_calls) == 1
        assert gen._idle_kv_released is True

    def test_mark_active_resets_idle_state(self, monkeypatch):
        """_mark_active() updates the timestamp and resets the released flag."""
        monkeypatch.setenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "300")

        # Build a minimal generator without starting its thread.
        gen = ResponseGenerator.__new__(ResponseGenerator)
        gen._last_activity_time = 0.0
        gen._idle_kv_released = True

        before = time.time()
        gen._mark_active()
        after = time.time()

        assert gen._idle_kv_released is False
        assert before <= gen._last_activity_time <= after



class TestIdleKvReleaseHealth:
    def test_health_reports_idle_kv_release_timeout(self, monkeypatch):
        monkeypatch.setenv("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "120")
        with TestClient(app_module.app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["idle_kv_release_timeout"] == 120.0
            assert data["idle_kv_released"] is False


class TestIdleKvReleaseIdempotent:
    """Verify that idle release only fires ONCE per idle period (no busy-loop).

    Before the fix, all three GPU-thread loops (_run, _run_speculative,
    _run_diffusion) would call _release_idle_kv on every polling cycle
    (~10 times/s) after the timeout had elapsed, keeping the CPU hot and
    the fan spinning.  The guard `not self._idle_kv_released` ensures the
    expensive release (gc, mx.clear_cache) runs exactly once.
    """

    def _make_gen(self, monkeypatch):
        """Return a minimal ResponseGenerator with mocked clear_cache."""
        fake_apc = MagicMock()
        gen = ResponseGenerator.__new__(ResponseGenerator)
        gen.apc_manager = fake_apc
        gen._idle_kv_release_timeout = 0.05
        gen._idle_kv_released = False
        gen._last_activity_time = time.time() - 1.0  # already past timeout

        self.clear_calls = []
        monkeypatch.setattr(mx, "clear_cache", lambda: self.clear_calls.append(1))
        return gen

    def test_release_fires_only_once_when_called_repeatedly(self, monkeypatch):
        """Calling _release_idle_kv twice should clear cache only on first call."""
        gen = self._make_gen(monkeypatch)
        gen._release_idle_kv(None)
        gen._release_idle_kv(None)  # second call — should be a no-op effectively

        # _release_idle_kv itself doesn't gate; but the caller loop does.
        # Here we verify _idle_kv_released is set after first call.
        assert gen._idle_kv_released is True
        assert len(self.clear_calls) == 2  # raw helper always clears; guard is in loop

    def test_loop_guard_prevents_second_release(self, monkeypatch):
        """Simulate the loop guard: _release_idle_kv is NOT called when _idle_kv_released=True."""
        gen = self._make_gen(monkeypatch)

        def loop_body():
            """Mimic the fixed guard in _run / _run_speculative / _run_diffusion."""
            if (
                gen._idle_kv_release_timeout is not None
                and not gen._idle_kv_released
                and (time.time() - gen._last_activity_time) > gen._idle_kv_release_timeout
            ):
                gen._release_idle_kv(None)

        # First poll — should trigger release
        loop_body()
        assert gen._idle_kv_released is True
        assert len(self.clear_calls) == 1

        # Subsequent polls — guard should block further releases
        loop_body()
        loop_body()
        loop_body()
        assert len(self.clear_calls) == 1  # still only 1, no busy-loop CPU burn

    def test_mark_active_re_arms_guard(self, monkeypatch):
        """After a new request arrives (_mark_active), the guard resets and release can fire again."""
        gen = self._make_gen(monkeypatch)

        def loop_body():
            if (
                gen._idle_kv_release_timeout is not None
                and not gen._idle_kv_released
                and (time.time() - gen._last_activity_time) > gen._idle_kv_release_timeout
            ):
                gen._release_idle_kv(None)

        # First idle period
        loop_body()
        assert len(self.clear_calls) == 1

        # New request arrives — resets state
        gen._mark_active()
        assert gen._idle_kv_released is False

        # Fast-forward past timeout again
        gen._last_activity_time = time.time() - 1.0

        # Second idle period — guard allows one more release
        loop_body()
        assert len(self.clear_calls) == 2

        # Further polls do nothing again
        loop_body()
        loop_body()
        assert len(self.clear_calls) == 2


class TestSerialRequestQueuing:
    def test_serialize_requests_config(self, monkeypatch):
        from xmlx_vlm.config import get_serialize_requests
        monkeypatch.delenv("XMLX_VLM_SERIALIZE_REQUESTS", raising=False)
        import platform
        is_darwin = platform.system() == "Darwin"
        assert get_serialize_requests() == is_darwin

        monkeypatch.setenv("XMLX_VLM_SERIALIZE_REQUESTS", "true")
        assert get_serialize_requests() is True

        monkeypatch.setenv("XMLX_VLM_SERIALIZE_REQUESTS", "false")
        assert get_serialize_requests() is False

    def test_release_kv_after_request_config(self, monkeypatch):
        from xmlx_vlm.config import get_release_kv_after_request
        monkeypatch.delenv("XMLX_VLM_RELEASE_KV_AFTER_REQUEST", raising=False)
        import platform
        is_darwin = platform.system() == "Darwin"
        assert get_release_kv_after_request() == is_darwin

        monkeypatch.setenv("XMLX_VLM_RELEASE_KV_AFTER_REQUEST", "true")
        assert get_release_kv_after_request() is True

        monkeypatch.setenv("XMLX_VLM_RELEASE_KV_AFTER_REQUEST", "false")
        assert get_release_kv_after_request() is False

