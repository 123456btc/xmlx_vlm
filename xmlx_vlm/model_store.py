"""
ModelStore — centralised model state and lifecycle management.

Replaces the four module-level globals that littered server.py:
  response_generator, apc_manager, _embedding_engine, _rerank_engine, model_cache

Design
------
- A single ``ModelStore`` instance (``_store``) is created at import time.
- ``get_store()`` returns that singleton and is suitable for use as a
  FastAPI ``Depends()`` factory: ``store: ModelStore = Depends(get_store)``.
- ``ModelStore.get_or_load()`` replaces ``get_cached_model()``.
- ``ModelStore.unload()`` replaces ``unload_model_sync()``.
- Lazy-loaded embedding / rerank engines are encapsulated inside the store
  so callers never touch ``_embedding_engine`` or ``_rerank_engine`` directly.

Note on ``load_model_resources``
---------------------------------
This function raises ``HTTPException`` on failure because the route layer
expects it.  A future refactor should raise a plain exception here and
translate it at the HTTP boundary (in a route exception handler).
"""
import gc
import logging
import os
import traceback
from typing import Optional

import mlx.core as mx
from fastapi import HTTPException

from . import apc as _apc
from .config import (
    get_kv_group_size,
    get_kv_quant_scheme,
    get_quantized_kv_bits,
    get_quantized_kv_start,
    get_top_logprobs_k,
)
from .embedding_engine import EmbeddingEngine
from .engine import ResponseGenerator
from .moe_topk import apply_moe_top_k_override
from .patches import apply_all_patches
from .rerank_engine import RerankEngine
from .utils import load
from .vision_cache import VisionFeatureCache

logger = logging.getLogger("xmlx_vlm.model_store")

# Sentinel: caller wants to inherit the adapter that's already loaded
_INHERIT_ADAPTER = object()


# ---------------------------------------------------------------------------
# Weight loader (pure function, but raises HTTPException on failure)
# ---------------------------------------------------------------------------

def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """Load model weights + processor from disk.

    Raises ``HTTPException(500)`` on failure so the error propagates cleanly
    through FastAPI without leaking raw tracebacks to the client.
    """
    try:
        print(f"Loading model from: {model_path}")
        if adapter_path:
            print(f"Loading adapter from: {adapter_path}")
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path, adapter_path, trust_remote_code=trust_remote_code
        )
        config = model.config
        # Apply runtime patches for model-specific fixes (MTP, BatchKVCache, etc.)
        apply_all_patches()
        print("Model and processor loaded successfully.")
        moe_top_k = os.environ.get("MLX_MOE_TOP_K")
        if moe_top_k:
            try:
                apply_moe_top_k_override(model, int(moe_top_k))
            except Exception as e:
                logger.warning("MoE top-k override failed: %s", e)
        return model, processor, config
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------

class ModelStore:
    """Holds all mutable model state for one server instance.

    Instantiated once as a module-level singleton; accessed via
    ``get_store()``.  All state mutations go through the public methods so
    there are no ``global`` statements scattered across route handlers.
    """

    def __init__(self) -> None:
        self.response_generator: Optional[ResponseGenerator] = None
        self.apc_manager: Optional[_apc.APCManager] = None
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._rerank_engine: Optional[RerankEngine] = None
        self.cache: dict = {}  # was: model_cache

    # ── model lifecycle ──────────────────────────────────────────────────────

    def get_or_load(self, model_path: str, adapter_path=_INHERIT_ADAPTER):
        """Return (model, processor, config), loading if necessary.

        Mirrors the old ``get_cached_model()`` behaviour:
        - Returns cached resources when ``model_path`` + ``adapter_path``
          match what's already in memory.
        - Calls ``unload()`` first if a *different* model is cached.
        - Creates a new ``ResponseGenerator`` and ``APCManager`` on load.
        """
        if model_path == "default" and self.cache:
            model_path = self.cache.get("model_path", model_path)

        if adapter_path is _INHERIT_ADAPTER:
            cached_key = self.cache.get("cache_key")
            if cached_key and cached_key[0] == model_path:
                adapter_path = cached_key[1]
            else:
                adapter_path = None

        cache_key = (model_path, adapter_path)

        if self.cache.get("cache_key") == cache_key:
            logger.debug("Using cached model: %s  adapter: %s", model_path, adapter_path)
            return self.cache["model"], self.cache["processor"], self.cache["config"]

        if self.cache:
            logger.info("New model requested — unloading previous model first.")
            self.unload()

        vision_cache_size = int(os.environ.get("XMLX_VLM_VISION_CACHE_SIZE", "20"))
        vision_cache = VisionFeatureCache(max_size=vision_cache_size)

        self.apc_manager = _apc.from_env(model_namespace=model_path)

        kv_bits = get_quantized_kv_bits(model_path)
        kv_group_size = get_kv_group_size()
        quantized_kv_start = get_quantized_kv_start()
        kv_quant_scheme = get_kv_quant_scheme()

        self.response_generator = ResponseGenerator(
            model_path=model_path,
            model_loader=load_model_resources,
            adapter_path=adapter_path,
            vision_cache=vision_cache,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            kv_quant_scheme=kv_quant_scheme,
            quantized_kv_start=quantized_kv_start,
            top_logprobs_k=get_top_logprobs_k(),
            apc_manager=self.apc_manager,
        )
        try:
            model, processor, config = self.response_generator.wait_until_ready()
        except Exception:
            self.response_generator.stop_and_join()
            self.response_generator = None
            vision_cache.clear()
            raise

        self.cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": adapter_path,
            "model": model,
            "processor": processor,
            "config": config,
            "vision_cache": vision_cache,
        }
        return model, processor, config

    def unload(self) -> bool:
        """Stop the generator, free weights, clear caches.  Returns False if
        no model was loaded."""
        if not self.cache:
            return False

        logger.info(
            "Unloading model: %s  adapter: %s",
            self.cache.get("model_path"),
            self.cache.get("adapter_path"),
        )

        if self.response_generator is not None:
            logger.info("Stopping ResponseGenerator…")
            self.response_generator.stop_and_join()
            self.response_generator = None

        if self.apc_manager is not None:
            self.apc_manager.clear()
            self.apc_manager = None

        if "vision_cache" in self.cache:
            self.cache["vision_cache"].clear()

        self.cache = {}
        gc.collect()
        mx.clear_cache()
        logger.info("Model unloaded and cache cleared.")
        return True

    # ── guard helper ─────────────────────────────────────────────────────────

    def require_generator(self) -> ResponseGenerator:
        """Return the active ``ResponseGenerator`` or raise HTTP 503."""
        if self.response_generator is None:
            raise HTTPException(
                status_code=503,
                detail="No model is currently loaded. POST to /load first.",
            )
        return self.response_generator

    # ── lazy auxiliary engines ────────────────────────────────────────────────

    def get_embedding_engine(self, model_name: str) -> EmbeddingEngine:
        if self._embedding_engine is None or self._embedding_engine.model_name != model_name:
            self._embedding_engine = EmbeddingEngine(model_name)
        return self._embedding_engine

    def get_rerank_engine(self, model_name: str) -> RerankEngine:
        if self._rerank_engine is None or self._rerank_engine.model_name != model_name:
            self._rerank_engine = RerankEngine(model_name)
        return self._rerank_engine


# ---------------------------------------------------------------------------
# Module-level singleton + factory
# ---------------------------------------------------------------------------

_store = ModelStore()


def get_store() -> ModelStore:
    """FastAPI-compatible factory for the global ModelStore singleton.

    Usage::

        from .model_store import get_store

        @router.get("/health")
        async def health(store: ModelStore = Depends(get_store)):
            return {"loaded": store.cache.get("model_path")}
    """
    return _store
