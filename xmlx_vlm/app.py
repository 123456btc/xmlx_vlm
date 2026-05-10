"""
FastAPI application factory.

Creates and configures the ``app`` instance used by uvicorn.

Split from server.py (Phase 3) so that:
- Route modules can import ``app`` without circular deps
- ``server.py`` is reduced to CLI arg-parsing + ``uvicorn.run()``
- Tests can import ``app`` directly without launching a process

Startup / shutdown flow
-----------------------
``lifespan`` handles:
  1. Optional model pre-load (XMLX_VLM_PRELOAD_MODEL env var)
  2. External MCP server connections (MLX_MCP_CONFIG env var)
  3. APC disk flush + MCP disconnect on shutdown
"""
import logging
import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import verify_api_key  # noqa: F401 — re-exported for route imports
from .mcp import get_manager
from .model_store import get_store
from .prompt_warmup import load_warmup_file, warm_prompts
from .version import __version__

# Route sub-packages
from .routes.completions import router as completions_router
from .routes.responses import router as responses_router
from .routes.anthropic import router as anthropic_router
from .routes.mcp import router as mcp_router
from .routes.admin import router as admin_router

logger = logging.getLogger("xmlx_vlm.app")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    store = get_store()

    # Optional model pre-load
    model_path = os.environ.pop("XMLX_VLM_PRELOAD_MODEL", None)
    if model_path:
        adapter_path = os.environ.pop("XMLX_VLM_PRELOAD_ADAPTER", None)
        logger.info("Pre-loading model: %s", model_path)
        model, processor, config = store.get_or_load(model_path, adapter_path)
        kv_bits = os.environ.get("KV_BITS")
        kv_scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
        if kv_bits:
            logger.info(
                "KV cache quantization: bits=%s scheme=%s", kv_bits, kv_scheme
            )
        logger.info("Model ready, continuous batching enabled.")

        # Prompt warmup for coding assistants / RAG workloads
        warmup_path = os.environ.get("XMLX_VLM_WARMUP_PROMPTS")
        if warmup_path:
            try:
                prompts = load_warmup_file(warmup_path)
                warm_prompts(model, processor, prompts)
            except Exception as exc:
                logger.warning("Prompt warmup failed: %s", exc)

    # Connect external MCP servers
    mcp_config_path = os.environ.pop("MLX_MCP_CONFIG", None)
    if mcp_config_path:
        try:
            mcp_manager = get_manager()
            await mcp_manager.connect_external_servers(mcp_config_path)
        except Exception as exc:
            logger.warning("Failed to connect external MCP servers: %s", exc)

    yield  # ── server is running ─────────────────────────────────────────────

    # Flush APC blocks to disk so prefix caches survive restarts
    apc = store.apc_manager
    if apc is not None:
        try:
            n_flushed = apc.flush_to_disk()
            if n_flushed:
                logger.info(
                    "APC: flushed %d blocks to disk before shutdown", n_flushed
                )
        except Exception as exc:
            logger.warning("APC flush on shutdown failed: %s", exc)

    # Disconnect MCP servers
    try:
        mcp_manager = get_manager()
        await mcp_manager.disconnect_all()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MLX-VLM Inference API",
    description=(
        "API for using Vision Language Models (VLMs) and Omni Models "
        "(Vision, Audio and Video support) with MLX."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── server-identification middleware ────────────────────────────────────────
@app.middleware("http")
async def add_server_header(request, call_next):
    response = await call_next(request)
    response.headers["Server"] = f"xmlx_vlm/{__version__}"
    return response


# ── register domain routers ─────────────────────────────────────────────────
app.include_router(responses_router)
app.include_router(completions_router)
app.include_router(anthropic_router)
app.include_router(mcp_router)
app.include_router(admin_router)
