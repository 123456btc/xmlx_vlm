"""
Admin / infrastructure endpoints:
  /v1/memory/*  /v1/embeddings  /v1/rerank
  /metrics      /models         /health
  /v1/cache/*   /unload
"""
import logging
import time
import traceback
from typing import Any, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..auth import verify_api_key
from ..model_store import _INHERIT_ADAPTER, get_store
from huggingface_hub import scan_cache_dir

from ..server_schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ModelsResponse,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from ..embedding_engine import EmbeddingEngine
from ..rerank_engine import RerankEngine
from ..config import get_idle_kv_release_timeout
from ..memory import get_memory_store
from ..metrics import metrics
from ..tool_parsers import _infer_tool_parser_from_processor
from ..engine import _infer_reasoning_parser
from ..version import __version__

logger = logging.getLogger("xmlx_vlm.routes.admin")

router = APIRouter()


@router.get("/v1/memory/status", response_model=None)
async def memory_status_endpoint(_=Depends(verify_api_key)):
    """Get memory store status."""
    store = get_memory_store()
    if store is None:
        return {"enabled": False}
    return {"enabled": True, **store.stats()}


@router.post("/v1/memory/search", response_model=None)
async def memory_search_endpoint(request: Request, _=Depends(verify_api_key)):
    """Search memories by query."""
    store = get_memory_store()
    if store is None:
        return {"error": "Memory store not enabled"}
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id")
    top_k = body.get("top_k")
    results = store.search(query, session_id=session_id, top_k=top_k)
    return {"results": results}


@router.post("/v1/memory/clear", response_model=None)
async def memory_clear_endpoint(request: Request, _=Depends(verify_api_key)):
    """Clear memories for a session or all memories."""
    store = get_memory_store()
    if store is None:
        return {"error": "Memory store not enabled"}
    body = await request.json()
    session_id = body.get("session_id")
    deleted = store.clear(session_id=session_id)
    return {"deleted": deleted, "session_id": session_id}


# ─── Embeddings endpoint ────────────────────────────────────────────────────


@router.post("/v1/embeddings", response_model=None)
async def embeddings_endpoint(request: EmbeddingRequest, _=Depends(verify_api_key)):
    embedding_engine = get_store().get_embedding_engine(request.model)
    texts = request.input if isinstance(request.input, list) else [request.input]
    vectors = embedding_engine.embed(texts)
    prompt_tokens = embedding_engine.count_tokens(texts)
    data = [
        EmbeddingData(embedding=vec, index=i)
        for i, vec in enumerate(vectors)
    ]
    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


# ─── Rerank endpoint ────────────────────────────────────────────────────────


@router.post("/v1/rerank", response_model=None)
async def rerank_endpoint(request: RerankRequest, _=Depends(verify_api_key)):
    rerank_engine = get_store().get_rerank_engine(request.model)
    docs: List[str] = []
    for d in request.documents:
        if isinstance(d, str):
            docs.append(d)
        else:
            docs.append(d.text)
    scores, total_tokens = rerank_engine.score_pairs(request.query, docs)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    if request.top_n is not None:
        indexed = indexed[: request.top_n]
    results = [
        RerankResult(
            index=idx,
            relevance_score=score,
            document=RerankDocument(text=docs[idx]),
        )
        for idx, score in indexed
    ]
    return RerankResponse(
        results=results,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


# ─── Prometheus metrics endpoint ────────────────────────────────────────────

@router.get("/metrics", response_model=None)
async def metrics_endpoint():
    if not metrics.enabled:
        raise HTTPException(status_code=503, detail="Metrics not enabled")
    try:
        data, content_type = metrics.render_metrics()
        return Response(content=data, media_type=content_type)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Metrics disabled")


@router.get("/models", response_model=ModelsResponse)
@router.get("/v1/models", response_model=ModelsResponse, include_in_schema=False)
def models_endpoint():
    """
    Return list of locally downloaded MLX models.
    """

    files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        return all(f in file_names for f in files)

    # Scan the cache directory for downloaded mlx models
    hf_cache_info = scan_cache_dir()
    downloaded_models = [repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)]

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]

    response = {"object": "list", "data": models}

    return response


@router.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    store = get_store()
    config = store.cache.get("config")
    text_config = getattr(config, "text_config", None)
    response_generator = store.response_generator

    return {
        "status": "healthy",
        "loaded_model": store.cache.get("model_path", None),
        "loaded_adapter": store.cache.get("adapter_path", None),
        "loaded_context_size": getattr(text_config, "max_position_embeddings", None),
        "loaded_tool_parser": (
            _infer_tool_parser_from_processor(store.cache.get("processor"))
            if store.cache.get("processor")
            else None
        ),
        "continuous_batching_enabled": response_generator is not None,
        "apc_enabled": store.apc_manager is not None,
        "idle_kv_release_timeout": get_idle_kv_release_timeout(),
        "idle_kv_released": (
            getattr(response_generator, "_idle_kv_released", False)
            if response_generator is not None
            else False
        ),
    }


@router.get("/v1/cache/stats")
@router.get("/cache/stats", include_in_schema=False)
async def apc_cache_stats():
    """Return Automatic Prefix Cache statistics (or ``enabled=false``)."""
    if get_store().apc_manager is None:
        return {"enabled": False}
    snap = get_store().apc_manager.stats_snapshot()
    snap["enabled"] = True
    return snap


@router.post("/v1/cache/reset")
@router.post("/cache/reset", include_in_schema=False)
async def apc_cache_reset():
    if get_store().apc_manager is None:
        return {"enabled": False}
    get_store().apc_manager.clear()
    return {"enabled": True, "status": "cleared"}


@router.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": get_store().cache.get("model_path", None),
        "adapter_name": get_store().cache.get("adapter_path", None),
    }

    if not get_store().unload():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": f"Model unloaded successfully",
        "unloaded": unloaded_info,
    }


