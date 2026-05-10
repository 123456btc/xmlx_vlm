"""
CLI entry-point for xmlx_vlm server.

All FastAPI application logic lives in ``app.py``.
Route handlers live in ``routes/``.
Model lifecycle lives in ``model_store.py``.

This module only:
  - Parses CLI arguments
  - Sets env vars consumed by config.py getters
  - Calls ``uvicorn.run("xmlx_vlm.app:app", ...)``

Backward compatibility
----------------------
``from xmlx_vlm.server import app`` still works via the re-export below.
"""
import argparse
import logging
import os

import uvicorn

from .app import app  # re-export for backward compat (uvicorn xmlx_vlm.server:app)
from .config import (
    DEFAULT_ENABLE_THINKING,
    DEFAULT_ENABLE_TOOL_LOGITS_BIAS,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_QUEUE_DEPTH,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_THINKING_BUDGET_CAP,
)
from .generate import DEFAULT_MAX_TOKENS
from .server_schemas import get_server_max_tokens

logger = logging.getLogger("xmlx_vlm.server")

def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the HTTP server (default:0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pre-load a model at startup (e.g. mlx-community/Qwen2.5-VL-3B-Instruct-4bit).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Adapter weights to load with the model.",
    )
    parser.add_argument(
        "--vision-cache-size",
        type=int,
        default=20,
        help="Max number of cached vision features (default: 20).",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Tokens per prefill step (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_server_max_tokens(),
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help=(
            "Enable thinking mode by default for requests that do not set "
            "enable_thinking explicitly."
        ),
    )
    parser.add_argument(
        "--default-thinking-budget",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Server-wide thinking-token cap applied when a client request omits "
            "thinking_budget and reasoning_effort.  Prevents infinite reasoning "
            "loops in long multi-tool conversations (e.g. after 10+ tool calls). "
            "Set to 0 to disable the cap (use with caution). "
            f"Default: {DEFAULT_THINKING_BUDGET_CAP} tokens (also overridable via "
            "XMLX_DEFAULT_THINKING_BUDGET env var)."
        ),
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits for KV cache quantization (e.g. 3.5 for TurboQuant).",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size in tokens.",
    )
    parser.add_argument(
        "--max-queue-depth",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of pending requests in the GPU generation queue. "
            "When full the server returns an error instead of OOM-crashing. "
            f"Default: {DEFAULT_MAX_QUEUE_DEPTH}. Set to 0 for unbounded (not "
            "recommended). Also overridable via XMLX_VLM_MAX_QUEUE_DEPTH."
        ),
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for quantized KV cache.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=(
            "Speculative drafter path or HF id "
            "(e.g. z-lab/Qwen3.5-4B-DFlash, google/gemma-4-31B-it-assistant)."
        ),
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "mtp"],
        help="Drafter family — 'dflash' or 'mtp' (Gemma 4). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--top-logprobs-k",
        type=int,
        default=None,
        help=(
            "Server-side cap for per-token top_logprobs (0-20, default 0 = "
            "disabled). Maps to the TOP_LOGPROBS_K env var."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication. If set, clients must provide it via Authorization: Bearer <key> header.",
    )
    parser.add_argument(
        "--apc-enabled",
        action="store_true",
        default=False,
        help="Enable Automatic Prefix Caching (APC) for KV cache reuse across requests.",
    )
    parser.add_argument(
        "--ssd-cache-dir",
        type=str,
        default=None,
        help="Directory for APC SSD-tiered cache (persists prefix cache to disk).",
    )
    parser.add_argument(
        "--ssd-cache-max-gb",
        type=float,
        default=0,
        help="Max SSD cache size in GB (0 = unbounded).",
    )
    parser.add_argument(
        "--moe-top-k",
        type=int,
        default=None,
        help="Override MoE top_k per token (e.g. 4 for Qwen3 MoE). Must be <= trained top_k.",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP config file (JSON/YAML) for external MCP servers.",
    )
    parser.add_argument(
        "--enable-tool-logits-bias",
        action="store_true",
        default=DEFAULT_ENABLE_TOOL_LOGITS_BIAS,
        help=(
            "Bias logits toward structured tool-call tokens (e.g. <tool_call>, {, "
            '\"name\") to accelerate tool-call generation. Similar to Rapid-MLX\'s '
            "jump-forward decoding."
        ),
    )
    args = parser.parse_args()
    global _API_KEY
    _API_KEY = args.api_key
    if _API_KEY:
        os.environ["XMLX_VLM_API_KEY"] = _API_KEY
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["XMLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["XMLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    os.environ["XMLX_VLM_VISION_CACHE_SIZE"] = str(args.vision_cache_size)
    if args.draft_model:
        os.environ["XMLX_VLM_DRAFT_MODEL"] = args.draft_model
        if args.draft_kind is not None:
            os.environ["XMLX_VLM_DRAFT_KIND"] = args.draft_kind
        if args.draft_block_size is not None:
            os.environ["XMLX_VLM_DRAFT_BLOCK_SIZE"] = str(args.draft_block_size)
    if args.prefill_step_size:
        os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    os.environ["XMLX_VLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["XMLX_VLM_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"
    if args.default_thinking_budget is not None:
        # 0 means "disable cap"; env var "0" is caught by get_server_default_thinking_budget
        os.environ["XMLX_DEFAULT_THINKING_BUDGET"] = str(args.default_thinking_budget)
    if args.kv_bits is not None:
        os.environ["KV_BITS"] = str(args.kv_bits)
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    if args.max_kv_size is not None:
        os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    if args.top_logprobs_k is not None:
        os.environ["TOP_LOGPROBS_K"] = str(args.top_logprobs_k)
    if args.apc_enabled:
        os.environ["APC_ENABLED"] = "1"
    if args.ssd_cache_dir:
        os.environ["APC_DISK_PATH"] = args.ssd_cache_dir
    if args.ssd_cache_max_gb > 0:
        os.environ["APC_DISK_MAX_GB"] = str(args.ssd_cache_max_gb)
    if args.moe_top_k is not None:
        os.environ["MLX_MOE_TOP_K"] = str(args.moe_top_k)
    if args.mcp_config:
        os.environ["MLX_MCP_CONFIG"] = args.mcp_config
    os.environ["XMLX_VLM_ENABLE_TOOL_LOGITS_BIAS"] = "1" if args.enable_tool_logits_bias else "0"
    if args.max_queue_depth is not None:
        os.environ["XMLX_VLM_MAX_QUEUE_DEPTH"] = str(args.max_queue_depth)

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)

    uvicorn.run(
        "xmlx_vlm.app:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        server_header=False,
    )


if __name__ == "__main__":
    main()
