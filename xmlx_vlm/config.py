"""
Server configuration constants and environment-variable getters.

All DEFAULT_* values are module-level constants that callers may reference
directly.  The matching ``get_server_*()``, ``get_kv_*()``, and
``get_*_timeout()`` helpers add env-var override logic on top.

Extracted from server.py as Phase 1 of the financial-software-grade split.
No heavy imports (no FastAPI, no MLX model objects) so this module loads fast
and is safe to import from any sub-package without circular deps.
"""
import gc
import logging
import os
from typing import Optional

import mlx.core as mx

from .generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
)

logger = logging.getLogger("xmlx_vlm.config")

# ---------------------------------------------------------------------------
# Per-request cache clearing (opt-in via env var)
# ---------------------------------------------------------------------------

_CLEAR_CACHE_PER_REQUEST = os.environ.get(
    "XMLX_VLM_CLEAR_CACHE_PER_REQUEST", "false"
).lower() in ("1", "true", "yes")


def _maybe_clear_cache() -> None:
    """Clear MLX cache + GC only when the user explicitly opts in.

    Continuous batching relies on KV-cache/APC reuse and generate.py already
    clears periodically every 256–512 tokens, so this defaults to off.
    """
    if _CLEAR_CACHE_PER_REQUEST:
        mx.clear_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Server-level defaults
# ---------------------------------------------------------------------------

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 5118
DEFAULT_TOKEN_QUEUE_TIMEOUT = 600.0
DEFAULT_FIRST_TOKEN_TIMEOUT = None  # No timeout for first token (prefill) by default

# When the generation thread has been idle for this many seconds, drop the
# active BatchGenerator prompt cache, flush APC, and clear the MLX cache pool.
# Set to 0 to disable automatic KV release.
DEFAULT_IDLE_KV_RELEASE_TIMEOUT = 300.0  # 5 minutes

DEFAULT_ENABLE_THINKING = True
DEFAULT_ENABLE_TOOL_LOGITS_BIAS = False

# Maximum number of pending requests in the ResponseGenerator queue.
# When the queue is full the server returns HTTP 503 rather than OOM-crashing.
# Override via XMLX_VLM_MAX_QUEUE_DEPTH or --max-queue-depth.
DEFAULT_MAX_QUEUE_DEPTH = 64

# Maximum number of concurrent generating sequences in continuous batching.
# Prevents unified memory OOM under multi-agent / burst workloads.
# Override via XMLX_VLM_MAX_NUM_SEQS or --max-num-seqs.
DEFAULT_MAX_NUM_SEQS = 16

# When a client request omits thinking_budget/reasoning_effort, this server-wide
# default caps thinking to avoid indefinitely-long reasoning that hangs agents
# (e.g. Pi.dev connected to many tool rounds).  Set to None to disable.
#
# Tuned for financial-quant / factor-research / agent-loop workloads:
#   • Multi-step math derivations and factor-tree construction need long chains
#   • Quant code generation benefits from thorough pre-flight reasoning
#   • Agent loop: each tool-call round re-reasons over results — budget must
#     cover a full analysis pass without cutting off mid-derivation
#   • 16 384 tokens ≈ 12 000 words of CoT — covers even deep Markowitz /
#     Kelly / alpha-decay derivations; model still exits early if it finishes
#     naturally, so short tasks are unaffected
DEFAULT_THINKING_BUDGET_CAP = 16384  # tokens; override via XMLX_DEFAULT_THINKING_BUDGET


# ---------------------------------------------------------------------------
# Timeout helpers
# ---------------------------------------------------------------------------

def _parse_timeout_env(env_name: str, default: Optional[float]) -> Optional[float]:
    raw_timeout = os.environ.get(env_name, "")
    if raw_timeout == "":
        return default
    try:
        timeout = float(raw_timeout)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %ss.",
            env_name,
            raw_timeout,
            default,
        )
        return default
    if timeout <= 0:
        return None
    return timeout


def get_token_queue_timeout() -> Optional[float]:
    return _parse_timeout_env("XMLX_VLM_TOKEN_QUEUE_TIMEOUT", DEFAULT_TOKEN_QUEUE_TIMEOUT)


def get_first_token_timeout() -> Optional[float]:
    return _parse_timeout_env("XMLX_VLM_FIRST_TOKEN_TIMEOUT", DEFAULT_FIRST_TOKEN_TIMEOUT)


def get_idle_kv_release_timeout() -> Optional[float]:
    """Return the idle duration (seconds) after which KV cache is auto-released.

    Reads ``XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT``.  Values <= 0 disable the
    mechanism.  ``None`` means disabled.
    """
    raw = os.environ.get("XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT", "")
    if raw.strip() == "":
        return DEFAULT_IDLE_KV_RELEASE_TIMEOUT
    try:
        timeout = float(raw.strip())
    except ValueError:
        logger.warning(
            "Invalid XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT=%r; falling back to %ss.",
            raw,
            DEFAULT_IDLE_KV_RELEASE_TIMEOUT,
        )
        return DEFAULT_IDLE_KV_RELEASE_TIMEOUT
    if timeout <= 0:
        return None
    return timeout


# ---------------------------------------------------------------------------
# Queue depth
# ---------------------------------------------------------------------------

def get_server_max_queue_depth() -> int:
    """Return the maximum number of pending requests in the ResponseGenerator queue.

    Override via ``XMLX_VLM_MAX_QUEUE_DEPTH`` env var or ``--max-queue-depth``
    CLI flag.  Set to 0 to make the queue unbounded (not recommended in
    production).
    """
    raw = os.environ.get("XMLX_VLM_MAX_QUEUE_DEPTH", "")
    if raw.strip():
        try:
            v = int(raw.strip())
            return max(0, v)
        except ValueError:
            pass
    return DEFAULT_MAX_QUEUE_DEPTH


# ---------------------------------------------------------------------------
# Thinking / reasoning toggles
# ---------------------------------------------------------------------------

def get_server_enable_thinking() -> bool:
    raw = os.environ.get("XMLX_VLM_ENABLE_THINKING")
    if raw is None:
        return DEFAULT_ENABLE_THINKING
    return raw.lower() in ("1", "true", "yes", "on")


def get_server_default_thinking_budget() -> Optional[int]:
    """Return the server-wide default thinking-token cap.

    Checked in order:
    1. ``XMLX_DEFAULT_THINKING_BUDGET`` env-var (int or "off"/"none"/"disabled" to disable)
    2. ``DEFAULT_THINKING_BUDGET_CAP`` module constant

    Returns ``None`` when the cap is explicitly disabled.
    """
    raw = os.environ.get("XMLX_DEFAULT_THINKING_BUDGET")
    if raw is not None:
        raw = raw.strip()
        if raw.lower() in ("off", "none", "disabled", "0", ""):
            return None
        try:
            return int(raw)
        except ValueError:
            pass  # fall through to default
    return DEFAULT_THINKING_BUDGET_CAP if DEFAULT_THINKING_BUDGET_CAP else None


# ---------------------------------------------------------------------------
# Tool-logits-bias toggle (global mutable, set once from CLI args)
# ---------------------------------------------------------------------------

_ENABLE_TOOL_LOGITS_BIAS: Optional[bool] = None


def get_server_enable_tool_logits_bias() -> bool:
    global _ENABLE_TOOL_LOGITS_BIAS
    if _ENABLE_TOOL_LOGITS_BIAS is not None:
        return _ENABLE_TOOL_LOGITS_BIAS
    env = os.environ.get("XMLX_VLM_ENABLE_TOOL_LOGITS_BIAS", "")
    return env.lower() in ("1", "true", "yes")


def set_server_enable_tool_logits_bias(value: bool) -> None:
    global _ENABLE_TOOL_LOGITS_BIAS
    _ENABLE_TOOL_LOGITS_BIAS = value


# ---------------------------------------------------------------------------
# KV-cache quantization
# ---------------------------------------------------------------------------

def get_quantized_kv_bits(model: str) -> Optional[float]:
    kv_bits = float(os.environ.get("KV_BITS", 0))
    if kv_bits == 0:
        return None
    if "qat" in model:
        print(f"Model {model} is quantization aware, KV cache will not be quantized.")
        return None
    return kv_bits


def get_kv_group_size() -> int:
    return int(os.environ.get("KV_GROUP_SIZE", DEFAULT_KV_GROUP_SIZE))


def get_kv_quant_scheme() -> str:
    return os.environ.get("KV_QUANT_SCHEME", DEFAULT_KV_QUANT_SCHEME)


def get_max_kv_size(model: str) -> Optional[int]:
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    if max_kv_tokens == 0:
        return None
    if get_quantized_kv_bits(model) is not None:
        print(f"Model {model} uses QuantizedKVCache, can't set max KV size.")
        return None
    return max_kv_tokens


def get_quantized_kv_start() -> int:
    return int(os.environ.get("QUANTIZED_KV_START", DEFAULT_QUANTIZED_KV_START))


def get_top_logprobs_k() -> int:
    """Max per-token top_logprobs honored by the server (0 = disabled).

    Set via TOP_LOGPROBS_K env var. OpenAI caps this at 20. When 0, requests
    with top_logprobs>0 still succeed but the top_logprobs list stays empty.
    """
    k = int(os.environ.get("TOP_LOGPROBS_K", 0))
    return max(0, min(k, 20))


def get_prefill_step_size() -> int:
    return int(os.environ.get("PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))


def get_serialize_requests() -> bool:
    """Return whether the server should serialize all requests (max concurrency = 1).

    Reads XMLX_VLM_SERIALIZE_REQUESTS env-var. Defaults to True on macOS (Darwin).
    """
    raw = os.environ.get("XMLX_VLM_SERIALIZE_REQUESTS")
    if raw is None:
        import platform
        return platform.system() == "Darwin"
    return raw.lower() in ("1", "true", "yes", "on")


def get_release_kv_after_request() -> bool:
    """Return whether the server should always release KV cache after every request.

    Reads XMLX_VLM_RELEASE_KV_AFTER_REQUEST env-var. Defaults to True on macOS (Darwin).
    """
    raw = os.environ.get("XMLX_VLM_RELEASE_KV_AFTER_REQUEST")
    if raw is None:
        import platform
        return platform.system() == "Darwin"
    return raw.lower() in ("1", "true", "yes", "on")


def get_max_num_seqs() -> int:
    """Return the maximum number of concurrent generating sequences in continuous batching.

    Reads ``XMLX_VLM_MAX_NUM_SEQS`` env var or ``--max-num-seqs`` CLI flag.
    Defaults to ``DEFAULT_MAX_NUM_SEQS`` (16).
    """
    raw = os.environ.get("XMLX_VLM_MAX_NUM_SEQS", "")
    if raw.strip():
        try:
            return max(1, int(raw.strip()))
        except ValueError:
            pass
    return DEFAULT_MAX_NUM_SEQS


def get_kv_bits_per_layer(
    num_layers: int,
    default_bits: Optional[float] = None,
    raw_config: Optional[str] = None,
) -> Optional[list[Optional[float]]]:
    """Parse and return per-layer KV cache quantization bit precision.

    Accepts:
    1. A raw config string or ``XMLX_VLM_KV_BITS_PER_LAYER`` / ``KV_BITS_PER_LAYER`` env var.
    2. Supported syntax formats:
       - "adaptive": Dynamically boosts first 2 & last 2 layers, reduces middle layers.
       - Comma-separated: "8,8,4,4,3.5,3.5,8,8"
       - Key-Value mapping: "0:8,1:8,-1:8,default:4" or "0-3:8,4-20:4,default:3.5"
         where "none", "fp16", or "0" indicates no quantization (FP16/BF16).

    Returns a list of length `num_layers`, where each element is a float (e.g. 4.0, 3.5, 8.0)
    or None (unquantized). Returns None if no quantization is requested.
    """
    if raw_config is None:
        raw_config = os.environ.get("XMLX_VLM_KV_BITS_PER_LAYER") or os.environ.get(
            "KV_BITS_PER_LAYER"
        )

    if not raw_config or not raw_config.strip():
        if default_bits is None or default_bits <= 0:
            return None
        return [default_bits] * num_layers

    raw = raw_config.strip()

    # Adaptive preset
    if raw.lower() == "adaptive":
        base_bits = default_bits if default_bits is not None and default_bits > 0 else 4.0
        bits_list: list[Optional[float]] = [base_bits] * num_layers
        if num_layers >= 6 and base_bits - 0.5 >= 1.0:
            M = num_layers - 4
            if M >= 4:
                indices_to_reduce = {2, 2 + M // 3, num_layers - 3 - M // 3, num_layers - 3}
                if len(indices_to_reduce) < 4:
                    indices_to_reduce = {2, 3, num_layers - 4, num_layers - 3}
                for idx in [0, 1, num_layers - 2, num_layers - 1]:
                    bits_list[idx] = base_bits + 0.5
                for idx in sorted(indices_to_reduce):
                    bits_list[idx] = base_bits - 0.5
            elif M >= 2:
                bits_list[0] = base_bits + 0.5
                bits_list[-1] = base_bits + 0.5
                bits_list[2] = base_bits - 0.5
                bits_list[num_layers - 3] = base_bits - 0.5
        return bits_list

    def parse_val(v: str) -> Optional[float]:
        v = v.strip().lower()
        if v in ("none", "fp16", "bf16", "0", "false", "null", "unquantized"):
            return None
        return float(v)

    # Check if mapping syntax (contains ":" or "=")
    if ":" in raw or "=" in raw:
        items = [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]
        mapping_default = default_bits
        layer_map: dict[int, Optional[float]] = {}

        for item in items:
            delim = ":" if ":" in item else "="
            k, v = item.split(delim, 1)
            k = k.strip().lower()
            val = parse_val(v)

            if k in ("default", "all", "*"):
                mapping_default = val
            elif "-" in k and not k.startswith("-"):  # Range e.g. "0-3"
                parts = k.split("-", 1)
                try:
                    start_l, end_l = int(parts[0]), int(parts[1])
                    for l_idx in range(start_l, end_l + 1):
                        if 0 <= l_idx < num_layers:
                            layer_map[l_idx] = val
                except ValueError:
                    pass
            else:
                try:
                    l_idx = int(k)
                    if l_idx < 0:
                        l_idx = num_layers + l_idx
                    if 0 <= l_idx < num_layers:
                        layer_map[l_idx] = val
                except ValueError:
                    pass

        result = [mapping_default] * num_layers
        for l_idx, val in layer_map.items():
            result[l_idx] = val
        return result

    # Comma-separated list of values
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if parts:
        parsed_vals = [parse_val(p) for p in parts]
        if len(parsed_vals) >= num_layers:
            return parsed_vals[:num_layers]
        # Pad with default_bits or last value
        pad_val = default_bits if default_bits is not None else parsed_vals[-1]
        return parsed_vals + [pad_val] * (num_layers - len(parsed_vals))

    return [default_bits] * num_layers if default_bits is not None else None

