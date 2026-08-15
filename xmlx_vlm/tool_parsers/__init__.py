"""
Tool parser utilities for xmlx-vlm.

Re-exports mlx_lm's _infer_tool_parser with additional model support,
and loads parsers from both mlx_lm.tool_parsers and xmlx_vlm.tool_parsers.
"""

import importlib

from mlx_lm.tokenizer_utils import _infer_tool_parser as _mlx_lm_infer_tool_parser

# Additional patterns not covered by mlx_lm
_EXTRA_PATTERNS = [
    ("<|tool_call>", "gemma4"),
    ("<atem", "atem"),
    ("<atem:call", "atem"),
    ("atem_tool_call", "atem"),
    ("atem", "atem"),
]


def _infer_tool_parser(chat_template):
    """Infer tool parser type, checking mlx_lm patterns first then extras."""
    result = _mlx_lm_infer_tool_parser(chat_template)
    if result is not None:
        return result

    if not isinstance(chat_template, str):
        return None

    for marker, parser_type in _EXTRA_PATTERNS:
        if marker in chat_template:
            return parser_type

    return None


def _infer_tool_parser_from_processor(processor):
    """Infer tool parser type from processor's chat template."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return _infer_tool_parser(tokenizer.chat_template)

    return None


def load_tool_module(tool_parser_type):
    """Load a tool parser module from xmlx_vlm.tool_parsers or mlx_lm.tool_parsers."""
    if importlib.util.find_spec(f"xmlx_vlm.tool_parsers.{tool_parser_type}"):
        return importlib.import_module(f"xmlx_vlm.tool_parsers.{tool_parser_type}")
    return importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")


# Import extra vllm-mlx parsers so they register themselves
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)
from .atem_tool_parser import AtemToolParser
from .auto_tool_parser import AutoToolParser
from .deepseek_tool_parser import DeepSeekToolParser
from .functionary_tool_parser import FunctionaryToolParser
from .gemma4_tool_parser import Gemma4ToolParser
from .granite_tool_parser import GraniteToolParser
from .hermes_tool_parser import HermesToolParser
from .kimi_tool_parser import KimiToolParser
from .llama_tool_parser import LlamaToolParser
from .mistral_tool_parser import MistralToolParser
from .nemotron_tool_parser import NemotronToolParser
from .qwen_tool_parser import QwenToolParser
from .xlam_tool_parser import xLAMToolParser
from .glm47_tool_parser import Glm47ToolParser
from .harmony_tool_parser import HarmonyToolParser
from .minimax_tool_parser import MiniMaxToolParser


def get_parser_stop_tokens(
    parser_name: str | None,
    user_stops: list[str] | None,
) -> list[str]:
    """Merge user-supplied stops with parser-declared extras (deduped)."""
    stops = list(user_stops or [])
    if not parser_name:
        return stops
    try:
        parser_cls = ToolParserManager.get_tool_parser(parser_name)
    except (KeyError, ImportError):
        return stops
    for s in getattr(parser_cls, "extra_stop_tokens", []):
        if s not in stops:
            stops.append(s)
    return stops


__all__ = [
    "_infer_tool_parser",
    "_infer_tool_parser_from_processor",
    "load_tool_module",
    "get_parser_stop_tokens",
    "ToolParser",
    "ToolParserManager",
    "ExtractedToolCallInformation",
]
