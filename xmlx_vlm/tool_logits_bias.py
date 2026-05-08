# SPDX-License-Identifier: Apache-2.0
"""Jump-forward decoding via tool-logits bias.

When a model is configured for tool calling, bias the logits of structured
output tokens (<tool_call>, {, "name", etc.) to accelerate entry into the
tool-call format. This cuts latency for the first tool token without
changing sampling parameters.

Inspired by Rapid-MLX's --enable-tool-logits-bias.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

# Default markers that indicate the start of a tool-call span.
# Bias is applied to the token IDs of these strings.
_DEFAULT_TOOL_MARKERS = [
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "{",
    "}",
    '",',
    '":',
    '"name"',
    '"arguments"',
    "name",
    "arguments",
    "[TOOL_CALLS]",
    "<|tool_call>",
    "<tool_call|>",
    "<invoke>",
    "</invoke>",
]


class ToolLogitsBiasProcessor:
    """Logits processor that adds a positive bias to tool-related tokens.

    Args:
        tokenizer: The model tokenizer (must support ``encode``).
        bias: Additive bias value (default 3.0). Higher = stronger push.
        markers: List of string markers to bias. Defaults to common tool tags.
    """

    def __init__(
        self,
        tokenizer: Any,
        bias: float = 3.0,
        markers: Optional[list[str]] = None,
    ) -> None:
        self.bias = bias
        self.token_ids: set[int] = set()
        markers = markers or _DEFAULT_TOOL_MARKERS
        for marker in markers:
            try:
                ids = tokenizer.encode(marker, add_special_tokens=False)
                if isinstance(ids, list):
                    self.token_ids.update(ids)
                elif hasattr(ids, "tolist"):
                    self.token_ids.update(ids.tolist())
            except Exception:
                pass
        if self.token_ids:
            logger.debug(
                "ToolLogitsBiasProcessor: biased %d token IDs (bias=%.1f)",
                len(self.token_ids),
                bias,
            )
        else:
            logger.warning(
                "ToolLogitsBiasProcessor: no token IDs resolved from markers"
            )

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if not self.token_ids:
            return logits
        # Handle both 1-D (vocab,) and 2-D (batch, vocab) logits shapes
        for tid in self.token_ids:
            if logits.ndim == 1:
                logits = logits.at[tid].add(self.bias)
            else:
                logits = logits.at[..., tid].add(self.bias)
        return logits
