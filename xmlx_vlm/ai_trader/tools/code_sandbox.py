"""PTC (Programmatic Tool Calling) 代码沙箱执行工具.

允许 AI Agent 生成并执行 Python 代码以编程方式编排多个工具，
避免多次往返对话带来的巨大延迟与 Token 浪费。
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
import sys
import time
import traceback
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.sdk.client import TraderSDK

logger = logging.getLogger(__name__)


def sanitize_traceback(text: str) -> str:
    """脱敏私钥与敏感地址信息."""
    if not isinstance(text, str):
        return str(text)
    # Mask private keys
    text = re.sub(r"\b(0x)?[a-fA-F0-9]{64}\b", "[REDACTED_KEY]", text)
    # Mask addresses
    text = re.sub(r"\b0x[a-fA-F0-9]{40}\b", "[REDACTED_ADDR]", text)
    return text


class ExecuteCodeTool:
    """PTC 模式的 Python 代码执行工具."""

    name = "execute_code"
    description = (
        "以程序化方式 (Programmatic Tool Calling / PTC 模式) 执行 Python 脚本。"
        "沙箱环境预置了 `sdk` (TraderSDK 实例，包含 sdk.market 和 sdk.oms)、`math`、`json` 等标准库。"
        "可以通过 `print()` 打印信息，或将最终分析/提案结果赋值给 `result` 变量。"
        "适合进行多币种批量指标筛选、相关性对比、复杂仓位测算等需要组合多个工具操作的高级任务。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码块",
            }
        },
        "required": ["code"],
    }

    def __init__(self, sdk: Optional[TraderSDK] = None, timeout_seconds: float = 10.0):
        self._sdk = sdk or TraderSDK()
        self._timeout_seconds = timeout_seconds

    def run(self, code: str, **kwargs) -> str:
        """同步执行 Python 代码并捕获输出."""
        if not code or not code.strip():
            return "错误：代码内容为空"

        # 去除 markdown 包装（如果模型输出带 ```python ... ```）
        clean_code = code.strip()
        if clean_code.startswith("```"):
            lines = clean_code.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_code = "\n".join(lines).strip()

        # 构建沙箱全局与局部命名空间
        sandbox_globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "sdk": self._sdk,
            "math": math,
            "json": json,
            "time": time,
        }
        try:
            import numpy as np
            sandbox_globals["np"] = np
        except ImportError:
            pass

        sandbox_locals: Dict[str, Any] = {}

        # 重定向标准输出与错误
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_stdout = io.StringIO()
        redirected_stderr = io.StringIO()

        start_time = time.time()
        try:
            sys.stdout = redirected_stdout
            sys.stderr = redirected_stderr

            # 执行代码
            compiled = compile(clean_code, "<ptc_sandbox>", "exec")
            exec(compiled, sandbox_globals, sandbox_locals)

            elapsed = round((time.time() - start_time) * 1000, 2)
            stdout_str = redirected_stdout.getvalue()
            stderr_str = redirected_stderr.getvalue()

            # 提取结果
            result_val = sandbox_locals.get("result")
            output_parts = []
            if stdout_str:
                output_parts.append(f"[Output]:\n{stdout_str.strip()}")
            if stderr_str:
                output_parts.append(f"[Stderr]:\n{stderr_str.strip()}")
            if result_val is not None:
                if isinstance(result_val, (dict, list)):
                    formatted = json.dumps(result_val, ensure_ascii=False, indent=2)
                else:
                    formatted = str(result_val)
                output_parts.append(f"[Result]:\n{formatted}")

            if not output_parts:
                return f"[Success]: 代码执行成功 (耗时 {elapsed}ms)，无任何标准输出或 result 赋值。"

            return f"[Execution Success ({elapsed}ms)]\n" + "\n\n".join(output_parts)

        except Exception as exc:
            tb = traceback.format_exc()
            sanitized = sanitize_traceback(tb)
            logger.warning("PTC 代码执行异常: %s", exc)
            return f"[Execution Error]: {type(exc).__name__}: {exc}\nTraceback:\n{sanitized}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
