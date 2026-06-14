"""AI Trader CLI —— 聊天即交易.

默认连接 service.sh 启动的本地服务 (http://localhost:8080)，
复用其加载的模型与工具调用能力。也支持 --local 本地加载模型.

用法:
    # 连接已运行的 service.sh 服务（推荐）
    xmlx_vlm.ai-trader

    # 本地加载模型
    xmlx_vlm.ai-trader --local --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit

    # 非交互模式执行一句指令
    xmlx_vlm.ai-trader --prompt "BTC 现在多少钱？"
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# 确保项目根目录在路径中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from xmlx_vlm import load
from xmlx_vlm.generate import stream_generate
from xmlx_vlm.prompt_utils import apply_chat_template
from xmlx_vlm.vision_cache import VisionFeatureCache

from xmlx_vlm.ai_trader.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)
console = Console()

# 支持两种工具调用格式：
# 1. JSON: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
# 2. XML: <tool_call>\n<function=func>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
JSON_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
XML_FUNCTION_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
XML_PARAM_PATTERN = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)
# 部分模型输出格式: <|tool_call>call:market_data{action:get_ticker,symbol:BTC/USDT}<tool_call|>
CALL_BRACE_PATTERN = re.compile(
    r"call:\s*(\w+)\s*\{([^}]*)\}", re.DOTALL
)


DEFAULT_SYSTEM_PROMPT = """你是 AI Trader，一个本地运行的 AI 量化交易助手。用户通过自然语言与你对话，你可以调用工具完成行情分析、K 线图绘制、模拟交易等操作。

所有行情分析统一使用 Hyperliquid 数据源。

当前可用工具：
{tools_schema}

机构级数据能力：
- L1 行情：最新价、24h 高低点、涨跌幅、成交量
- 多周期技术分析：默认同时看 5m（短线情绪）、15m（中短线结构）、1h（趋势结构），并给出多空共振/分歧结论
- L2 订单簿：买卖深度、spread、深度失衡（depth imbalance）、VWAP
- 逐笔成交：主动买卖压力、大单识别
- 资金费率 / 持仓量
- 综合市场摘要：一键获取价格、成交量、OI、funding、盘口失衡、成交流

分析原则：
- 判断趋势或给出交易建议前，优先调用 get_multi_timeframe_summary 获取 5m/15m/1h 三周期结构，避免只看单一周期。

规则：
1. 当需要查行情、画图、下单时，必须输出工具调用。可以使用以下任一格式：

格式 A（推荐）：
<tool_call>
<function=工具名>
<parameter=参数名>参数值</parameter>
</function>
</tool_call>

格式 B：
<tool_call>{{"name": "工具名", "arguments": {{"参数名": "值"}}}}</tool_call>

格式 C（部分模型输出）：
<|tool_call>call:工具名{{参数名:值,参数名:值}}<tool_call|>

2. 你可以在一次回复中连续调用多个工具。
3. 工具结果会返回给你，你根据结果用中文向用户解释。
4. 默认所有下单都是 paper（纸盘模拟），不会动用真实资金。
5. 涉及交易时，务必先做风控说明，并明确告知用户这是模拟还是实盘。
6. 如果用户要求急停，调用 trading action=emergency_stop。
7. 交易对示例：BTC/USDC、ETH/USDC；调用 market_data 时无需指定 exchange。

请保持专业、简洁。"""


def build_system_prompt(registry: ToolRegistry) -> str:
    tools = registry.list_tools()
    schema_text = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']} 参数: {json.dumps(t['function']['parameters'], ensure_ascii=False)}"
        for t in tools
    )
    return DEFAULT_SYSTEM_PROMPT.format(tools_schema=schema_text)


def _parse_call_brace_arguments(body: str) -> Dict[str, Any]:
    """解析 action:get_ticker,symbol:BTC/USDT 风格参数."""
    arguments: Dict[str, Any] = {}
    if not body.strip():
        return arguments
    for part in re.split(r",\s*(?=[\w_]+\s*:)", body.strip()):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"\'')
        if not key:
            continue
        # 尝试数字/布尔/null
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "null":
            value = None
        else:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        arguments[key] = value
    return arguments


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """从模型输出中解析工具调用，支持 JSON / XML / call: 三种格式."""
    calls = []

    # 先尝试 XML function 格式
    for match in XML_FUNCTION_PATTERN.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        arguments: Dict[str, Any] = {}
        for pmatch in XML_PARAM_PATTERN.finditer(body):
            key = pmatch.group(1).strip()
            value = pmatch.group(2).strip()
            # 尝试解析为 JSON，失败则保留字符串
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            arguments[key] = value
        calls.append({"name": name, "arguments": arguments})

    # 再尝试 JSON 格式
    for match in JSON_TOOL_CALL_PATTERN.finditer(text):
        try:
            data = json.loads(match.group(1))
            if "name" in data:
                calls.append(data)
        except json.JSONDecodeError:
            continue

    # 兼容 call:tool_name{...} 格式
    for match in CALL_BRACE_PATTERN.finditer(text):
        name = match.group(1).strip()
        arguments = _parse_call_brace_arguments(match.group(2))
        calls.append({"name": name, "arguments": arguments})

    return calls


def remove_tool_calls(text: str) -> str:
    """去掉模型输出中的工具调用标签，保留解释文本."""
    text = XML_FUNCTION_PATTERN.sub("", text)
    text = JSON_TOOL_CALL_PATTERN.sub("", text)
    text = CALL_BRACE_PATTERN.sub("", text)
    return text.strip()


def encode_image_to_base64(image_path: str) -> str:
    """把图片编码为 base64 data URL."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


class AITraderChat:
    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        local_only: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key or os.getenv("XMLX_VLM_API_KEY") or "x123456"
        self.model_path = model_path or "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        self.local_only = local_only
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.registry = ToolRegistry()
        self.history: List[Dict[str, Any]] = []
        self.use_server = False
        self.server_model: Optional[str] = None
        self.model = None
        self.processor = None
        self.vision_cache = VisionFeatureCache()
        self.prompt_cache_state = None
        self.current_image_path: Optional[str] = None

    def _probe_server(self) -> bool:
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = requests.get(
                f"{self.server_url}/health", headers=headers, timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                loaded = data.get("loaded_model")
                if loaded:
                    self.server_model = loaded
                    rprint(
                        f"[bold green]已连接到本地服务:[/bold green] {self.server_url} (model: {loaded})"
                    )
                    return True
        except Exception:
            pass
        return False

    def load_model(self):
        if self.local_only:
            self._load_local_model()
            return

        if self._probe_server():
            self.use_server = True
            return

        rprint(
            f"[bold yellow]未检测到 {self.server_url} 上的服务，将本地加载模型...[/bold yellow]"
        )
        self._load_local_model()

    def _load_local_model(self):
        with console.status(f"[bold green]加载本地模型 {self.model_path}..."):
            self.model, self.processor = load(self.model_path)
        rprint(f"[bold green]本地模型加载完成:[/bold green] {self.model_path}")

    def print_help(self):
        mode = "server" if self.use_server else "local"
        help_text = f"""
[bold yellow]AI Trader 命令:[/bold yellow]
• /image <path> — 加载图片让模型分析
• /clear — 清空对话
• /help — 显示帮助
• /exit — 退出
• 直接输入自然语言，例如：
  - "BTC 现在多少钱？"
  - "画一张 BTC 1小时 K 线图"
  - "分析一下 BTC 走势"
  - "模拟买入 0.01 BTC"
  - "急停"

[dim]当前模式: {mode}[/dim]
        """
        rprint(Panel(help_text, title="AI Trader", border_style="blue"))

    def _add_message(self, role: str, content: Any):
        if isinstance(content, str):
            self.history.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            self.history.append({"role": role, "content": content})
        else:
            self.history.append({"role": role, "content": [{"type": "text", "text": str(content)}]})

    def _build_messages(self) -> List[Dict[str, Any]]:
        """转换为 OpenAI 兼容消息格式."""
        messages = []
        for msg in self.history:
            role = msg["role"]
            content_parts = msg.get("content", [])
            if not content_parts:
                continue

            # 文本部分
            text_parts = [
                p.get("text", "") for p in content_parts
                if isinstance(p, dict) and "text" in p
            ]
            text = "\n".join(text_parts)

            # 图片部分
            images = [
                p for p in content_parts
                if isinstance(p, dict) and ("image" in p or p.get("type") == "image_url")
            ]

            if images and role == "user":
                content: Any = [{"type": "text", "text": text}]
                for img in images:
                    if img.get("type") == "image_url":
                        content.append(img)
                    elif "path" in img:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": encode_image_to_base64(img["path"])},
                        })
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": text})
        return messages

    def _generate_server(self) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = self._build_messages()
        payload = {
            "model": self.server_model or "default",
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools": self.registry.list_tools(),
        }

        text = ""
        with requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=600,
        ) as resp:
            if resp.status_code == 401:
                rprint(
                    "[bold red]API Key 校验失败。[/bold red]"
                )
                rprint(
                    "[dim]service.sh 默认 key 是 x123456；如果你在启动服务时修改了 XMLX_VLM_API_KEY，"
                    "请在启动 ai-trader 时传入相同的 key：[/dim]"
                )
                rprint(
                    f"[dim]  XMLX_VLM_API_KEY=<你的key> xmlx_vlm.ai-trader --server-url {self.server_url}[/dim]"
                )
                raise SystemExit(1)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].decode("utf-8")
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        text += content
                except json.JSONDecodeError:
                    continue
        return text

    def _generate_local(self) -> str:
        chat_template_kwargs = {"tools": self.registry.list_tools()}
        num_images = 1 if self.current_image_path else 0
        image = [self.current_image_path] if self.current_image_path else None

        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            self.history,
            num_images=num_images,
            **chat_template_kwargs,
        )

        text = ""
        for chunk in stream_generate(
            self.model,
            self.processor,
            prompt,
            image=image,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            vision_cache=self.vision_cache,
            prompt_cache_state=self.prompt_cache_state,
        ):
            text += chunk.text
        return text

    def _generate(self) -> str:
        if self.use_server:
            return self._generate_server()
        return self._generate_local()

    def _run_tool_loop(self, model_output: str, depth: int = 0, max_depth: int = 5) -> str:
        if depth >= max_depth:
            return model_output + "\n[达到最大工具调用深度，停止循环]"

        tool_calls = parse_tool_calls(model_output)
        if not tool_calls:
            return model_output

        explanation = remove_tool_calls(model_output)
        if explanation:
            rprint(Markdown(explanation))

        for call in tool_calls:
            name = call.get("name")
            args = call.get("arguments", {})
            rprint(f"[dim]调用工具: {name}({json.dumps(args, ensure_ascii=False)})[/dim]")
            output = self.registry.execute(name, args)

            if name == "render_chart" and isinstance(output, str) and "保存于" in output:
                path = output.split("保存于 ")[-1].strip()
                if Path(path).exists():
                    self.current_image_path = path

            # 工具结果作为用户消息返回
            self._add_message(
                "user",
                f"工具 {name} 的返回结果：\n{output}",
            )

        follow_up = self._generate()
        return self._run_tool_loop(follow_up, depth + 1, max_depth)

    def chat(self, user_input: str):
        self._add_message("user", user_input)
        self.current_image_path = None

        model_output = self._generate()
        final_output = self._run_tool_loop(model_output)

        cleaned = remove_tool_calls(final_output)
        if cleaned:
            rprint("[bold green]AI Trader:[/bold green]")
            rprint(Markdown(cleaned))

        self._add_message("assistant", cleaned)

    def handle_command(self, command: str, args: str) -> bool:
        if command == "exit":
            return False
        if command == "clear":
            self.history.clear()
            self.current_image_path = None
            rprint("[bold yellow]对话历史已清空[/bold yellow]")
            return True
        if command == "help":
            self.print_help()
            return True
        if command == "image":
            if not args:
                rprint("[bold red]用法：/image <图片路径>[/bold red]")
                return True
            path = args.strip()
            if not Path(path).exists():
                rprint(f"[bold red]图片不存在: {path}[/bold red]")
                return True
            self.current_image_path = path
            self._add_message(
                "user",
                [
                    {"type": "text", "text": "我上传了一张图片，请分析。"},
                    {"type": "image", "path": path},
                ],
            )
            rprint(f"[bold blue]已加载图片:[/bold blue] {path}")
            return True
        rprint(f"[bold red]未知命令: /{command}[/bold red]")
        return True

    def run(self):
        self.load_model()
        system_prompt = build_system_prompt(self.registry)
        self.history.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        self.print_help()

        while True:
            try:
                user_input = Prompt.ask("\n[你]").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input[1:].split(" ", 1)
                command = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                if not self.handle_command(command, args):
                    break
                continue

            self.chat(user_input)

        rprint("[bold yellow]再见！[/bold yellow]")


def main():
    parser = argparse.ArgumentParser(description="AI Trader — 聊天即交易")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8080",
        help="已运行的 xmlx_vlm server 地址 (默认 http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="server API Key",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="不连 server，直接本地加载模型",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="本地加载的模型路径或 HuggingFace ID",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="生成温度 (默认 0.3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="最大生成 token 数 (默认 2048)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="非交互模式：直接执行一句指令后退出",
    )
    args = parser.parse_args()

    trader = AITraderChat(
        server_url=args.server_url,
        api_key=args.api_key,
        model_path=args.model,
        local_only=args.local,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    if args.prompt:
        trader.load_model()
        system_prompt = build_system_prompt(trader.registry)
        trader.history.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        trader.chat(args.prompt)
    else:
        trader.run()


if __name__ == "__main__":
    main()
