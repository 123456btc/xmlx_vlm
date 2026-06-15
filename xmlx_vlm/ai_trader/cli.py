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

import asyncio
from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_SERVER_URL
from xmlx_vlm.ai_trader.intelligence.brain import Brain, BrainConfig
from xmlx_vlm.ai_trader.market_service.service import MarketDataService
from xmlx_vlm.ai_trader.oms.config.settings import get_settings, reset_settings
from xmlx_vlm.ai_trader.runtime.strategy_config import StrategyConfig
from xmlx_vlm.ai_trader.runtime.trader_manager import TraderManager
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


DEFAULT_SYSTEM_PROMPT = """你是 AI Trader，一个专业的合约高频与量化交易专家。用户通过自然语言与你对话，你可以调用工具完成行情分析、K 线图绘制、模拟交易以及联网检索等操作。

所有行情分析统一使用 Hyperliquid 数据源。

当前可用工具：
{tools_schema}

机构级数据能力与分析原则：
- L1 行情：以 mark_price 为基准，分析 basis/premium（基差/溢价）。盘口名义价值统一使用格式化字符串（B/M/K），如 1.24B、12.4M、850K，严禁读错单位。
- 多周期技术分析：优先调用 get_multi_timeframe_summary 获取 5m（短线情绪）、15m（中短线结构）、1h（趋势结构）三周期共振。ADX < 20 判定为震荡市，顺势做区间；ADX > 25 判定为强趋势。
- 资金流向与筹码分布：CVD（累积成交量差）与 OI（持仓量）1h/24h 变化率（ΔOI）优先级高于原始买卖比例。
- 联网检索规则：遇到任何需要获取实时新闻、常识或外部信息时，必须调用 web_search 工具，结合搜索结果用中文回答。

合约交易核心算法与风控约束（给出任何交易建议/执行下单时必须遵守并明确向用户说明）：

1. 动态仓位管理算法（Position Sizing）：
   - 基于风险系数：单笔最大风险敞口控制在账户总权益的 1%-2%。仓位名义价值 = (账户权益 * 风险系数) / (止损距离比例)。
   - 基于波动率（ATR）：止损距离优先采用 1.5 - 2.5 倍 ATR(14)。仓位名义价值 = (账户权益 * 风险系数) / (ATR * 乘数)。
   - 凯利公式（Kelly Criterion）：基于胜率 W 和盈亏比 R 计算 f* = W - (1-W)/R。推荐四分之一凯利（0.25 * f*）进行微调。单笔名义价值绝对上限为账户权益的 20%。

2. 严格风控算法（Risk Control）：
   - 强制双边止盈止损：入场建议必须带有明确的止损位（SL）与止盈位（TP），盈亏比必须 >= 1.5（推荐 >= 2.0）。
   - 保证金警戒线：整体已用保证金使用率（Margin Utilization）上限为 50%。超过 50% 时只允许 wait/hold 或减仓/平仓，绝不新开仓。
   - 最大回撤控制：当最大回撤达到 5% 时，应强烈建议停止开仓，进入观望与本金保护阶段。
   - 保本损机制：当浮盈达到止盈目标的一半时，必须主动将止损位移动至开仓均价（保本损）。

3. 加减仓与分批算法（Scaling In/Out）：
   - 盈利金字塔加仓（Pyramid Adding）：仅在已有持仓盈利时允许同向加仓；加仓金额必须少于初次建仓金额（如前一次的 50%）。亏损持仓绝对不加仓。
   - 减仓执行与锁盈：在关键阻力/支撑位，或趋势转弱时，调用 trading 的 close_position 进行部分减仓（如 50%）以锁定收益。
   - 下单指引：
     - 默认下单为 paper（纸盘模拟）。当需要执行部分减仓时，可计算对应减仓的名义价值（USD），在 `position_size_usd` 中填入该数值，或者告诉用户将要进行部分平仓；如果完全平仓，则不填或大于当前持仓价值。
     - 涉及交易时，务必先做详细的风控与算法逻辑说明，明确告知用户这是模拟还是实盘。
     - 如果用户要求急停，立即调用 trading action=emergency_stop。

请保持专业、严谨、简洁。"""


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


def _get_unlocked_kms_credentials() -> Dict[str, Dict[str, Any]]:
    """获取解密后的 KMS 凭证."""
    from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
    from xmlx_vlm.ai_trader.store import vault
    
    db = QuantSessionDB()
    password = os.environ.get("XMLX_VLM_VAULT_PASSWORD", "xmlx_vlm_default_secure_vault_passphrase_123456!")
    
    initialized = db.get_kms_config("vault_initialized") == "true"
    if not initialized:
        logger.info("KMS Vault is not initialized.")
        return {}
        
    salt_hex = db.get_kms_config("vault_salt")
    verifier_hex = db.get_kms_config("vault_verifier")
    try:
        salt = bytes.fromhex(salt_hex)
        derived = vault.derive_key(password, salt)
        if derived.hex() != verifier_hex:
            logger.error("KMS Vault auto-unlock verifier mismatch.")
            return {}
    except Exception as e:
        logger.error(f"Failed to verify vault password: {e}")
        return {}
        
    encrypted_keys = db.list_kms_keys()
    unlocked_creds = {}
    for key_row in encrypted_keys:
        key_id = key_row["key_id"]
        full_row = db.get_encrypted_kms_key(key_id)
        enc_payload_str = full_row["encrypted_private_key"]
        try:
            enc_dict = json.loads(enc_payload_str)
            decrypted_private_key = vault.decrypt_data(enc_dict, password)
            unlocked_creds[full_row["wallet_address"]] = {
                "key_id": key_id,
                "label": full_row["label"],
                "wallet_address": full_row["wallet_address"],
                "private_key": decrypted_private_key,
                "testnet": bool(full_row["testnet"])
            }
        except Exception as e:
            logger.error(f"Failed to decrypt key {key_id}: {e}")
    return unlocked_creds


async def _sync_strategies_with_watchlist(
    trader_manager: TraderManager,
    market_service: MarketDataService,
    unlocked_kms_creds: Dict[str, Dict[str, Any]],
    args: argparse.Namespace
) -> None:
    """动态同步 watchlist 币种的策略实例."""
    watchlist = market_service.get_watched_coins()
    if not watchlist:
        logger.info("Watchlist is empty, waiting for market data service...")
        return

    desired_strategy_ids = set()

    for coin in watchlist:
        coin = coin.upper()
        
        # 1. Paper trading strategy
        paper_id = f"trend_follow_{coin.lower()}_paper"
        desired_strategy_ids.add(paper_id)
        if not trader_manager.get(paper_id):
            config = StrategyConfig(
                id=paper_id,
                name=f"{coin} Trend Follower (Paper)",
                exchange="paper",
                strategy_type="trend",
                symbols=[f"{coin}/USDC"],
                scan_interval_seconds=300,
                enabled=True,
                live_enabled=False,
                dry_run=True,
                server_url=args.server_url,
                api_key=args.api_key,
                model_path=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            trader_manager.register(config)
            logger.info("Dynamically registered strategy %s", paper_id)

        paper_instance = trader_manager.get(paper_id)
        if paper_instance and not paper_instance.is_running:
            try:
                await trader_manager.start(paper_id)
                logger.info("Dynamically started strategy %s", paper_id)
                await asyncio.sleep(15)  # 15s stagger delay
            except Exception as e:
                logger.error("Failed to start strategy %s: %s", paper_id, e)

        # 2. Live trading strategies for unlocked KMS wallets
        for wallet_address, cred in unlocked_kms_creds.items():
            live_id = f"trend_follow_{coin.lower()}_hyperliquid"
            desired_strategy_ids.add(live_id)
            if not trader_manager.get(live_id):
                config = StrategyConfig(
                    id=live_id,
                    name=f"{coin} Trend Follower (Hyperliquid)",
                    exchange="hyperliquid",
                    strategy_type="trend",
                    symbols=[f"{coin}/USDC"],
                    scan_interval_seconds=300,
                    enabled=True,
                    live_enabled=True,
                    dry_run=False,
                    server_url=args.server_url,
                    api_key=args.api_key,
                    model_path=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    wallet_address=cred["wallet_address"],
                    private_key=cred["private_key"],
                    testnet=cred["testnet"],
                )
                trader_manager.register(config)
                logger.info("Dynamically registered live strategy %s", live_id)

            live_instance = trader_manager.get(live_id)
            if live_instance and not live_instance.is_running:
                try:
                    await trader_manager.start(live_id)
                    logger.info("Dynamically started live strategy %s", live_id)
                    await asyncio.sleep(15)  # 15s stagger delay
                except Exception as e:
                    logger.error("Failed to start live strategy %s: %s", live_id, e)

    # Clean up strategies no longer in desired set
    for sid in trader_manager.list_ids():
        if sid not in desired_strategy_ids:
            logger.info("Strategy %s no longer in watchlist, stopping and unregistering...", sid)
            try:
                await trader_manager.stop(sid)
                trader_manager.unregister(sid)
            except Exception as e:
                logger.error("Failed to stop/unregister strategy %s: %s", sid, e)


async def _run_auto_start(args: argparse.Namespace) -> None:
    """运行 AI 策略引擎主循环."""
    logger.info("Starting MarketDataService...")
    market_service = MarketDataService()
    market_service.start()

    # 初始化 TraderManager
    trader_manager = TraderManager(market_service=market_service)

    # 获取 KMS 凭证
    unlocked_kms_creds = _get_unlocked_kms_credentials()
    logger.info("KMS Vault unlocked. Loaded %d live trading wallet(s).", len(unlocked_kms_creds))

    try:
        while True:
            await _sync_strategies_with_watchlist(trader_manager, market_service, unlocked_kms_creds, args)
            await asyncio.sleep(30)
    except asyncio.CancelledError:
        logger.info("Received stop signal.")
    finally:
        logger.info("Stopping all strategies and market service...")
        await trader_manager.stop_all()
        market_service.stop()


def cmd_auto_start(args: argparse.Namespace) -> None:
    """启动策略引擎自动交易守护进程."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True
    )
    logger.info("Starting AI Strategy Engine daemon...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    main_task = loop.create_task(_run_auto_start(args))
    
    try:
        loop.run_until_complete(main_task)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown signal received, cancelling main task...")
        main_task.cancel()
        try:
            loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            pass
    finally:
        loop.close()
        logger.info("Strategy Engine daemon stopped.")


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
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="自动启动后台策略引擎",
    )
    parser.add_argument(
        "--auto-stop",
        action="store_true",
        help="停止后台策略引擎",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="查看后台策略引擎状态",
    )
    parser.add_argument(
        "--emergency-stop",
        action="store_true",
        help="紧急停止所有仓位与策略",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="指定策略配置文件路径",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="启用实盘交易",
    )
    args = parser.parse_args()

    if args.auto_start:
        cmd_auto_start(args)
        return

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
