from __future__ import annotations

import asyncio
import json
import logging
import uuid
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from enum import Enum

import requests
import httpx
from rich import print as rprint

from xmlx_vlm import load
from xmlx_vlm.generate import stream_generate
from xmlx_vlm.prompt_utils import apply_chat_template
from xmlx_vlm.vision_cache import VisionFeatureCache

from xmlx_vlm.agent_core import (
    ContextCompressor,
    ThinkScrubber,
    ToolCallGuardrailConfig,
    ToolCallGuardrails,
)
from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_SERVER_URL
from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
from xmlx_vlm.ai_trader.tools.registry import ToolRegistry
from xmlx_vlm.ai_trader.cli import build_system_prompt, parse_tool_calls, remove_tool_calls
from xmlx_vlm.ai_trader.agent.telemetry import QuantTracer

logger = logging.getLogger(__name__)

class AgentState(str, Enum):
    PLANNING = "planning"
    READONLY_EXECUTION = "readonly_execution"
    APPROVAL_GATE = "approval_gate"
    SENSITIVE_EXECUTION = "sensitive_execution"
    COMPLETED = "completed"

def sanitize_error(text: str) -> str:
    """Mask credentials (like private keys or addresses) in tool tracebacks before returning to LLM."""
    if not isinstance(text, str):
        return text
    # Mask private key (typically 64 hex characters, with optional 0x prefix)
    text = re.sub(r"\b(0x)?[a-fA-F0-9]{64}\b", "[REDACTED_PRIVATE_KEY]", text)
    # Mask Hyperliquid address (0x followed by 40 hex characters)
    text = re.sub(r"\b0x[a-fA-F0-9]{40}\b", "[REDACTED_ADDRESS]", text)
    return text

class AITraderAgent:
    """Async AI Trader Agent managing the conversation loop, database persistence, and tools."""

    def __init__(
        self,
        db: QuantSessionDB,
        server_url: str = DEFAULT_SERVER_URL,
        api_key: Optional[str] = DEFAULT_API_KEY,
        model_path: Optional[str] = None,
        local_only: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        live: bool = False,
        exchange: str = "local",
        risk_profile: str = "conservative",
        dry_run: bool = False,
    ):
        self.db = db
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.model_path = model_path or DEFAULT_MODEL
        self.local_only = local_only
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.live = live
        self.exchange = exchange
        self.risk_profile = risk_profile
        self.dry_run = dry_run

        self.registry = ToolRegistry(
            live=live,
            exchange=exchange,
            risk_profile=risk_profile,
            dry_run=dry_run,
        )

        self.use_server = False
        self.server_model: Optional[str] = None
        self.model = None
        self.processor = None
        self.vision_cache = VisionFeatureCache()
        self.prompt_cache_state = None
        
        # Enterprise Guardrails & Context Compressor
        self.guardrails = ToolCallGuardrails(
            ToolCallGuardrailConfig(
                warnings_enabled=True,
                hard_stop_enabled=True,
                exact_failure_warn_after=2,
                exact_failure_block_after=3,
                no_progress_warn_after=2,
                no_progress_block_after=4,
            )
        )
        self.compressor = ContextCompressor(
            max_context_tokens=16384,
            compression_threshold=0.75,
            tail_token_budget=4096,
        )

        # Pending approvals future dictionary: tool_call_id -> asyncio.Future
        self.pending_approvals: Dict[str, asyncio.Future] = {}

    def probe_server(self) -> bool:
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = requests.get(
                f"{self.server_url}/health",
                headers=headers,
                timeout=3,
                proxies={"http": None, "https": None},
            )
            if resp.status_code == 200:
                data = resp.json()
                loaded = data.get("loaded_model")
                if loaded:
                    self.server_model = loaded
                    logger.info("Connected to inference server: %s (model: %s)", self.server_url, loaded)
                    return True
        except Exception as e:
            logger.debug("Server probe failed: %s", e)
        return False

    async def load_agent(self):
        """Pre-load model or connect to the server."""
        # Connect to MCP servers
        try:
            await self.registry.connect_mcp_servers()
        except Exception as e:
            logger.error("Failed to connect MCP servers: %s", e)

        if self.local_only:
            await asyncio.to_thread(self._load_local_model)
            return

        if await asyncio.to_thread(self.probe_server):
            self.use_server = True
            return

        logger.warning("Could not connect to inference server. Loading model locally...")
        await asyncio.to_thread(self._load_local_model)

    def _load_local_model(self):
        logger.info("Loading local model %s...", self.model_path)
        self.model, self.processor = load(self.model_path)
        logger.info("Local model loaded: %s", self.model_path)

    def _build_history_for_mlx(self, db_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert database message rows to list format suitable for chat template."""
        history = []
        for msg in db_messages:
            role = msg["role"]
            content = msg["content"]
            
            # Translate content representation
            if isinstance(content, str):
                try:
                    content_parsed = json.loads(content)
                    if isinstance(content_parsed, list):
                        content = content_parsed
                except Exception:
                    pass

            if isinstance(content, list):
                history.append({"role": role, "content": content})
            else:
                history.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
        return history

    def _build_history_for_openai(self, db_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert database message rows to OpenAI-compatible chat message list."""
        messages = []
        for msg in db_messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                try:
                    content_parsed = json.loads(content)
                    if isinstance(content_parsed, list):
                        content = content_parsed
                except Exception:
                    pass

            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
                text = "\n".join(text_parts)
                images = [p for p in content if isinstance(p, dict) and (p.get("type") == "image_url" or "image" in p)]
                videos = [p for p in content if isinstance(p, dict) and (p.get("type") == "video_url" or "video" in p)]
                
                if (images or videos) and role == "user":
                    openai_content = [{"type": "text", "text": text}]
                    for img in images:
                        if img.get("type") == "image_url":
                            openai_content.append(img)
                        elif "path" in img:
                            try:
                                with open(img["path"], "rb") as f:
                                    import base64
                                    data = base64.b64encode(f.read()).decode("utf-8")
                                mime_type = "image/png"
                                if img["path"].lower().endswith((".jpg", ".jpeg")):
                                    mime_type = "image/jpeg"
                                elif img["path"].lower().endswith(".webp"):
                                    mime_type = "image/webp"
                                openai_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{data}"}
                                })
                            except Exception as e:
                                logger.error("Failed to base64 encode image %s: %s", img["path"], e)
                                
                    for vid in videos:
                        if vid.get("type") == "video_url":
                            openai_content.append(vid)
                        elif "path" in vid:
                            try:
                                with open(vid["path"], "rb") as f:
                                    import base64
                                    data = base64.b64encode(f.read()).decode("utf-8")
                                mime_type = "video/mp4"
                                if vid["path"].lower().endswith(".webm"):
                                    mime_type = "video/webm"
                                elif vid["path"].lower().endswith(".mov"):
                                    mime_type = "video/quicktime"
                                openai_content.append({
                                    "type": "video_url",
                                    "video_url": {"url": f"data:{mime_type};base64,{data}"}
                                })
                            except Exception as e:
                                logger.error("Failed to base64 encode video %s: %s", vid["path"], e)
                                
                    messages.append({"role": role, "content": openai_content})
                else:
                    messages.append({"role": role, "content": text})
            else:
                messages.append({"role": role, "content": str(content)})
        return messages

    async def _summarize_session_title(self, user_input: str) -> str:
        """Generate a short 3-6 words Chinese title for the session based on the user's first input."""
        prompt = (
            "请根据用户的输入，总结成一个非常简短、生动的中文会话标题（3-6个字，不要有标点符号，不要有前缀如“标题：”或解释）。\n"
            f"用户输入: {user_input}\n"
            "会话标题:"
        )
        if self.use_server:
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                payload = {
                    "model": self.server_model or "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "max_tokens": 20,
                    "temperature": 0.1,
                }
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), trust_env=False) as client:
                    resp = await client.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    title = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    title = title.replace('"', '').replace("'", "").strip()
                    for prefix in ["标题", "会话标题", "：", ":"]:
                        if title.startswith(prefix):
                            title = title[len(prefix):].strip()
                    if title:
                        return title
            except Exception as e:
                logger.error("Failed to summarize title via server: %s", e)
        else:
            try:
                history = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                formatted_prompt = apply_chat_template(
                    self.processor,
                    self.model.config,
                    history,
                    num_images=0,
                )
                iterator = stream_generate(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    max_tokens=20,
                    temperature=0.1,
                    vision_cache=self.vision_cache,
                    prompt_cache_state=self.prompt_cache_state,
                )
                def get_all_tokens(it):
                    tokens = []
                    for chunk in it:
                        tokens.append(chunk.text)
                    return "".join(tokens)
                title = await asyncio.to_thread(get_all_tokens, iterator)
                title = title.strip().replace('"', '').replace("'", "").strip()
                for prefix in ["标题", "会话标题", "：", ":"]:
                    if title.startswith(prefix):
                        title = title[len(prefix):].strip()
                if title:
                    return title
            except Exception as e:
                logger.error("Failed to summarize title locally: %s", e)
        fallback = user_input[:8].strip()
        if not fallback:
            fallback = "新交易会话"
        return fallback

    async def generate_stream(
        self, session_id: str, user_input: str, attachments: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Main generator yielding chunks of conversation, executing tool calls, and updating state."""
        # 1. Fetch/Init session
        session = self.db.get_session(session_id)
        if not session:
            model_name = self.server_model if self.use_server else self.model_path
            self.db.create_session(
                session_id=session_id,
                title="New Session",
                model=model_name or "default",
                mode="live" if self.live else "paper",
            )
            
            # Inject system prompt
            system_prompt = build_system_prompt(self.registry)
            self.db.add_message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="system",
                content=system_prompt,
            )

        # 2. Append User message
        if attachments:
            user_content = [{"type": "text", "text": user_input}]
            for att in attachments:
                att_type = att.get("type")
                att_url = att.get("url")
                att_path = att.get("path")
                att_name = att.get("name", "file")
                
                if att_type == "image":
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": att_url},
                        "path": att_path
                    })
                elif att_type == "video":
                    user_content.append({
                        "type": "video_url",
                        "video_url": {"url": att_url},
                        "path": att_path
                    })
                elif att_type == "text" and att_path:
                    try:
                        with open(att_path, "r", encoding="utf-8", errors="ignore") as f:
                            file_text = f.read()
                        user_content.append({
                            "type": "text",
                            "text": f"\n\n[Attached File: {att_name}]\n```\n{file_text}\n```"
                        })
                    except Exception as e:
                        user_content.append({
                            "type": "text",
                            "text": f"\n\n[Error reading attached file {att_name}: {e}]"
                        })
            self.db.add_message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=user_content,
            )
        else:
            self.db.add_message(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=user_input,
            )

        # 3. Enter state-graph routing loop
        depth = 0
        max_depth = 5
        current_state = AgentState.PLANNING
        
        # Initialize Tracer
        tracer = QuantTracer()
        tracer.start_span("AITraderAgent.generate_stream", {"session_id": session_id})
        
        try:
            while current_state != AgentState.COMPLETED and depth < max_depth:
                t0_state = time.perf_counter()
                
                if current_state == AgentState.PLANNING:
                    db_messages = self.db.get_messages(session_id)
                    
                    # Apply Anti-Hijack Context Compression if messages exceed token budget or count
                    if self.compressor.should_compress(db_messages) or len(db_messages) > 16:
                        compressed_msgs, was_comp = self.compressor.compress(db_messages)
                        if was_comp:
                            logger.info("Session [%s] context compressed with Anti-Hijack prefix", session_id)
                            db_messages = compressed_msgs

                    # Stream text from model
                    model_output = ""
                    server_tool_calls = []
                    t0_llm = time.perf_counter()
                    if self.use_server:
                        async for chunk_type, chunk in self._stream_from_server(db_messages):
                            if chunk_type == "tool_call":
                                try:
                                    tc_obj = json.loads(chunk)
                                    server_tool_calls.append(tc_obj)
                                except Exception:
                                    pass
                            else:
                                model_output += chunk
                                yield {"type": chunk_type, "content": chunk}
                    else:
                        async for chunk_type, chunk in self._stream_from_local(db_messages):
                            model_output += chunk
                            yield {"type": chunk_type, "content": chunk}
                    
                    # Tracing LLM call in agent_loop
                    llm_duration = time.perf_counter() - t0_llm
                    tracer.log_step("llm_planning_stream", llm_duration, {"session_id": session_id})
                    
                    # Use ThinkScrubber to separate reasoning from clean tool/text payload
                    clean_output, reasoning = ThinkScrubber.scrub(model_output)
                    if reasoning:
                        tracer.log_step("llm_reasoning_extracted", 0.0, {"reasoning_length": len(reasoning)})

                    # Check for tool calls (either structured from server or parsed from clean text)
                    tool_calls = server_tool_calls or parse_tool_calls(clean_output or model_output)
                    if not tool_calls:
                        explanation = remove_tool_calls(clean_output or model_output)
                        self.db.add_message(
                            message_id=str(uuid.uuid4()),
                            session_id=session_id,
                            role="assistant",
                            content=explanation or clean_output or model_output,
                        )
                        current_state = AgentState.COMPLETED
                    else:
                        self.db.add_message(
                            message_id=str(uuid.uuid4()),
                            session_id=session_id,
                            role="assistant",
                            content=model_output,
                        )
                        
                        # Separate sensitive calls from read-only calls
                        sensitive_calls = []
                        readonly_calls = []
                        for call in tool_calls:
                            name = call.get("name")
                            args = call.get("arguments", {})
                            is_sensitive = (
                                name == "trading"
                                and args.get("action") in ["place_order", "close_position", "emergency_stop"]
                            )
                            if is_sensitive:
                                sensitive_calls.append(call)
                            else:
                                readonly_calls.append(call)
                        
                        # Route next state
                        if readonly_calls:
                            current_state = AgentState.READONLY_EXECUTION
                        elif sensitive_calls:
                            current_state = AgentState.APPROVAL_GATE
                        else:
                            current_state = AgentState.COMPLETED
                    
                    tracer.log_step(f"state_{AgentState.PLANNING.value}", time.perf_counter() - t0_state)

                elif current_state == AgentState.READONLY_EXECUTION:
                    # Execute read-only tools
                    for call in readonly_calls:
                        yield {"type": "tool_start", "name": call.get("name"), "arguments": call.get("arguments", {})}

                    # Define concurrent execution coroutines
                    async def run_readonly(c):
                        n = c.get("name")
                        a = c.get("arguments", {})
                        logger.info("Executing concurrent tool: %s", n)
                        t0_tool = time.perf_counter()
                        try:
                            out = await asyncio.to_thread(self.registry.execute, n, a)
                            out = sanitize_error(out)
                        except Exception as err:
                            out = f"Error: {err}"
                        
                        tracer.log_step(f"tool_{n}", time.perf_counter() - t0_tool, {"args": a})
                        
                        chart_url = None
                        if n == "render_chart" and isinstance(out, str) and "保存于" in out:
                            path_str = out.split("保存于 ")[-1].strip()
                            if Path(path_str).exists():
                                filename = Path(path_str).name
                                chart_url = f"/api/static/charts/{filename}"
                        return c, out, chart_url

                    # Run parallel gathers
                    results = await asyncio.gather(*[run_readonly(c) for c in readonly_calls])

                    # Process results, evaluate guardrails, and yield tool_end events
                    for c, out, chart_url in results:
                        n = c.get("name")
                        a = c.get("arguments", {})
                        
                        # Check guardrails
                        is_error = str(out).lower().startswith("error")
                        decision = self.guardrails.observe_and_check(
                            tool=n,
                            args=a if isinstance(a, dict) else {},
                            result=out,
                            is_error=is_error,
                        )
                        feedback_content = f"工具 {n} 的返回结果：\n{out}"
                        if decision.synthetic_message:
                            feedback_content += f"\n\n{decision.synthetic_message}"

                        if chart_url:
                            yield {"type": "image_render", "url": chart_url}
                        yield {
                            "type": "tool_end",
                            "name": n,
                            "output": out,
                            "chart_url": chart_url,
                        }
                        self.db.add_message(
                            message_id=str(uuid.uuid4()),
                            session_id=session_id,
                            role="user",
                            content=feedback_content,
                        )
                    
                    # Clear processed readonly tools
                    readonly_calls = []
                    # Route to approval if sensitive calls exist, else plan again
                    if sensitive_calls:
                        current_state = AgentState.APPROVAL_GATE
                    else:
                        depth += 1
                        current_state = AgentState.PLANNING
                        
                    tracer.log_step(f"state_{AgentState.READONLY_EXECUTION.value}", time.perf_counter() - t0_state)

                elif current_state == AgentState.APPROVAL_GATE:
                    # Get the next sensitive call
                    current_sensitive_call = sensitive_calls[0]
                    name = current_sensitive_call.get("name")
                    args = current_sensitive_call.get("arguments", {})
                    tool_call_id = current_sensitive_call.get("id") or str(uuid.uuid4())

                    # Yield interactive approval event to frontend Client
                    yield {
                        "type": "approval_required",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "arguments": args,
                    }

                    # Await user decision over WebSocket
                    fut = asyncio.Future()
                    self.pending_approvals[tool_call_id] = fut
                    
                    try:
                        approved = await fut
                    except asyncio.CancelledError:
                        approved = False
                    finally:
                        self.pending_approvals.pop(tool_call_id, None)

                    if not approved:
                        output = "[Rejected by User] The user cancelled execution for this trading tool."
                        yield {
                            "type": "tool_end",
                            "name": name,
                            "output": output,
                            "chart_url": None,
                        }
                        # Save rejection message to DB
                        self.db.add_message(
                            message_id=str(uuid.uuid4()),
                            session_id=session_id,
                            role="user",
                            content=f"工具 {name} 的返回结果：\n{output}",
                        )
                        # Remove from list and route back to planning
                        sensitive_calls.pop(0)
                        if not sensitive_calls:
                            depth += 1
                            current_state = AgentState.PLANNING
                        else:
                            current_state = AgentState.APPROVAL_GATE
                    else:
                        current_state = AgentState.SENSITIVE_EXECUTION
                        
                    tracer.log_step(f"state_{AgentState.APPROVAL_GATE.value}", time.perf_counter() - t0_state)

                elif current_state == AgentState.SENSITIVE_EXECUTION:
                    # Execute current sensitive tool call
                    current_sensitive_call = sensitive_calls[0]
                    name = current_sensitive_call.get("name")
                    args = current_sensitive_call.get("arguments", {})
                    
                    yield {"type": "tool_start", "name": name, "arguments": args}
                    logger.info("Executing approved sensitive tool: %s with args: %s", name, args)
                    
                    t0_tool = time.perf_counter()
                    try:
                        output = await asyncio.to_thread(self.registry.execute, name, args)
                        output = sanitize_error(output)
                    except Exception as err:
                        output = f"Execution Error: {err}"
                    
                    tracer.log_step(f"tool_{name}", time.perf_counter() - t0_tool, {"args": args})

                    # Check guardrails on sensitive trading action
                    is_error = str(output).lower().startswith("error") or "execution error" in str(output).lower()
                    guard_decision = self.guardrails.observe_and_check(
                        tool=name,
                        args=args if isinstance(args, dict) else {},
                        result=output,
                        is_error=is_error,
                    )
                    feedback_content = f"工具 {name} 的返回结果：\n{output}"
                    if guard_decision.synthetic_message:
                        feedback_content += f"\n\n{guard_decision.synthetic_message}"

                    # Log trade if it's trading execution
                    if name == "trading" and args.get("action") == "place_order" and "已提交" in output:
                        try:
                            trade_id = str(uuid.uuid4())
                            symbol = args.get("symbol", "unknown")
                            side = args.get("side", "buy")
                            qty = float(args.get("qty", 0))
                            
                            price_match = re.search(r"@\s*([\d\.]+)", output)
                            price = float(price_match.group(1)) if price_match else 0.0
                            
                            self.db.log_trade(
                                trade_id=trade_id,
                                session_id=session_id,
                                symbol=symbol,
                                side=side,
                                qty=qty,
                                price=price,
                                pnl=0.0,
                                status="filled",
                            )
                        except Exception as e:
                            logger.error("Failed to parse and log trade: %s", e)

                    yield {
                        "type": "tool_end",
                        "name": name,
                        "output": output,
                        "chart_url": None,
                    }

                    # Save tool output to database as a user role message to feed back to the LLM
                    self.db.add_message(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        role="user",
                        content=feedback_content,
                    )

                    # Remove completed sensitive call
                    sensitive_calls.pop(0)
                    if sensitive_calls:
                        current_state = AgentState.APPROVAL_GATE
                    else:
                        depth += 1
                        current_state = AgentState.PLANNING
                        
                    tracer.log_step(f"state_{AgentState.SENSITIVE_EXECUTION.value}", time.perf_counter() - t0_state)

            if depth >= max_depth:
                yield {"type": "error", "message": "Max tool calling loop depth reached."}
                tracer.end_span("max_depth_reached")
            else:
                tracer.end_span("success")
                
        except Exception as e:
            tracer.end_span("failed", str(e))
            raise e

        # Check if the title needs to be summarized (e.g. first user message)
        session = self.db.get_session(session_id)
        if session:
            current_title = session.get("title", "")
            messages = self.db.get_messages(session_id)
            user_msg_count = sum(1 for m in messages if m["role"] == "user")
            is_default_title = current_title in [
                "Quant Chat Session",
                "New Trading Session",
                "New Session",
                "Select a Session",
                "Quant Session",
            ]
            if is_default_title or user_msg_count == 1:
                try:
                    summarized_title = await self._summarize_session_title(user_input)
                    self.db.update_session_activity(session_id, title=summarized_title)
                    yield {
                        "type": "title_update",
                        "session_id": session_id,
                        "title": summarized_title,
                    }
                except Exception as e:
                    logger.error("Failed to run title summarization: %s", e)

    async def _stream_from_server(self, db_messages: List[Dict[str, Any]]) -> AsyncGenerator[Tuple[str, str], None]:
        """Call the FastAPI server completions endpoint asynchronously and yield (type, token) tuples."""
        messages = self._build_history_for_openai(db_messages)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.server_model or "default",
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tools": self.registry.list_tools(),
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), trust_env=False) as client:
                async with client.stream(
                    "POST",
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            
                            reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                            if reasoning:
                                yield "thinking", reasoning
                            
                            content = delta.get("content", "")
                            if content:
                                yield "text", content
                                
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                for tc in tool_calls:
                                    if isinstance(tc, dict):
                                        fn = tc.get("function", {})
                                        fn_name = fn.get("name") if isinstance(fn, dict) else tc.get("name")
                                        fn_args = fn.get("arguments", {}) if isinstance(fn, dict) else tc.get("arguments", {})
                                        if isinstance(fn_args, str):
                                            try:
                                                fn_args = json.loads(fn_args)
                                            except Exception:
                                                pass
                                        if fn_name:
                                            yield "tool_call", json.dumps({"name": fn_name, "arguments": fn_args, "id": tc.get("id")})
                        except Exception:
                            continue
        except Exception as e:
            logger.error("Error connecting to completions server: %s", e)
            yield "text", f"\n[连接服务器失败: {e}]"

    async def _stream_from_local(self, db_messages: List[Dict[str, Any]]) -> AsyncGenerator[Tuple[str, str], None]:
        """Directly invoke MLX local generator in a thread and yield (type, token) tuples."""
        history = self._build_history_for_mlx(db_messages)
        chat_template_kwargs = {"tools": self.registry.list_tools()}
        
        # Extract images and videos
        image_paths = []
        video_paths = []
        for msg in history:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for part in msg["content"]:
                    if isinstance(part, dict) and "path" in part:
                        path = part["path"]
                        if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
                            video_paths.append(path)
                        else:
                            image_paths.append(path)

        num_images = len(image_paths)
        image = image_paths if image_paths else None
        
        if video_paths:
            chat_template_kwargs["video"] = video_paths[0] if len(video_paths) == 1 else video_paths
            chat_template_kwargs["fps"] = 1.0

        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            history,
            num_images=num_images,
            **chat_template_kwargs,
        )

        gen_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "vision_cache": self.vision_cache,
            "prompt_cache_state": self.prompt_cache_state,
        }
        
        if video_paths:
            gen_kwargs["video"] = video_paths[0] if len(video_paths) == 1 else video_paths

        iterator = stream_generate(
            self.model,
            self.processor,
            prompt,
            image=image,
            **gen_kwargs,
        )

        def get_next_chunk(it):
            try:
                return next(it)
            except StopIteration:
                return None

        while True:
            chunk = await asyncio.to_thread(get_next_chunk, iterator)
            if chunk is None:
                break
            yield "text", chunk.text
