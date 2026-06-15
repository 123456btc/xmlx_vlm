"""本地私有 LLM 客户端.

优先连接 service.sh 启动的本地服务 (OpenAI 兼容 /v1/chat/completions)，
未检测到服务时回退到本地 MLX 推理 (stream_generate)。
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_SERVER_URL
from xmlx_vlm.ai_trader.decision.engine import LLMClient

logger = logging.getLogger(__name__)


def _resolve_server_url(server_url: Optional[str]) -> str:
    """解析服务 URL：显式参数 > XMLX_VLM_PORT > 默认值."""
    if server_url:
        return server_url.rstrip("/")
    port = os.getenv("XMLX_VLM_PORT")
    if port:
        return f"http://localhost:{port}"
    return DEFAULT_SERVER_URL


def _resolve_api_key(api_key: Optional[str]) -> str:
    return api_key or os.getenv("XMLX_VLM_API_KEY") or DEFAULT_API_KEY


class BaseLLMClient(ABC):
    """本地 LLM 客户端基类."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.server_url = _resolve_server_url(server_url)
        self.api_key = _resolve_api_key(api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def _build_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages


class LocalServiceLLMClient(BaseLLMClient):
    """连接本地 service.sh 推理服务."""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL, **kwargs):
        super().__init__(server_url=server_url, **kwargs)
        self._server_model: Optional[str] = None
        self._available: Optional[bool] = None

    def probe(self, timeout: float = 5.0) -> bool:
        """探测本地服务是否可用."""
        if self._available is not None:
            return self._available
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.get(
                f"{self.server_url}/health",
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._server_model = data.get("loaded_model")
                self._available = bool(self._server_model)
                logger.info(
                    "Local LLM service available at %s (model: %s)",
                    self.server_url,
                    self._server_model,
                )
                return self._available
        except Exception as exc:
            logger.debug("Local LLM service probe failed: %s", exc)
        self._available = False
        return False

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self._server_model or self.model or "default",
            "messages": self._build_messages(system_prompt, user_prompt),
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "release_kv": True,
        }

        text = ""
        reasoning_text = ""
        try:
            with requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=600,
            ) as resp:
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
                        
                        # Read reasoning content (split by server)
                        reasoning = delta.get("reasoning", "")
                        if reasoning:
                            reasoning_text += reasoning
                            
                        # Read normal content
                        content = delta.get("content", "")
                        if content:
                            text += content
                    except json.JSONDecodeError:
                        continue
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                raise RuntimeError(
                    "本地服务 API Key 校验失败。请检查 XMLX_VLM_API_KEY 是否与 service.sh 一致。"
                ) from exc
            raise
            
        if reasoning_text:
            return f"<think>\n{reasoning_text.strip()}\n</think>\n\n{text}"
        return text


class LocalMLXLLMClient(BaseLLMClient):
    """本地 MLX 模型推理（无服务时回退）."""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path or os.getenv("XMLX_VLM_MODEL") or DEFAULT_MODEL
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from xmlx_vlm import load
            from xmlx_vlm.prompt_utils import apply_chat_template

            self._model, self._processor = load(self.model_path)
            self._apply_chat_template = apply_chat_template
            logger.info("Local MLX model loaded: %s", self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load local MLX model: {exc}") from exc

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._load()
        from xmlx_vlm.generate import stream_generate

        messages = self._build_messages(system_prompt, user_prompt)
        prompt = self._apply_chat_template(
            self._processor,
            self._model.config,
            messages,
            num_images=0,
        )
        text = ""
        for chunk in stream_generate(
            self._model,
            self._processor,
            prompt,
            image=None,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ):
            text += chunk.text
        return text


class AutoLLMClient(BaseLLMClient):
    """自动选择：优先本地服务，未启动则回退本地 MLX.

    若 allow_mlx_fallback=False 且服务不可用，直接报错，避免静默加载第二份模型。
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        allow_mlx_fallback: bool = True,
    ):
        super().__init__(
            server_url=server_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model_path = model_path
        self.allow_mlx_fallback = allow_mlx_fallback
        self._service_client = LocalServiceLLMClient(
            server_url=self.server_url,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._mlx_client: Optional[LocalMLXLLMClient] = None
        self._use_service: Optional[bool] = None

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self._use_service is None:
            self._use_service = self._service_client.probe()

        if self._use_service:
            logger.info("Using local service at %s", self.server_url)
            return await self._service_client.complete(system_prompt, user_prompt)

        if not self.allow_mlx_fallback:
            raise RuntimeError(
                f"本地推理服务未在 {self.server_url} 上运行，且已禁用 MLX 回退。"
                "请启动 service.sh 或检查 XMLX_VLM_PORT 环境变量。"
            )

        logger.warning(
            "Local service not available at %s, falling back to local MLX model",
            self.server_url,
        )
        if self._mlx_client is None:
            self._mlx_client = LocalMLXLLMClient(
                model_path=self.model_path,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return await self._mlx_client.complete(system_prompt, user_prompt)
