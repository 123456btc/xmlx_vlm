# SPDX-License-Identifier: Apache-2.0
"""
Gradio Chat UI that connects to xmlx_vlm server via OpenAI-compatible API.

Supports text, images, and video. Can use all server features:
- MCP tools
- Structured output (response_format)
- Thinking mode
- API Key auth

Usage:
    # Start the server first
    xmlx_vlm serve --model mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8080

    # Then run the chat UI
    xmlx_vlm chat_server

    # With auth and thinking
    xmlx_vlm chat_server --server-url http://localhost:8080 --api-key mykey --enable-thinking
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Optional

import requests

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "xmlx_vlm.chat_server requires 'gradio'. "
        "Install it with: pip install 'xmlx-vlm[ui]'"
    )


def _encode_file_to_base64(file_path: str) -> tuple[str, str]:
    """Encode a file to base64 data URL. Returns (data_url, media_type)."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    image_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    video_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }

    if suffix in image_types:
        mime_type, media_type = image_types[suffix], "image"
    elif suffix in video_types:
        mime_type, media_type = video_types[suffix], "video"
    else:
        mime_type, media_type = "image/jpeg", "image"

    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}", media_type


def _build_message_content(text: str, files: list[str] | None = None) -> list | str:
    """Build OpenAI-compatible message content with optional files."""
    if not files:
        return text

    content = []
    if text:
        content.append({"type": "text", "text": text})

    for file_path in files:
        data_url, media_type = _encode_file_to_base64(file_path)
        if media_type == "image":
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        elif media_type == "video":
            content.append({"type": "video_url", "video_url": {"url": data_url}})

    return content if content else text


def _fetch_model_name(server_url: str, headers: dict) -> Optional[str]:
    """Get the currently loaded model from /health, fallback to /v1/models."""
    # Prefer /health because it returns the actually loaded model
    try:
        resp = requests.get(f"{server_url}/health", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        loaded = data.get("loaded_model")
        if loaded:
            return loaded
    except Exception:
        pass

    # Fallback: scan /v1/models and pick a known-safe default
    try:
        resp = requests.get(f"{server_url}/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        # Prefer models that are likely already loaded or well-known
        for m in models:
            name = m.get("id", "")
            if "qwen" in name.lower() or "gemma" in name.lower():
                return name
        if models:
            return models[0].get("id", "default")
    except Exception as e:
        print(f"Warning: Could not fetch model list: {e}")
    return None


def _create_chat_function(
    server_url: str,
    headers: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
    thinking_budget: Optional[int] = None,
):
    """Create a chat function for Gradio ChatInterface."""
    media_cache: dict[int, list] = {}

    def chat(message: dict, history: list) -> str:
        text = message.get("text", "") if isinstance(message, dict) else message
        files = message.get("files", []) if isinstance(message, dict) else []

        messages = []
        # Rebuild history with cached media
        for i, msg in enumerate(history):
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if i in media_cache and role == "user":
                    text_parts = [
                        p.get("text", "")
                        for p in (content if isinstance(content, list) else [])
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    rebuilt = [{"type": "text", "text": " ".join(text_parts)}] if text_parts else []
                    rebuilt.extend(media_cache[i])
                    messages.append({"role": role, "content": rebuilt})
                else:
                    if isinstance(content, list):
                        text_parts = [
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        content = " ".join(text_parts)
                    messages.append({"role": role, "content": content})

        # Current message
        current_content = _build_message_content(text, files if files else None)
        messages.append({"role": "user", "content": current_content})

        if files:
            idx = len(history)
            items = []
            for fp in files:
                data_url, mt = _encode_file_to_base64(fp)
                if mt == "image":
                    items.append({"type": "image_url", "image_url": {"url": data_url}})
                elif mt == "video":
                    items.append({"type": "video_url", "video_url": {"url": data_url}})
            if items:
                media_cache[idx] = items

        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if enable_thinking:
            payload["enable_thinking"] = True
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget

        try:
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=600,
            )
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to server. Make sure xmlx_vlm server is running."
        except requests.exceptions.Timeout:
            return "Error: Server took too long to respond."
        except Exception as e:
            return f"Error: {e}"

    return chat


def _create_text_chat_function(
    server_url: str,
    headers: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
    thinking_budget: Optional[int] = None,
):
    """Create a text-only chat function."""

    def chat(message: str, history: list) -> str:
        messages = []
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if enable_thinking:
            payload["enable_thinking"] = True
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget

        try:
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=600,
            )
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to server. Make sure xmlx_vlm server is running."
        except requests.exceptions.Timeout:
            return "Error: Server took too long to respond."
        except Exception as e:
            return f"Error: {e}"

    return chat


def main():
    parser = argparse.ArgumentParser(
        description="Gradio Chat UI for xmlx_vlm server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    xmlx_vlm chat_server
    xmlx_vlm chat_server --server-url http://localhost:8080 --api-key mykey
    xmlx_vlm chat_server --enable-thinking --thinking-budget 50
        """,
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8080",
        help="xmlx_vlm server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("XMLX_VLM_API_KEY", "x123456"),
        help="API key for server authentication (default: x123456 or XMLX_VLM_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: auto-detect from /v1/models)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio interface (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Max thinking tokens",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Text-only mode (no image/video upload)",
    )
    args = parser.parse_args()

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    # Auto-detect model name if not provided
    model = args.model
    if not model:
        model = _fetch_model_name(args.server_url, headers)
    if not model:
        model = "default"

    print(f"Connecting to xmlx_vlm server at: {args.server_url}")
    print(f"Using model: {model}")
    print(f"Starting Gradio on port: {args.port}")

    if args.text_only:
        chat_fn = _create_text_chat_function(
            server_url=args.server_url,
            headers=headers,
            model=model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
        )
        demo = gr.ChatInterface(
            fn=chat_fn,
            title="mlx_vlm Chat",
            description="Chat with xmlx_vlm server. All server features (MCP, structured output, thinking) are available.",
            examples=[
                "Hello, who are you?",
                "What is 2+2? Answer in JSON.",
                "Explain quantum computing simply.",
            ],
        )
    else:
        chat_fn = _create_chat_function(
            server_url=args.server_url,
            headers=headers,
            model=model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
        )
        demo = gr.ChatInterface(
            fn=chat_fn,
            title="mlx_vlm Multimodal Chat",
            description="Chat with vision-language models via xmlx_vlm server. Upload images or videos!",
            multimodal=True,
            textbox=gr.MultimodalTextbox(
                file_types=["image", "video"],
                file_count="multiple",
                placeholder="Type a message or upload an image/video...",
            ),
        )

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
