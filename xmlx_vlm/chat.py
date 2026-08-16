import argparse
import base64
import codecs
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# When this script is run directly (``python xmlx_vlm/chat.py``), ensure the
# local source tree is used instead of an older installed ``xmlx_vlm`` package.
if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    del _project_root

import requests

try:
    from rich import print as rprint
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError:
    rprint = print
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        def rule(self, *args, **kwargs):
            pass
    class Markdown:
        def __init__(self, text):
            self.text = text
    class Panel:
        def __init__(self, renderable, **kwargs):
            self.renderable = renderable
    class Prompt:
        @staticmethod
        def ask(prompt, **kwargs):
            return input(prompt)

from xmlx_vlm import load
from xmlx_vlm.config import (
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_SERVER_URL,
)
from xmlx_vlm.generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_TOP_P,
    GenerationResult,
    PromptCacheState,
    stream_generate,
)
from xmlx_vlm.prompt_utils import apply_chat_template
from xmlx_vlm.utils import load_image
from xmlx_vlm.vision_cache import VisionFeatureCache


def _encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a base64 data URL for the OpenAI API."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_types.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


class MLXVisionChat:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        verbose: bool = False,
        server_url: Optional[str] = DEFAULT_SERVER_URL,
        api_key: Optional[str] = None,
        local_only: bool = False,
        **kwargs,
    ):
        self.console = Console()
        self.verbose = verbose
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: List[Dict] = []
        self.current_image = None
        self.current_image_path = None
        self.image_paths: List[str] = []
        self.stream_kwargs = kwargs

        self.use_server = False
        self.server_url = server_url
        self.api_key = api_key

        # Prefer an already-running server to avoid loading a second model.
        if not local_only and server_url:
            loaded_model = self._probe_server(server_url, api_key)
            if loaded_model:
                self.use_server = True
                self.model = loaded_model
                rprint(
                    f"[bold green]Connected to running server at {server_url} "
                    f"(model: {loaded_model})[/bold green]"
                )
                self.print_help()
                return

        # Local mode: load model and use local caches.
        self.vision_cache = VisionFeatureCache()
        self.prompt_cache_state = PromptCacheState()
        with self.console.status("[bold green]Loading model..."):
            self.model, self.processor = load(model_path)

        rprint("[bold green]Model loaded successfully![/bold green]")
        self.print_help()

    def print_help(self) -> None:
        """Print available commands."""
        mode = (
            f"server mode ({self.server_url})"
            if self.use_server
            else "local mode (model loaded in this process)"
        )
        help_text = f"""
[bold yellow]Available Commands:[/bold yellow]
• /image <path> - Load a new image for discussion
• /clear - Clear conversation history
• /help - Show this help message
• /exit - Exit the chat
• Any other input will be treated as a question or comment about the current image

[dim]Current mode: {mode}[/dim]
        """
        rprint(Panel(help_text, title="Help", border_style="blue"))

    def _probe_server(
        self, server_url: str, api_key: Optional[str]
    ) -> Optional[str]:
        """Check if a server is healthy and return the loaded model id."""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = requests.get(
                f"{server_url}/health", headers=headers, timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                loaded = data.get("loaded_model")
                if loaded:
                    return loaded
        except Exception:
            pass
        return None

    def process_image(self, image_path: str) -> bool:
        """Process an image and prepare it for the model. Returns True if successful."""
        try:
            if not os.path.exists(image_path):
                rprint(
                    f"[bold red]Error:[/bold red] Image file not found: {image_path}"
                )
                return False

            # Local mode loads the image now; server mode only needs the path.
            if not self.use_server:
                self.current_image = load_image(image_path)
            self.current_image_path = image_path
            if image_path not in self.image_paths:
                self.image_paths.append(image_path)
            rprint(f"[bold blue]Loaded image:[/bold blue] {image_path}")
            return True
        except Exception as e:
            rprint(f"[bold red]Error loading image:[/bold red] {str(e)}")
            return False

    def add_to_history(self, role: str, text: str) -> None:
        """Add a message to the conversation history."""
        content = [{"type": "text", "text": text}]
        self.history.append({"role": role, "content": content})

    def generate_response(self) -> str:
        """Generate a response from the model based on the conversation history."""
        if self.use_server:
            return self._generate_response_server()
        return self._generate_response_local()

    def _build_server_messages(self) -> List[Dict]:
        """Convert conversation history to OpenAI-compatible messages."""
        messages = []
        for msg in self.history:
            text_parts = [
                p.get("text", "")
                for p in msg.get("content", [])
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = " ".join(text_parts)
            messages.append({"role": msg["role"], "content": text})

        # Attach the current image to the last user message if present.
        if self.current_image_path and messages and messages[-1]["role"] == "user":
            text = messages[-1]["content"]
            content = [{"type": "text", "text": text}]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _encode_image_to_base64(self.current_image_path)
                    },
                }
            )
            messages[-1]["content"] = content

        return messages

    def _generate_response_server(self) -> str:
        """Generate a response by calling the running xmlx_vlm server."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = self._build_server_messages()

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.stream_kwargs.get("enable_thinking", False):
            payload["enable_thinking"] = True
        if self.stream_kwargs.get("thinking_budget") is not None:
            payload["thinking_budget"] = self.stream_kwargs["thinking_budget"]

        rprint("[bold green]Assistant:[/bold green]", end=" ", flush=True)

        text = ""
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
                    content = delta.get("content", "")
                    if content:
                        text += content
                        if self.verbose:
                            rprint(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue

        return text

    def _generate_response_local(self) -> str:
        """Generate a response using the locally loaded model."""
        chat_template_kwargs = {
            "enable_thinking": self.stream_kwargs.get("enable_thinking", False),
        }

        num_images = 1 if self.current_image_path else 0
        image = [self.current_image_path] if self.current_image_path else None

        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            self.history,
            num_images=num_images,
            **chat_template_kwargs,
        )

        rprint("[bold green]Assistant:[/bold green]", end=" ", flush=True)

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
            **self.stream_kwargs,
        ):
            text += chunk.text
            if self.verbose:
                rprint(chunk.text, end="", flush=True)

        return text

    def handle_command(self, command: str, args: str) -> bool:
        """Handle special commands. Returns True if should continue chat, False if should exit."""
        if command == "/exit":
            rprint("[bold yellow]Goodbye![/bold yellow]")
            return False
        elif command == "/help":
            self.print_help()
        elif command == "/clear":
            self.history.clear()
            self.image_paths.clear()
            if not self.use_server:
                self.prompt_cache_state = PromptCacheState()
            rprint("[bold blue]Conversation history cleared.[/bold blue]")
        elif command == "/image":
            if not args:
                rprint("[bold red]Error:[/bold red] Please provide an image path")
                return True
            self.process_image(args.strip())
        else:
            rprint(f"[bold red]Unknown command:[/bold red] {command}")
        return True

    def chat_loop(self) -> None:
        """Main chat loop for interaction."""
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    if not self.handle_command(command, args):
                        break
                    continue
                self.add_to_history("user", user_input)
                response = self.generate_response()

                if not self.verbose:
                    rprint(Panel(Markdown(response), border_style="green"))

                # Remove the eos token from the response
                response = response.replace("<end_of_utterance>", "")

                self.add_to_history("assistant", response)

            except KeyboardInterrupt:
                rprint(
                    "\n[bold yellow]Interrupted by user. Type /exit to quit.[/bold yellow]"
                )
                continue
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] {str(e)}")
                continue


def main():
    parser = argparse.ArgumentParser(description="MLX Vision Chat CLI")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to the model or model identifier",
    )
    parser.add_argument("--verbose", action="store_false", help="Enable verbose output")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs="+",
        default=None,
        help="Resize shape for the image.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Number of tokens to process per prefill step.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV size for the prompt cache.",
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits to quantize the KV cache to.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for the quantized KV cache.",
    )
    parser.add_argument(
        "--eos-tokens",
        type=str,
        nargs="+",
        default=None,
        help="EOS tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens in the detokenizer.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode in the chat template.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Maximum number of thinking tokens before forcing end-of-thinking.",
    )
    parser.add_argument(
        "--thinking-start-token",
        type=str,
        default=DEFAULT_THINKING_START_TOKEN,
        help="Token that marks the start of a thinking block.",
    )
    parser.add_argument(
        "--thinking-end-token",
        type=str,
        default=DEFAULT_THINKING_END_TOKEN,
        help="Token that marks the end of a thinking block.",
    )

    # Diffusion-model generation options (only used by block-diffusion models
    # such as DiffusionGemma). These mainly affect generation speed/quality.
    parser.add_argument(
        "--max-denoising-steps",
        type=int,
        default=None,
        help="Maximum diffusion denoising steps per token block. Lower values "
        "(e.g. 16 or 24) make responses faster but may reduce quality. "
        "Default is taken from the model config (usually 48).",
    )
    parser.add_argument(
        "--diffusion-compile",
        action="store_true",
        help="Compile the diffusion decoder graph. Can speed up generation "
        "after the first call, with automatic fallback on error.",
    )
    parser.add_argument(
        "--diffusion-static-cache",
        action="store_true",
        help="Use a static KV cache sized for the full request.",
    )
    parser.add_argument(
        "--diffusion-full-canvas",
        action="store_true",
        help="Always denoise the model's full canvas length.",
    )
    parser.add_argument(
        "--diffusion-min-canvas-length",
        type=int,
        default=None,
        help="Minimum number of tokens in each denoised block.",
    )
    parser.add_argument(
        "--diffusion-max-canvas-length",
        type=int,
        default=None,
        help="Maximum number of tokens in each denoised block.",
    )
    parser.add_argument(
        "--diffusion-sampler",
        type=str,
        choices=("entropy-bound", "confidence-threshold"),
        default="entropy-bound",
        help="Diffusion token-selection strategy.",
    )
    parser.add_argument(
        "--diffusion-threshold",
        type=float,
        default=None,
        help="Confidence threshold for the confidence-threshold sampler "
        "(default 0.9).",
    )

    # Server mode: connect to an already-running xmlx_vlm server instead of
    # loading a model into this process. By default chat.py always loads the
    # model locally.
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="URL of a running xmlx_vlm server (e.g. http://localhost:5118). "
        "If provided and reachable, chat.py will use the server's loaded "
        "model instead of loading one locally.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("XMLX_VLM_API_KEY", "x123456"),
        help="API key for server authentication "
        "(default: x123456 or XMLX_VLM_API_KEY env var).",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Deprecated: local loading is now the default. "
        "Kept for backward compatibility.",
    )

    args = parser.parse_args()

    # Build stream_generate kwargs matching generate.py's main()
    kwargs = {}

    if args.eos_tokens is not None:
        eos_tokens = []
        for token in args.eos_tokens:
            try:
                decoded_token = codecs.decode(token, "unicode_escape")
                eos_tokens.append(decoded_token)
            except (UnicodeDecodeError, UnicodeError):
                eos_tokens.append(token)
        kwargs["eos_tokens"] = eos_tokens

    if args.skip_special_tokens:
        kwargs["skip_special_tokens"] = args.skip_special_tokens

    # Thinking kwargs
    kwargs["enable_thinking"] = args.enable_thinking
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget
        kwargs["thinking_end_token"] = args.thinking_end_token
        if args.thinking_start_token is not None:
            kwargs["thinking_start_token"] = args.thinking_start_token

    # KV cache kwargs
    if args.max_kv_size is not None:
        kwargs["max_kv_size"] = args.max_kv_size
    if args.kv_bits is not None:
        kwargs["kv_bits"] = args.kv_bits
        kwargs["kv_group_size"] = args.kv_group_size
        kwargs["kv_quant_scheme"] = args.kv_quant_scheme
        kwargs["quantized_kv_start"] = args.quantized_kv_start

    if args.resize_shape is not None:
        kwargs["resize_shape"] = args.resize_shape
    if args.prefill_step_size is not None:
        kwargs["prefill_step_size"] = args.prefill_step_size

    # Diffusion-model kwargs
    if args.max_denoising_steps is not None:
        kwargs["max_denoising_steps"] = args.max_denoising_steps
    if args.diffusion_compile:
        kwargs["diffusion_compile"] = args.diffusion_compile
    if args.diffusion_static_cache:
        kwargs["diffusion_static_cache"] = args.diffusion_static_cache
    if args.diffusion_full_canvas:
        kwargs["diffusion_full_canvas"] = args.diffusion_full_canvas
    if args.diffusion_min_canvas_length is not None:
        kwargs["diffusion_min_canvas_length"] = args.diffusion_min_canvas_length
    if args.diffusion_max_canvas_length is not None:
        kwargs["diffusion_max_canvas_length"] = args.diffusion_max_canvas_length
    if args.diffusion_sampler != "entropy-bound":
        kwargs["diffusion_sampler"] = args.diffusion_sampler
        kwargs["diffusion_threshold"] = (
            0.9 if args.diffusion_threshold is None else args.diffusion_threshold
        )
    elif args.diffusion_threshold is not None:
        kwargs["diffusion_threshold"] = args.diffusion_threshold

    try:
        chat = MLXVisionChat(
            model_path=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            server_url=args.server_url,
            api_key=args.api_key,
            local_only=args.local_only,
            **kwargs,
        )
        chat.chat_loop()
    except Exception as e:
        rprint(f"[bold red]Fatal error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
