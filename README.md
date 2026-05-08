# XMLX-VLM

<p align="center">
  <strong>The Production-Ready Vision-Language Inference Engine for Apple Silicon</strong>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform: macOS" src="https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

---

## 🚀 Why XMLX-VLM?

**XMLX-VLM** is an opinionated hard-fork built for teams that need **more than just inference** on Apple Silicon — they need a **complete, deployable, observable Vision-Language stack**.

We stand on the shoulders of two excellent open-source projects — [**Blaizzy/mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) for the core VLM loader and [**vllm-mlx**](https://github.com/vllm-project/vllm) community patterns for serving infrastructure — then added the missing layers that turn a model loader into a **production system**: a scalable API server, speculative decoding, structured output, tool calling, embedding & reranking engines, automatic prefix caching with SSD persistence, and a built-in operations suite.

> If you run VLMs on Mac Studio, Mac Pro, or a fleet of M4 Max machines — this is the stack that closes the gap between "it works on my laptop" and "it runs in production."

---

## 🎯 Product Features

| Capability | What You Get |
|------------|-------------|
| **OpenAI-Compatible Server** | Drop-in replacement for OpenAI vision endpoints. Supports `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`, and streaming SSE out of the box. |
| **Anthropic / Claude Compatible** | Native `/v1/messages` endpoint with tool-use and streaming support. Swap Claude SDK base URL and it just works. |
| **Gradio Chat UI** | One-command launch (`--chat`) gives you a polished web interface for demos, QA, and internal tooling. |
| **Service Manager** | `service.sh` handles daemonization, health checks, log rotation, port management, and zero-downtime restarts. |
| **API Key Auth** | Rotate keys via environment variables. Enterprise-grade access control without a proxy. |
| **Model Zoo & Conversion** | One-line conversion from Hugging Face to MLX format. Native support for Qwen, LLaVA, Phi-vision, and dozens more. |
| **Batch Inference** | Process multiple images / prompts in a single pass with intelligent memory scheduling. |
| **Vision Cache** | Intelligent feature caching so repeated visual prompts do not re-encode the image. |

---

## ⚡ Technical Advantages

### 1. Thinking-Aware Constrained Generation (Logits-Level Lifecycle Management)

This is the architectural capability that no other open-source inference engine offers today.

Modern reasoning models (Qwen3, DeepSeek-R1, Gemma4, etc.) emit a chain-of-thought inside `<think>...</think>` tags before producing the final answer. The industry has two broken approaches to structured output with these models:

| Approach | Problem |
|----------|---------|
| **Constraint everywhere** | The JSON Schema is enforced during the thinking phase. The model cannot reason freely — it starts hallucinating structure inside its own monologue. |
| **Post-process parsing** | The model generates unconstrained text, then you pray a regex extracts valid JSON from the tail. Fragile, non-deterministic, and unusable for tool calling. |

XMLX-VLM solves this at the **logits level** with a four-phase state machine that lives inside the token loop:

```
IDLE ──► THINKING ──► TRANSITIONING ──► CONTENT
```

- **THINKING** — The model is free. No JSON Schema, no regex mask, no constraints. It can ramble, backtrack, and explore. We only count tokens against a configurable `thinking_budget`.
- **TRANSITIONING** — When the budget expires (or the model naturally emits the end token), we **force the exact end-token sequence via logits masking** (`-inf` everywhere except the target token). This guarantees a clean, deterministic exit from the thinking span — no half-open tags, no drift.
- **CONTENT** — The instant the thinking span closes, control is handed to the inner logits processor (JSON Schema, regex, or tool-parser). The very first content token is already constrained.

Key implementation details:
- **BoundedSuffixMatcher** with overlapping-prefix recovery detects `<think>` / `</think>` token sequences in O(1) amortized time.
- **Snapshot/rollback** support means the state machine survives speculative-decoding rejections and dynamic batching without desync.
- **Content-phase mask** prevents `<think>` tokens from leaking back into the final output after transition.
- **Retirement signal** — once the processor reaches CONTENT with no inner constraint, it signals the engine to drop itself and re-enable MTP for the remaining generation.

Result: your model can think for 512 tokens, then emit a perfectly valid JSON object or tool call — with zero post-processing.

### 2. Multi-Format Reasoning Parsers + Tool-Call Promotion

Six dedicated streaming parsers handle the post-processing side:

| Parser | Format | Special Handling |
|--------|--------|-----------------|
| **Qwen3** | `<think>...</think>` | Implicit reasoning (prompt-injected `<think>`) |
| **DeepSeek-R1** | `<think>...</think>` | Missing start-tag detection |
| **Gemma4** | `<start_of_thought>...<end_of_thought>` | Multi-turn thought blocks |
| **GLM4** | `<|channel>thought...<channel|>` | Channel-based reasoning |
| **GPT-OSS** | Custom delimiter | OSS reasoning trace format |
| **Harmony** | Structured thinking | Multi-step reasoning chains |

Each parser implements a streaming state machine (`pre_think → thinking → content`) that splits delta chunks into `reasoning` and `content` streams in real time. The server exposes both via OpenAI-compatible `reasoning_content` and Anthropic-compatible `thinking` blocks.

**Tool-Call Promotion**: When a `<tool_call>` block appears inside the thinking span, the parser automatically promotes it to the content stream. Closed tool calls are extracted via regex and appended to the final content; unclosed calls are flushed at stream end with a warning. This means a reasoning model can "think about calling a tool" and actually call it — without the tool XML leaking into the reasoning channel.

### 3. Automatic Prefix Caching with SSD Persistence (APC)

We ported the block-level prefix-caching concept from vLLM down to the MLX runtime, then added a **disk tier**:
- KV cache is split into 16-token blocks, each identified by a chained hash (`H(prev_hash, token_slice, image_hash)`).
- LRU eviction with reference counting keeps hot blocks in memory.
- When `APC_DISK_PATH` is set, full blocks are written to sharded SSD files and survive process restarts.
- Identical prompts warm-start in milliseconds instead of seconds — even after a server restart.

### 4. Speculative Decoding at Scale

XMLX-VLM ships with **two speculative drafter families**:
- **DFlash** — ultra-light draft models that predict 2–3 tokens ahead with near-zero overhead.
- **MTP** (Multi-Token Prediction) — parallel draft paths for high-entropy prompts.

Combined with adaptive acceptance thresholds, speculative paths can significantly cut time-to-first-token on long-form generation.

### 5. KV-Cache Quantization

Memory is the bottleneck on unified-memory architectures. We support:
- **Uniform quantization** (4-bit, 3.5-bit, 8-bit)
- **TurboQuant** — an adaptive scheme that preserves attention precision where it matters and compresses where it does not.

### 6. MoE Top-K Override

Mixture-of-Experts models are the new standard, but default routing wastes cycles. XMLX-VLM exposes **dynamic top-k override** so you can trade a fraction of accuracy for latency wins in latency-sensitive workloads.

### 7. Tool Calling & MCP (Model Context Protocol)

- Automatic tool-parser inference from the model processor
- Pluggable tool modules
- Built-in **MCP Manager** for connecting to external data sources, IDEs, and agent frameworks via stdio or SSE

### 8. Embedding & Rerank Engines

A single process serves **vision-language chat**, **text embeddings**, and **reranking**. No need to run a separate embedding micro-service — reduce network hops and context-switching overhead.

### 9. Apple-Silicon Native Optimization

- Flash Attention via `mx.fast.scaled_dot_product_attention`
- Metal kernel fusion for vision encoders
- Hardware-aware memory budgeting (M1 → M4 Ultra profiles baked in)
- Unified-memory zero-copy between CPU pre-processing and GPU inference

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (OpenAI SDK, LangChain, curl, Gradio UI, Agent frameworks) │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      XMLX-VLM Server                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │   Chat API  │ │ Embeddings  │ │  Rerank / Classify  │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Tool Parse │ │    MCP      │ │ Structured Output   │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Inference Core                           │
│  ┌──────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │   Generate   │ │   Batch     │ │  Speculative Draft  │  │
│  └──────────────┘ └─────────────┘ └─────────────────────┘  │
│  ┌──────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │ KV Quantize  │ │  MoE Top-K  │ │  Vision Cache       │  │
│  └──────────────┘ └─────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    MLX / Metal Runtime                       │
│         (Apple Silicon Unified Memory & GPU Cores)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚦 Quick Start

### Install

```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Start the Server

```bash
# Basic server
./service.sh start

# Server + Chat UI
./service.sh start --chat

# With speculative decoding & KV quantization
./service.sh start --chat \
  --draft-model mlx-community/Qwen3.6-35B-A3B-DFlash \
  --draft-kind dflash \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant
```

### Call the API

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/qwen3.6-35B-A3B-4bit",
    "messages": [
      {"role": "user", "content": "Describe this image"}
    ],
    "stream": true
  }'
```

---

## 🛠 Operations & Observability

```bash
# Check health, model loaded, PIDs, ports
./service.sh status

# Tail live logs
./service.sh logs server
./service.sh logs chat

# Zero-downtime restart
./service.sh restart --chat
```

- **PID tracking** with orphan-process fallback
- **Port collision** auto-resolution
- **Health endpoint** at `/health` for load-balancer integration
- **Structured logs** with rotation-friendly output

---

## 🧩 Supported Models

- **Qwen-VL / Qwen2-VL / Qwen3.6-VL** (recommended)
- **LLaVA 1.5 / 1.6 / NeXT**
- **Phi-3 / Phi-4 Vision**
- **InternVL2**
- **MiniCPM-V**
- **DeepSeek-VL**
- ... and any Hugging Face model with an MLX community port.

---

## 🏛 Acknowledgments & Lineage

XMLX-VLM is a **hard-fork** that consciously builds on top of several outstanding open-source projects:

| Project | What We Borrowed | What We Added |
|---------|-----------------|---------------|
| [**Blaizzy/mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) | Core VLM model loading, weight conversion, and MLX generation primitives | Production server, speculative decoding, structured output, tool calling, MCP, embedding/rerank engines |
| [**vllm-mlx**](https://github.com/vllm-project/vllm) (community patterns) | Metrics design, model registry patterns, hardware detection concepts | SSD-persistent APC cache, Apple-Silicon-specific memory budgeting, unified CLI |
| **Anthropic SDK** | Message format and tool-use schema | First-class `/v1/messages` endpoint with streaming, tool results, and thinking blocks |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp) | Mixed quantization predicates (Q4_K_M-style strategies) | Integration into the MLX conversion pipeline |
| [**Hugging Face Transformers**](https://github.com/huggingface/transformers) | Tokenizer utilities, sampling logic, AutoModel loading | MLX-native weight conversion, batch streaming, thinking-aware processors |

We are deeply grateful to the authors and communities behind these projects. XMLX-VLM exists because they laid the groundwork.

---

## 🤝 Community & Roadmap

- [x] OpenAI-compatible REST API
- [x] Speculative decoding (DFlash + MTP)
- [x] KV-cache quantization
- [x] Tool calling & MCP
- [x] Embedding & Rerank engines
- [x] Automatic Prefix Caching with SSD persistence
- [x] Thinking-aware structured generation
- [ ] LoRA serving (hot-swap adapters)
- [ ] Multi-GPU pipeline parallelism
- [ ] Community benchmark suite (contributions welcome!)

**License:** MIT  
**Origin:** Hard-fork from [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — rebuilt for production workloads.

---

<p align="center">
  <strong>Built for teams who ship.</strong><br>
  Star ⭐ the repo if XMLX-VLM accelerates your vision pipeline.
</p>
