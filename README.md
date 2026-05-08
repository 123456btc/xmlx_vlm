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
| **Gradio Chat UI** | One-command launch (`--chat`) gives you a polished web interface for demos, QA, and internal tooling. |
| **Service Manager** | `service.sh` handles daemonization, health checks, log rotation, port management, and zero-downtime restarts. |
| **API Key Auth** | Rotate keys via environment variables. Enterprise-grade access control without a proxy. |
| **Model Zoo & Conversion** | One-line conversion from Hugging Face to MLX format. Native support for Qwen, LLaVA, Phi-vision, and dozens more. |
| **Batch Inference** | Process multiple images / prompts in a single pass with intelligent memory scheduling. |
| **Vision Cache** | Intelligent feature caching so repeated visual prompts do not re-encode the image. |

---

## ⚡ Technical Advantages

### 1. Thinking-Aware Constrained Generation
Most structured-output engines break when a model enters a chain-of-thought "thinking" phase. XMLX-VLM ships a **`ThinkingAwareLogitsProcessor`** that maintains JSON-Schema and regex constraints **even while the model thinks out loud**. No more corrupted schemas after a `<think>` block.

### 2. Automatic Prefix Caching with SSD Persistence (APC)
We ported the block-level prefix-caching concept from vLLM down to the MLX runtime, then added a **disk tier**: when `APC_DISK_PATH` is set, fully-computed KV blocks are written to sharded SSD files and can survive process restarts. Identical prompts warm-start in milliseconds instead of seconds.

### 3. Speculative Decoding at Scale
XMLX-VLM ships with **two speculative drafter families**:
- **DFlash** — ultra-light draft models that predict 2–3 tokens ahead with near-zero overhead.
- **MTP** (Multi-Token Prediction) — parallel draft paths for high-entropy prompts.

Combined with adaptive acceptance thresholds, speculative paths can significantly cut time-to-first-token on long-form generation.

### 4. KV-Cache Quantization
Memory is the bottleneck on unified-memory architectures. We support:
- **Uniform quantization** (4-bit, 3.5-bit, 8-bit)
- **TurboQuant** — an adaptive scheme that preserves attention precision where it matters and compresses where it does not.

### 5. MoE Top-K Override
Mixture-of-Experts models are the new standard, but default routing wastes cycles. XMLX-VLM exposes **dynamic top-k override** so you can trade a fraction of accuracy for latency wins in latency-sensitive workloads.

### 6. Multi-Format Reasoning Parsers
Out-of-the-box parsers for **Qwen3, DeepSeek-R1, Gemma4, GLM4, GPT-OSS, and Harmony** reasoning formats. The server automatically strips internal monologue before returning structured JSON, while still exposing the raw chain-of-thought when you need it.

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
