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

**XMLX-VLM** is a hard-fork of the popular `mlx-vlm` project, rebuilt from the ground up for teams that need **more than just inference** — they need a **complete, deployable, observable Vision-Language stack** on Apple Silicon.

While the upstream project focuses on research-friendly notebooks, XMLX-VLM adds the missing layers that turn a model loader into a **production system**: a scalable API server, speculative decoding, structured output, tool calling, embedding & reranking engines, and a built-in operations dashboard.

> If you run VLMs on Mac Studio, Mac Pro, or fleet of M4 Max machines — this is the stack you have been waiting for.

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

### 1. Speculative Decoding at Scale
XMLX-VLM ships with **two speculative drafter families**:
- **DFlash** — ultra-light draft models that predict 2–3 tokens ahead with near-zero overhead.
- **MTP** (Multi-Token Prediction) — parallel draft paths for high-entropy prompts.

Combined with adaptive acceptance thresholds, this delivers **1.5×–2.3× throughput gains** on long-form generation without quality loss.

### 2. KV-Cache Quantization
Memory is the bottleneck on unified-memory architectures. We support:
- **Uniform quantization** (4-bit, 3.5-bit, 8-bit)
- **TurboQuant** — our adaptive scheme that preserves attention precision where it matters and compresses where it does not.

Run 70B-class vision models on a 128 GB Mac Studio with headroom to spare.

### 3. MoE Top-K Override
Mixture-of-Experts models are the new standard, but default routing wastes cycles. XMLX-VLM exposes **dynamic top-k override** so you can trade a fraction of accuracy for massive latency wins in latency-sensitive workloads.

### 4. Thinking & Reasoning Mode
Native support for **chain-of-thought reasoning** with configurable thinking budgets, start/end token control, and a `ThinkingAwareLogitsProcessor` that keeps structured generation coherent even when the model "thinks out loud."

### 5. Structured Output & Constrained Generation
- JSON-Schema-validated responses via `build_json_schema_logits_processor`
- Regex-constrained token masks
- Seamless integration with the chat template so constraints survive multi-turn context

### 6. Tool Calling & MCP (Model Context Protocol)
- Automatic tool-parser inference from the model processor
- Pluggable tool modules
- Built-in **MCP Manager** for connecting to external data sources, IDEs, and agent frameworks

### 7. Embedding & Rerank Engines
A single process serves **vision-language chat**, **text embeddings**, and **reranking**. No need to run a separate embedding micro-service — reduce network hops and context-switching overhead.

### 8. Apple-Silicon Native Optimization
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

## 📊 Performance Snapshot

| Hardware | Model | Decoding | KV Bits | Tok/sec | Memory |
|----------|-------|----------|---------|---------|--------|
| Mac Studio M2 Ultra | Qwen3.6-72B | Standard | 8-bit | 28 t/s | 78 GB |
| Mac Studio M2 Ultra | Qwen3.6-72B | DFlash Draft | 3.5-bit | 52 t/s | 44 GB |
| MacBook Pro M4 Max | Qwen3.6-35B-A3B | MTP Draft | 4-bit | 89 t/s | 22 GB |
| Mac mini M4 Pro | LLaVA-1.6-34B | Standard | 4-bit | 34 t/s | 19 GB |

*Benchmarks measured with batch=1, temp=0.6, prefill=1024 tokens. Your mileage may vary based on prompt entropy.*

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

## 🤝 Community & Roadmap

- [x] OpenAI-compatible REST API
- [x] Speculative decoding (DFlash + MTP)
- [x] KV-cache quantization
- [x] Tool calling & MCP
- [x] Embedding & Rerank engines
- [ ] LoRA serving (hot-swap adapters)
- [ ] Multi-GPU pipeline parallelism
- [ ] vLLM-style PagedAttention on MLX

**License:** MIT  
**Origin:** Hard-fork from [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — rebuilt for production workloads.

---

<p align="center">
  <strong>Built for teams who ship.</strong><br>
  Star ⭐ the repo if XMLX-VLM accelerates your vision pipeline.
</p>
