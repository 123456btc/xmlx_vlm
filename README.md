# XMLX-VLM

<p align="center">
  <strong>Privacy-First Local Vision-Language AI & Institutional-Grade Fully Private Local Quantitative Trading Platform</strong>
</p>

<p align="center">
  <em>Apple Silicon-native inference engine. Zero data leaves your machine. Designed for highly sensitive document parsing and completely private local AI algorithmic/model trading.</em>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform: macOS" src="https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <strong>🇺🇸 English</strong> | <a href="README.zh.md">🇨🇳 中文</a> | <a href="README.ja.md">🇯🇵 日本語</a> | <a href="README.ko.md">🇰🇷 한국어</a>
</p>

---

## 🚀 Why XMLX-VLM?

**For professionals who handle sensitive data, privacy is not a feature — it is the baseline.**

Legal documents, medical records, government files, proprietary research, trading algorithms — the moment they pass through a cloud API, you lose control. They become training data. They get logged. They get subpoenaed.

**XMLX-VLM** is a **local-first, production-grade Vision-Language inference engine** that runs entirely on Apple Silicon. It reads documents, parses images, reasons through complex problems, and emits structured outputs — **without a single network call**.

No cloud subscription. No data retention policy. No third-party terms of service. Just your Mac, your data, and your models.

> **Data sovereignty is the architecture. Everything else is built on top of it.**

### 🧬 The AFRE Ecosystem

XMLX-VLM is the **private AI brain** of the **AFRE (AI Factor Research Engine)** ecosystem — a domain-first, agent-augmented quantitative research platform built on DDD, Hexagonal Architecture, and Clean Architecture.

AFRE researches the **genealogy of market factors**: why they were invented, how they spread, why they decayed, and what broken assumptions can generate modern hypotheses. XMLX-VLM powers AFRE's agent runtime by providing:

| AFRE Capability | What XMLX-VLM Enables Locally |
|-----------------|------------------------------|
| **Factor Genealogy Intelligence** | Parse research PDFs and chart images; extract structured factor histories from visual documents |
| **Inventor Thinking Simulation** | Deep reasoning (`<think>` mode) to simulate a factor creator's constraints, incentives, and knowledge stack |
| **Hypothesis-Centric Research** | JSON-Schema-constrained output ensures every generated factor variant carries a testable hypothesis and broken-assumption trace |
| **Reproducible Experimentation** | Tool calling + MCP connects to local backtesters and signal generators; experiments run on your hardware, auditable by design |
| **Anti-Overfitting Governance** | Structured output enforces walk-forward params, regime splits, and turnover penalties as machine-readable schema |
| **Knowledge Evolution** | Embedding + Rerank indexes validated findings into a governed, queryable local knowledge base |
| **Multi-Agent Parallel Research** | Continuous batching + speculative decoding lets multiple AI workers reason independently without latency collapse |

**AFRE is the methodology. XMLX-VLM is the私有化推理层 that makes it possible.**

While AFRE represents XMLX-VLM's flagship implementation in quantitative finance, the same local-privacy architecture serves legal, healthcare, government, and enterprise R&D domains equally.

---

## 🎯 Who Is This For?

| Domain | Sensitive Data | What XMLX-VLM Does Locally |
|--------|---------------|---------------------------|
| **Quantitative Finance** | Proprietary factors, internal research reports, alpha signals | Parse PDF reports and chart images; reason through factor hypotheses; emit structured factor definitions; call local backtesters via MCP |
| **Legal** | Case files, contracts, discovery documents, client communications | Analyze document images and scans; extract structured clauses; reason through legal arguments; generate redlined summaries |
| **Government** | Classified briefings, policy drafts, citizen records, intelligence imagery | Process sensitive imagery and scanned documents; structured output for intelligence reports; full audit trail on local hardware |
| **Healthcare** | Patient records, medical imaging, clinical notes, lab results | Parse medical document images; reason through differential diagnoses; structured output for clinical summaries; HIPAA-compliant by architecture |
| **Enterprise R&D** | Trade secrets, patent drafts, experimental data, internal memos | Vision-language understanding of technical diagrams; reasoning through research hypotheses; structured output for experiment designs |

---

## 🎯 Core Capabilities

| Capability | What You Get |
|------------|-------------|
| **Local Document Intelligence** | Feed PDFs, scanned documents, screenshots, and image-heavy reports directly into the model. No OCR SaaS. No cloud vision API. Your documents never leave localhost. |
| **Structured Output with Reasoning** | Enable `thinking` mode for deep reasoning, then enforce JSON-Schema constraints on the final output. Perfect for audit-ready reports, factor definitions, and clinical summaries. |
| **Dual-Protocol API** | One server speaks both OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) protocols. Drop in as the backend for Cursor, Claude Code, LangChain, PydanticAI — all traffic stays on `localhost:5118`. |
| **Local Tool Calling & MCP** | Connect to local databases, backtesters, EHR systems, case-management tools, and document pipelines via MCP. The model calls your tools; your data never leaves the machine. |
| **Embedding & Rerank for Private Knowledge** | Index internal documents, research notes, case files, and patient histories. Semantic search over your proprietary knowledge base — with zero cloud exposure. |
| **AI Trader (Local Quant Assistant)** | Chat with a local AI trading assistant connected to Hyperliquid via a persistent WebSocket feed. It auto-watches the top 30 perpetuals by 24h volume, computes real-time indicators, emits threshold-based alerts, renders charts, and simulates trades — all on your machine. |
| **SSD-Persistent Prefix Cache** | Repeated analysis of the same document or system prompt warm-starts in milliseconds, even after server restart. The cache lives on your SSD, not someone else's server. |
| **AI Trader Chat UI** | One-command launch (`--chat` on port `5119`) with secure KMS credentials vault, real-time monitoring of Hyperliquid assets, positions, and trade history. |
| **Service Manager** | `service.sh` handles daemonization, health checks, log rotation, port management, and zero-downtime restarts. |
| **API Key Auth** | Rotate keys via environment variables. Enterprise-grade access control without a proxy. |

---

## ⚡ Technical Advantages

### 1. Thinking-Aware Constrained Generation

Most reasoning models emit a chain-of-thought inside `<think>...</think>` tags. Standard structured-output engines either break during thinking or corrupt the JSON. XMLX-VLM manages a **four-phase logits state machine** (`IDLE → THINKING → TRANSITIONING → CONTENT`) at the token level:

- **THINKING** — The model reasons freely. No JSON constraints. It can explore assumptions, edge cases, and contradictions.
- **TRANSITIONING** — When the budget expires, we **force the end-token sequence via logits masking** (`-inf` everywhere except the target). Clean, deterministic exit.
- **CONTENT** — The instant thinking closes, control hands to the inner JSON-Schema processor. The first content token is already constrained.

Result: your model can think for 512 tokens about a legal argument or medical differential, then emit a perfectly valid structured JSON — with zero post-processing.

### 2. Automatic Prefix Caching with SSD Persistence (APC)

When you iterate on the same document or system prompt, XMLX-VLM reuses the KV cache across requests. For hybrid SSM/attention models (Qwen3.5 DeltaNet, Nemotron-H), the **recurrent state is also snapshotted and persisted to SSD**:

- Block-level KV cache with chained hashing
- LRU + reference-count eviction
- `APC_DISK_PATH` writes full blocks to sharded SSD files — **survives process restart**
- Identical prompts warm-start in milliseconds, even after a server restart

### 3. Multi-Format Reasoning Parsers + Tool-Call Promotion

Six streaming parsers handle reasoning extraction for Qwen3, DeepSeek-R1, Gemma4, GLM4, GPT-OSS, and Harmony. When a `<tool_call>` block appears inside the thinking phase, it is **automatically promoted to the content stream** — so the model can "think about calling a tool" and actually call it.

### 4. Tool-Call Auto-Recovery + Jump-Forward Decoding

Quantized models degrade after multiple tool rounds. XMLX-VLM adds two defenses:

- **Auto-Recovery** — Repairs unclosed XML tags, balances truncated JSON braces, and extracts bare JSON objects from garbled output.
- **Jump-Forward Logits Bias** (`--enable-tool-logits-bias`) — Additive bias on tool-related token IDs pushes the model faster into structured format, cutting time-to-first-tool-token.

### 5. Speculative Decoding at Scale

- **DFlash** — ultra-light draft models predict 2–3 tokens ahead
- **MTP** (Multi-Token Prediction) — parallel draft paths for high-entropy prompts

Cuts latency on long-form document analysis and reasoning tasks.

### 6. KV-Cache Quantization

- **Uniform** (4-bit, 3.5-bit, 8-bit)
- **TurboQuant** — adaptive scheme preserving attention precision where it matters

Run 70B-class vision models on a 128 GB Mac Studio with headroom for long-context documents.

### 7. MoE Top-K Override

Dynamic top-k override lets you trade a fraction of accuracy for latency wins in interactive analysis sessions.

### 8. Apple-Silicon Native Optimization

- Flash Attention via `mx.fast.scaled_dot_product_attention`
- Metal kernel fusion for vision encoders
- Hardware-aware memory budgeting (M1 → M5 Max profiles baked in)
- Unified-memory zero-copy between CPU pre-processing and GPU inference

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              AFRE (AI Factor Research Engine)                │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ Factor       │ │ Inventor     │ │ Hypothesis-Centric  │  │
│  │ Genealogy    │ │ Thinking     │ │ Research            │  │
│  │ Intelligence │ │ Simulation   │ │                     │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ Anti-Overfit │ │ Knowledge    │ │ Multi-Agent         │  │
│  │ Governance   │ │ Evolution    │ │ Parallel Research   │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Private AI Agents & Clients                 │
│  (Cursor, Claude Code, LangChain, PydanticAI, AFRE agents,  │
│   AI Trader — local quant assistant)                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   XMLX-VLM Server (local)                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │   Chat API  │ │ Embeddings  │ │  Rerank / Classify  │   │
│  │ (OpenAI +   │ │  (private   │ │  (document / case   │   │
│  │  Anthropic) │ │   memory)   │ │   retrieval)        │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Tool Parse │ │    MCP      │ │ Structured Output   │   │
│  │ (local DBs) │ │ (internal   │ │ (audit-ready JSON)  │   │
│  │             │ │  systems)   │ │                     │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Inference Core                           │
│  ┌──────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │   Generate   │ │   Batch     │ │  Speculative Draft  │  │
│  │  (reasoning) │ │  (docs)     │ │  (latency cut)      │  │
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

### One-Click Install (macOS Apple Silicon)

For fresh Macs or users without dev tools:

```bash
curl -fsSL https://raw.githubusercontent.com/123456btc/xmlx_vlm/master/install.sh | bash
```

This script automatically:
- ✅ Checks Apple Silicon (M1/M2/M3/M4/M5)
- ✅ Installs Xcode Command Line Tools (if missing)
- ✅ Installs Homebrew (if missing)
- ✅ Installs Python 3.12 (if < 3.10)
- ✅ Installs `uv` (fast Python package manager)
- ✅ Clones the repo and creates virtual environment
- ✅ Installs MLX, XMLX-VLM, and all dependencies
- ✅ Sets default API key (`x123456`) and environment variables
- ✅ Optionally pre-downloads the default model (~20GB)
- ✅ Starts the server

**Expected time:** 10-20 minutes on a fresh Mac (mostly downloading models).

### Manual Install

If you prefer manual setup:

```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Start the Server

```bash
# Default start — server only (headless, no Chat UI)
./service.sh start

# Enable chat UI alongside server
./service.sh start --chat

# Override default API key + KV quantization for production workloads
XMLX_VLM_API_KEY=mykey ./service.sh start --kv-bits 3.5 --kv-quant-scheme turboquant

# Enable per-layer mixed KV quantization and sequence concurrency bounding
./service.sh start --kv-bits-per-layer "0:8,1:8,-1:8,default:3.5" --max-num-seqs 16

# Enable tool-call acceleration for MCP-heavy workflows
./service.sh start --enable-tool-logits-bias

# Disable speculative decoding entirely (fallback to standard generation)
XMLX_VLM_DRAFT_MODEL="" XMLX_VLM_DRAFT_KIND="" ./service.sh start
```

### Call the API (Local Only)

```bash
curl http://localhost:5118/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
    "messages": [
      {"role": "user", "content": "Analyze the attached document and extract structured findings"}
    ],
    "stream": true
  }'
```

### Launch AI Trader (Local Quant Assistant)

```bash
# Start the server first
./service.sh start

# Chat with your local trading assistant
xmlx_vlm.ai-trader

# Or run a single prompt
xmlx_vlm.ai-trader --prompt "分析 BTC 走势"
```

AI Trader connects to Hyperliquid through a persistent WebSocket market-data service. It auto-subscribes to the top 30 perpetuals by 24h volume, maintains an in-memory state machine, computes real-time indicators, and emits threshold-based alerts. To optimize local decisions, AI Trader integrates a **single-request Bull/Bear adversarial debate mechanism** to prevent unilateral bias, and automatically starts an **asynchronous post-trade reflection task** upon position closure to save trade insights into local SQLite, enabling closed-loop adaptive learning. Tools read the local snapshot first and fall back to REST when needed.

---

## 🤖 Agent Client Integration

XMLX-VLM is designed to be the **local backend for coding agents and AI assistants**. Because agent clients resend the full conversation history every turn, enabling APC disk persistence (`APC_DISK_PATH`) is strongly recommended — it eliminates repeated prefill overhead after the first expensive warm-up.

### Claude Code (Anthropic-compatible)

Create `~/.local/bin/claude-xmlx`:

```bash
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${XMLX_ANTHROPIC_BASE_URL:-http://127.0.0.1:5118}"
export ANTHROPIC_AUTH_TOKEN="${XMLX_API_KEY:-x123456}"
export ANTHROPIC_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"

export ANTHROPIC_CUSTOM_MODEL_OPTION="mlx-community/diffusiongemma-26B-A4B-it-4bit"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="XMLX-VLM Local Qwen3.6"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="Local MLX inference via xmlx_vlm"

export ANTHROPIC_DEFAULT_SONNET_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"
export ANTHROPIC_DEFAULT_OPUS_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"
export CLAUDE_CODE_SUBAGENT_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Start the server with disk KV cache:

```bash
APC_ENABLED=1 APC_DISK_PATH=/tmp/xmlx-apc ./service.sh start
```

### Cline / Continue.dev (OpenAI-compatible)

In your VS Code settings or `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "XMLX-VLM Local",
      "provider": "openai",
      "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
      "apiBase": "http://localhost:5118/v1",
      "apiKey": "x123456"
    }
  ]
}
```

### Aider (OpenAI-compatible)

```bash
export OPENAI_API_BASE=http://localhost:5118/v1
export OPENAI_API_KEY=x123456
aider --model openai/mlx-community/diffusiongemma-26B-A4B-it-4bit
```

### Cursor (OpenAI-compatible)

In Cursor Settings → Models → Add Model:
- **Base URL**: `http://localhost:5118/v1`
- **API Key**: `x123456`
- **Model**: `mlx-community/diffusiongemma-26B-A4B-it-4bit`

### Pi (pi.dev)

Pi is a local-first coding agent that pairs well with XMLX-VLM. Add the provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "xmlx-local": {
      "name": "XMLX-VLM (local)",
      "baseUrl": "http://localhost:5118/v1",
      "api": "openai-completions",
      "apiKey": "x123456",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "qwen",
        "requiresReasoningContentOnAssistantMessages": false
      },
      "models": [
        {
          "id": "mlx-community/Qwen3.8-27B-4bit",
          "name": "Qwen 3.8 27B 4bit (XMLX-VLM local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 32768,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Then set as default in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "xmlx-local",
  "defaultModel": "mlx-community/Qwen3.8-27B-4bit"
}
```

### Recommended Agent Server Flags

```bash
# Full agent stack: disk APC + tool acceleration + long context
APC_ENABLED=1 \
APC_DISK_PATH=/tmp/xmlx-apc \
XMLX_VLM_ENABLE_TOOL_LOGITS_BIAS=1 \
./service.sh start --ctx 100000
```

> **Tip**: Agent clients often send 10k–30k tokens for the initial system prompt. With `APC_DISK_PATH`, this prefix is written to SSD during the first prefill and restored instantly on subsequent sessions — even after a server restart.

#### ds4-style Full Session Checkpointing (new)

Inspired by [antirez/ds4](https://github.com/antirez/ds4)'s disk KV cache design, XMLX-VLM now saves the **full conversation KV state** to SSD at the end of every generation — not just the input prefix. This means:

- **Multi-turn conversations skip the entire re-prefill**, including all previous assistant responses, not just the system prompt.
- Each saved checkpoint stores the KV tensors **plus** the next-token log-softmax vector (ds4's "logits snapshot"), enabling future optimizations like single-step-skip on restore.
- The checkpoint key is the full token sequence hash (input + output), so multi-client workloads don't collide.

Two checkpoints are written per generation turn when `APC_DISK_PATH` is set and `APC_ENABLED=1`:

| Checkpoint | Token coverage | Written at |
|---|---|---|
| **prefix** | input tokens only | after prefill (n=0) |
| **session** | input + all generated tokens | after generation ends |

The session checkpoint is what turns the SSD into a "first-class KV citizen" — subsequent turns in the same conversation find a longer cached prefix and skip far more computation. On MLA-compressed models (Kimi K25, DeepSeek VL V2) the KV is small enough that even long-context sessions fit comfortably on SSD.

---

## 🛠 Operations & Observability

```bash
# Check health, model loaded, PIDs, ports
./service.sh status

# Tail live logs
./service.sh logs server
./service.sh logs chat

# Zero-downtime restart
./service.sh restart
```

- **PID tracking** with orphan-process fallback
- **Port collision** auto-resolution
- **Health endpoint** at `/health`
- **Structured logs** with rotation-friendly output

### Automatic idle KV-cache release

Long conversations and prefix caching can keep the KV cache resident long after the last request. After the model has been idle for **5 minutes** (default), XMLX-VLM automatically drops the active batch cache, clears the APC memory tier, and flushes the MLX cache pool — while keeping the model weights loaded so the next request only needs to re-prefill.

```bash
# Disable auto-release (keep KV cache in memory indefinitely)
./service.sh start --idle-kv-release-timeout 0

# Release KV cache after 60 seconds of inactivity
./service.sh start --idle-kv-release-timeout 60

# Or via environment variable
XMLX_VLM_IDLE_KV_RELEASE_TIMEOUT=120 ./service.sh start
```

Check `/health` for `idle_kv_release_timeout` and `idle_kv_released` status.

---

## 🧩 Supported Models

- **Qwen-VL / Qwen2-VL / Qwen3.6-VL** (recommended for CJK documents)
- **LLaVA 1.5 / 1.6 / NeXT**
- **Phi-3 / Phi-4 Vision**
- **InternVL2**
- **MiniCPM-V**
- **DeepSeek-VL**
- ... and any Hugging Face model with an MLX community port.

---

## 🏛 Acknowledgments & Lineage

XMLX-VLM is a **hard-fork** that consciously builds on several outstanding open-source projects:

| Project | What We Borrowed | What We Added |
|---------|-----------------|---------------|
| [**Blaizzy/mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) | Core VLM model loading, weight conversion, and MLX generation primitives | Production server, speculative decoding, structured output, tool calling, MCP, embedding/rerank engines |
| [**vllm-mlx**](https://github.com/vllm-project/vllm) (community patterns) | Metrics design, model registry patterns, hardware detection concepts | SSD-persistent APC cache, Apple-Silicon-specific memory budgeting, unified CLI |
| [**Rapid-MLX**](https://github.com/raullenchai/Rapid-MLX) | Tool-call auto-recovery, jump-forward logits bias, DeltaNet state snapshots | Adapted auto-recovery and jump-forward decoding; inspired hybrid-cache architecture roadmap |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp) | Mixed quantization predicates (Q4_K_M-style strategies) | Integration into the MLX conversion pipeline |
| [**Hugging Face Transformers**](https://github.com/huggingface/transformers) | Tokenizer utilities, sampling logic, AutoModel loading | MLX-native weight conversion, batch streaming, thinking-aware processors |

We are deeply grateful to the authors and communities behind these projects. XMLX-VLM exists because they laid the groundwork.

---

## 🤝 Community & Roadmap

- [x] Dual-protocol REST API (OpenAI + Anthropic/Claude)
- [x] Speculative decoding (DFlash + MTP)
- [x] KV-cache quantization
- [x] Tool calling & MCP
- [x] Embedding & Rerank engines
- [x] Automatic Prefix Caching with SSD persistence
- [x] Thinking-aware structured generation
- [x] Tool-call auto-recovery + Jump-Forward decoding
- [x] LoRA training & adapter loading
- [~] LoRA hot-swap serving (training & loading work; runtime adapter switching needs server API)
- [~] Tensor / Pipeline Parallelism (utils layer supports `mx.distributed`; server integration pending)
- [x] Built-in benchmark (TTFT / TPOT / TPS / memory)
- [ ] Cross-engine benchmark suite (contributions welcome!)

**License:** MIT  
**Origin:** Hard-fork from [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — rebuilt for production workloads.

---

<p align="center">
  <strong>Your data. Your models. Your privacy.</strong><br>
  Star ⭐ the repo if XMLX-VLM protects your sensitive pipeline.
</p>
