# XMLX-VLM: Local AI Trading OS & Private Quant Terminal

<p align="center">
  <strong>The World's First Apple Silicon-Native, Fully Private Autonomous AI Trading OS</strong>
</p>

<p align="center">
  <em>Zero Cloud Leakage. $0 Token Bills. Sub-Millisecond In-Memory Market Infrastructure. Multi-Agent Fleet with Institutional Risk Guardrails — Running 100% Locally on Your Mac.</em>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="https://github.com/123456btc/xmlx_vlm/releases/tag/v1.0.0"><img alt="Release: v1.0.0" src="https://img.shields.io/badge/Release-v1.0.0%20Latest-brightgreen"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform: macOS" src="https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon%20M1--M5)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <strong>🇺🇸 English</strong> | <a href="README.zh.md">🇨🇳 中文</a> | <a href="README.ja.md">🇯🇵 日本語</a> | <a href="README.ko.md">🇰🇷 한국어</a>
</p>

---

## ⚡ Why a Local Private AI Trading OS?

Most existing AI trading bots rely on **cloud LLM APIs** (OpenAI, Anthropic, DeepSeek). In live quantitative trading, cloud reliance introduces fatal structural flaws:

| Critical Dimension | Traditional Cloud AI Bots | **XMLX-VLM Local AI Trading OS** |
| :--- | :--- | :--- |
| **🛡️ Strategy & Key Privacy** | API keys, secret seeds, order intents, and proprietary alphas are exposed to cloud providers and network MITM. | **100% Air-Gapped & Sovereign**<br>Model weights and execution run entirely on Apple Silicon. Keys stay in a local KMS vault. Zero bytes leave your machine. |
| **💰 7x24 Operational Cost** | High-frequency 24/7 market polling incurs **$300 - $3,000+/month** in cloud token bills. | **$0 Token Cost**<br>Unlimited local inference. Monitor 30+ symbols 24/7 without spending a single cent on API calls. |
| **⏱️ Latency & Rate Limits** | Flash crashes trigger cloud `429 Rate Limit` errors or timeout spikes (500ms - 3s), leading to missed liquidations. | **Sub-Millisecond In-Memory Execution**<br>Dedicated local MLX inference with Continuous Batching and zero external rate limits. |
| **🏛️ Execution Governance** | Single-model direct ordering often enters infinite retry loops or hallucinated overtrading. | **The Model Proposes, Runtime Clamps**<br>Hard pure-functional guardrails, anti-overtrading throttling (re-entry cooldowns), and 4-role Kanban fleet. |

---

## 🚀 5 Core Pillars of XMLX-VLM

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             XMLX-VLM TRADING OS ARCHITECTURE                             │
│                                                                                          │
│  [ Hyperliquid Persistent WS Stream ]                                                    │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. In-Memory Columnar Market Engine                                                │  │
│  │    • Top 30 Perpetuals real-time auto-subscription & RAM state machine             │  │
│  │    • Microsecond indicators: L2 Imbalance, Multi-window CVD, Volume Profile, ATR   │  │
│  │    • Point-in-Time (`as_of`) Time Travel: Strict prevention of lookahead bias      │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │ (High-priority alert triggers / 0-delay memory snapshots)                │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 2. 4-Role Autonomous Kanban Fleet                                                  │  │
│  │    [ Scout (Market Anomalies) ] ──▶ [ Analyst (Multi-TF Strategy) ]                │  │
│  │                                               │                                    │  │
│  │    [ Executor (Smart Routing) ] ◀── [ Risk Officer (Hard Margin & Approval) ]      │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. Enterprise Agent Core & Runtime Guardrails                                      │  │
│  │    • ThinkScrubber: Separates `<think>` reasoning from final structured payloads   │  │
│  │    • ToolCallGuardrails: Loop breaker, duplicate order blocker, no-progress shield │  │
│  │    • Anti-Overtrading Throttling: Re-entry cooldown (30m) & Hourly entry caps      │  │
│  │    • ContextCompressor: Anti-hijack token budgeting (emergency commands priority) │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 4. Apple Silicon MLX Native Acceleration                                           │  │
│  │    • TurboQuant 3.5b/4b hybrid quantization                                        │  │
│  │    • Continuous Batching & SSD-persistent Automatic Prefix Caching (APC)           │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. In-Memory Columnar Market Infrastructure
- **WebSocket-First**: Live connection to `wss://api.hyperliquid.xyz/ws` auto-tracking 24h Top 30 volume perpetuals.
- **Zero-Latency In-Memory Math**: Computes order book depth imbalance, multi-period Cumulative Volume Delta (CVD), Volume Profile (POC/VAH/VAL), and ATR/ADX directly in RAM.
- **Point-in-Time Time Travel (`as_of`)**: Hedge-fund grade columnar architecture with Point-in-Time precision; queries can travel back to any historical timestamp with zero lookahead bias.

### 2. The Model Proposes, Local Runtime Clamps
- **Loop Breaking & Order Shield**: Hard-blocks duplicate failing orders to prevent runaway liquidation spirals.
- **Anti-Overtrading Frequency Discipline**: Injects quantitative psychology anchors (2-4 trades/day, 45-90m holding target) and enforces runtime **Re-Entry Cooldowns (30m)** and **Hourly Entry Limits**.
- **Anti-Hijack Token Compression**: Compresses long conversations with explicit `SUMMARY_PREFIX` guarantees, ensuring user emergency stop/close commands always take top precedence.

### 3. 4-Role Autonomous Kanban Fleet
- **Scout**: Monitors order flow anomalies, funding rate flips, and volatility breakouts.
- **Analyst**: Deep multi-timeframe structural and technical pattern analysis.
- **Risk Officer**: Evaluates account leverage, margin utilization (< 50%), and drawdown limits.
- **Executor**: Executes paper or live signed trades via local OMS.

### 4. Apple Silicon-Native MLX Inference
- Native support for **Apple M1 / M2 / M3 / M4 / M5** chips (Pro/Max/Ultra).
- **Continuous Batching**: Dynamic request scheduling enabling multi-agent concurrency without latency degradation.
- **Tiered APC**: Cache system prompts and market context in RAM and SSD for instant warm-starts.

### 5. Native Quant Web Terminal & Security
- Integrated modern dark-themed web terminal running locally at `http://localhost:5119`.
- **KMS Encrypted Storage**: API credentials encrypted locally via AES-256-GCM.
- **Interactive Approval Gates**: Live execution requires manual one-click confirmation unless explicitly placed on full autopilot.

---

## ⚡ Quick Start (30 Seconds)

### 1. Installation
```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
pip install -e .
```

### 2. Launch Local AI Trading OS & Terminal
```bash
# Launch inference engine + AI Trader Terminal in one command
./service.sh start
```
- 🧠 **OpenAI/Anthropic Dual Inference API**: `http://localhost:5118`
- 🖥️ **AI Trader Web Terminal**: `http://localhost:5119`

### 3. Stop or Check Service Status
```bash
./service.sh status
./service.sh stop
```

---

## 🧪 Comprehensive Verification Suite

XMLX-VLM maintains strict automated test coverage across all subsystems:

```bash
PYTHONPATH=. pytest tests/test_agent_core.py \
                     tests/test_skills_curator.py \
                     tests/test_kanban_board.py \
                     tests/test_ai_trader_agent_core.py \
                     tests/test_columnar_market_store.py \
                     tests/test_throttle_guardrails.py -v
```
> **Test Status**: `35 / 35 Passed (100% Green)` ✅

---

## 🧬 AFRE Research Foundation

XMLX-VLM also serves as the **private AI brain** of the **AFRE (AI Factor Research Engine)** ecosystem — exploring the genealogy, decay, and recombination of quantitative alpha factors across market regimes.

| Research Module | Local Capability |
| :--- | :--- |
| **Factor Genealogy** | Visual document parsing of research PDFs, academic papers, and chart images. |
| **Hypothesis Verification** | Deep CoT reasoning (`<think>` mode) coupled with local backtesting via MCP. |
| **Anti-Overfitting Governance** | Schema-constrained outputs enforcing walk-forward testing and turnover penalties. |

---

## 📄 License

XMLX-VLM is licensed under the [MIT License](LICENSE).
