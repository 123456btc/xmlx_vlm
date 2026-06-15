# XMLX-VLM

<p align="center">
  <strong>隐私优先的本地视觉语言 AI</strong>
</p>

<p align="center">
  <em>Apple Silicon 原生推理引擎。数据零外泄。零云 API 暴露。</em>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform" src="https://img.shields.io/badge/平台-macOS%20(Apple%20Silicon)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> | <strong>🇨🇳 中文</strong> | <a href="README.ja.md">🇯🇵 日本語</a> | <a href="README.ko.md">🇰🇷 한국어</a>
</p>

---

## 🚀 为什么选 XMLX-VLM？

**对于处理敏感数据的专业人士，隐私不是功能——它是底线。**

法律文件、医疗记录、政务档案、 proprietary 研究成果、交易策略——一旦它们经过云 API，你就失去了控制权。它们变成了别人的训练数据。它们被记录。它们可能被传唤。

**XMLX-VLM** 是一个**本地优先、生产级的视觉语言推理引擎**，完全运行在 Apple Silicon 上。它读取文档、解析图片、推理复杂问题、输出结构化结果——**全程零网络调用**。

没有云订阅。没有数据保留政策。没有第三方服务条款。只有你的 Mac、你的数据、你的模型。

> **数据主权是架构本身。其他一切都是建立在这个基础之上。**

### 🧬 AFRE 生态系统

XMLX-VLM 是 **AFRE（AI Factor Research Engine）** 生态系统的**私有化 AI 大脑**——一个基于 DDD、六边形架构和 Clean Architecture 的领域优先、Agent 增强的量化研究平台。

AFRE 研究**市场因子的谱系**：它们为什么被发明、如何传播、为何失效，以及哪些失效假设可以生成现代假设。XMLX-VLM 为 AFRE 的 Agent Runtime 提供本地推理能力：

| AFRE 能力 | XMLX-VLM 在本地实现什么 |
|-----------|------------------------|
| **因子谱系智能** | 本地解析研报 PDF 和图表；从视觉文档中提取结构化因子历史 |
| **发明者思维模拟** | 深度推理（`<think>` 模式）模拟因子创造者的约束、激励和知识栈 |
| **假设驱动研究** | JSON-Schema 约束输出确保每个生成的因子变体都携带可测试假设和失效假设追溯 |
| **可复现实验** | Tool Calling + MCP 连接本地回测器和信号生成器；实验在你的硬件上运行，天生可审计 |
| **反过拟合治理** | 结构化输出强制 walk-forward 参数、regime split、换手率惩罚等机器可读模式 |
| **知识进化** | Embedding + Rerank 将验证发现索引为受治理的、可查询的本地知识库 |
| **多 Agent 并行研究** | Continuous Batching + 投机解码让多个 AI Worker 独立推理而不崩塌延迟 |

**AFRE 是方法论。XMLX-VLM 是让这一切成为可能的私有化推理层。**

虽然 AFRE 是 XMLX-VLM 在量化金融领域的旗舰实现，同样的本地隐私架构同样服务于法律、医疗、政务和企业研发领域。

---

## 🎯 这是为谁打造的？

| 领域 | 敏感数据 | XMLX-VLM 在本地做什么 |
|------|---------|---------------------|
| **量化金融** |  proprietary 因子、内部研报、alpha 信号 | 本地解析 PDF 研报和图表；推理因子假设；输出结构化因子定义；通过 MCP 调用本地回测工具；AI Trader 本地分析 Hyperliquid 行情与交易 |
| **法律** | 案件卷宗、合同、证据材料、客户沟通记录 | 分析扫描文档和证据图片；提取结构化条款；推理法律论证；生成批注摘要 |
| **政务** | 涉密简报、政策草案、公民档案、情报图像 | 处理敏感图像和扫描文档；结构化情报报告输出；完整审计轨迹保留在本地硬件 |
| **医疗** | 患者病历、医学影像、临床笔记、检验结果 | 解析医疗文档图片；推理鉴别诊断；结构化临床摘要输出；架构层面 HIPAA 合规 |
| **企业研发** | 商业机密、专利草稿、实验数据、内部备忘录 | 理解技术图纸的视觉语言；推理研究假设；结构化实验设计输出 |

---

## 🎯 核心能力

| 能力 | 你得到什么 |
|------|-----------|
| **本地文档智能** | 直接把 PDF、扫描件、截图、图文混排报告喂给模型。没有 OCR SaaS。没有云端视觉 API。你的文档永不离开 localhost。 |
| **推理后的结构化输出** | 开启 `thinking` 模式进行深度推理，然后对最终输出强制执行 JSON-Schema 约束。审计级报告、因子定义、临床摘要的完美选择。 |
| **双协议 API** | 一个 Server 同时说 OpenAI（`/v1/chat/completions`）和 Anthropic（`/v1/messages`）两种协议。作为 Cursor、Claude Code、LangChain、PydanticAI 的后端——全部流量留在 `localhost:5118`。 |
| **本地工具调用 & MCP** | 通过 MCP 连接本地数据库、回测器、电子病历系统、案件管理工具、文档流水线。模型调用你的工具；你的数据永不离开机器。 |
| **Embedding & Rerank 用于私有知识** | 索引内部文档、研究笔记、案件卷宗、患者病史。在你的专有知识库上做语义搜索——零云端暴露。 |
| **AI Trader（本地量化助手）** | 与本地 AI 交易助手对话。通过 Hyperliquid WebSocket 常驻行情服务自动监控 24h 成交额前 30 名永续合约，实时计算技术指标并推送阈值警报，本地渲染图表并模拟交易。 |
| **SSD 持久化前缀缓存** | 重复分析同一文档或系统 prompt 时毫秒级 warm-start，即使 server 重启后也是如此。缓存活在你的 SSD 上，不是别人的服务器。 |
| **AI Trader Chat UI** | 一条命令启动（`--chat`，端口 `5119`），集成安全 KMS 凭证保险箱，可实时监控 Hyperliquid 资产、持仓与成交历史。 |
| **Service Manager** | `service.sh` 一键守护进程化，含健康检查、日志轮转、端口管理、零停机重启。 |
| **API Key 认证** | 通过环境变量轮换密钥。无需代理即可实现企业级访问控制。 |

---

## ⚡ 技术优势

### 1. Thinking-Aware 约束生成

现代推理模型会在 `<think>...</think>` 标签内输出思维链。标准结构化输出引擎要么在 thinking 阶段崩掉，要么腐蚀 JSON。XMLX-VLM 在 **token 生成阶段**用四阶段状态机管理：

```
IDLE ──► THINKING ──► TRANSITIONING ──► CONTENT
```

- **THINKING** — 模型完全自由推理。JSON 约束不生效。可以探索假设、边界情况、矛盾点。
- **TRANSITIONING** — 预算耗尽时，我们通过 logits masking **强制输出精确的 end-token 序列**（除目标 token 外全部 `-inf`）。干净、确定性地闭合。
- **CONTENT** — thinking 标签闭合的**瞬间**，控制权交给内层 JSON-Schema processor。第一个 content token 就已经受约束。

结果：你的模型可以思考 512 个 token 来分析法律论证或医疗鉴别诊断，然后吐出一个完美合法的结构化 JSON——**零后处理**。

### 2. SSD 持久化的自动前缀缓存（APC）

当你反复迭代同一文档或系统 prompt 时，XMLX-VLM 跨请求复用 KV cache。对于混合 SSM/attention 模型（Qwen3.5 DeltaNet、Nemotron-H），**recurrent state 也一并被 snapshot 并持久化到 SSD**：

- 块级 KV cache，链式哈希标识
- LRU + 引用计数淘汰
- `APC_DISK_PATH` 把完整块写入分片 SSD 文件——**进程重启后仍能恢复**
- 相同 prompt 毫秒级 warm-start，即使 server 重启后

### 3. 多格式推理解析器 + Tool-Call 提升

六个专用流式解析器处理 Qwen3、DeepSeek-R1、Gemma4、GLM4、GPT-OSS、Harmony 的 reasoning 提取。当 `<tool_call>` 块出现在 thinking 阶段时，它会被**自动提升到 content 流**——模型可以"思考要不要调用工具"然后真的调用。

### 4. Tool-Call 自动修复 + Jump-Forward 解码

量化模型在多轮 tool call 后开始输出畸形标签。XMLX-VLM 两层防御：

- **Auto-Recovery** — 修复未闭合 XML 标签、平衡截断 JSON 大括号、从混乱文本提取裸 JSON。
- **Jump-Forward Logits Bias**（`--enable-tool-logits-bias`）— 对 tool-related token ID 施加加性偏置，更快推入结构化格式。

### 5. 大规模投机解码

- **DFlash** — 超轻量 draft 模型提前预测 2–3 个 token
- **MTP**（Multi-Token Prediction）— 高熵 prompt 下的并行 draft 路径

削减长文档分析和推理任务的延迟。

### 6. KV-Cache 量化

- **Uniform**（4-bit、3.5-bit、8-bit）
- **TurboQuant** — 自适应策略，在关键位置保留 attention 精度

128 GB Mac Studio 上运行 70B 级 vision 模型，长文档上下文仍有充裕空间。

### 7. MoE Top-K 覆盖

动态 top-k 覆盖，在交互式分析会话中用极小的精度损失换取巨大的延迟收益。

### 8. Apple Silicon 原生优化

- Flash Attention 通过 `mx.fast.scaled_dot_product_attention`
- Metal kernel fusion 加速视觉编码器
- 硬件感知内存预算（M1 → M5 Max 配置文件内置）
- CPU 预处理与 GPU 推理之间的统一内存零拷贝

---

## 🏗 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│              AFRE（AI Factor Research Engine）                │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ 因子谱系     │ │ 发明者       │ │ 假设驱动研究        │  │
│  │ 智能         │ │ 思维模拟     │ │                     │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ 反过拟合     │ │ 知识         │ │ 多 Agent            │  │
│  │ 治理         │ │ 进化         │ │ 并行研究            │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     私有 AI Agent & 客户端                    │
│  (Cursor、Claude Code、LangChain、PydanticAI、AFRE agents、  │
│   AI Trader — 本地量化助手)                                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   XMLX-VLM Server（本地）                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │   Chat API  │ │ Embeddings  │ │  Rerank / Classify  │   │
│  │ (OpenAI +   │ │  (私有知识  │ │  (文档 / 案件       │   │
│  │  Anthropic) │ │   记忆)     │ │   检索)             │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Tool Parse │ │    MCP      │ │ Structured Output   │   │
│  │ (本地数据库) │ │ (内部系统)  │ │ (审计级 JSON)       │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     推理核心                                   │
│  ┌──────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │   Generate   │ │   Batch     │ │  Speculative Draft  │  │
│  │  (推理)      │ │  (文档)     │ │  (延迟削减)         │  │
│  └──────────────┘ └─────────────┘ └─────────────────────┘  │
│  ┌──────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │ KV Quantize  │ │  MoE Top-K  │ │  Vision Cache       │  │
│  └──────────────┘ └─────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    MLX / Metal Runtime                       │
│         (Apple Silicon 统一内存 & GPU 核心)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚦 快速开始

### 一键安装（macOS Apple Silicon）

全新 Mac 或没有开发工具的用户：

```bash
curl -fsSL https://raw.githubusercontent.com/123456btc/xmlx_vlm/master/install.sh | bash
```

该脚本全自动完成：
- ✅ 检测 Apple Silicon（M1/M2/M3/M4/M5）
- ✅ 安装 Xcode Command Line Tools（如缺失）
- ✅ 安装 Homebrew（如缺失）
- ✅ 安装 Python 3.12（如低于 3.10）
- ✅ 安装 `uv`（高速 Python 包管理器）
- ✅ 克隆仓库并创建虚拟环境
- ✅ 安装 MLX、XMLX-VLM 及全部依赖
- ✅ 设置默认 API key（`x123456`）和环境变量
- ✅ 可选预下载默认模型（~20GB）
- ✅ 启动服务

**预计耗时：** 全新 Mac 上 10-20 分钟（主要是模型下载）。

### 手动安装

如果你偏好手动：

```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 启动服务

```bash
# 默认启动 — 仅 server（无 Chat UI）
./service.sh start

# 同时启动 chat UI
./service.sh start --chat

# 覆盖默认 API key + KV 量化，用于生产负载
XMLX_VLM_API_KEY=mykey ./service.sh start --kv-bits 3.5 --kv-quant-scheme turboquant

# 启用 tool-call 加速，用于 MCP 密集型工作流
./service.sh start --enable-tool-logits-bias

# 完全禁用投机解码（回退到标准生成）
XMLX_VLM_DRAFT_MODEL="" XMLX_VLM_DRAFT_KIND="" ./service.sh start
```

### 调用 API（仅限本地）

```bash
curl http://localhost:5118/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
    "messages": [
      {"role": "user", "content": "分析附件文档并提取结构化发现"}
    ],
    "stream": true
  }'
```

### 启动 AI Trader（本地量化助手）

```bash
# 先启动服务
./service.sh start

# 与本地交易助手对话
xmlx_vlm.ai-trader

# 或执行单条指令
xmlx_vlm.ai-trader --prompt "分析 BTC 走势"
```

AI Trader 通过 Hyperliquid WebSocket 常驻行情服务获取数据：自动订阅 24h 成交额前 30 名永续合约，维护内存状态机，实时计算技术指标，并在价格突破、OI 异动、大单集群、Funding 反转、盘口失衡、波动率扩张时发出警报。为了使本地决策最优化，AI Trader 新增了**单请求多空对抗辩论机制**以防范单边倾向，并在平仓时自动启动**异步复盘反思任务**将经验写入本地 SQLite，自动闭环学习并指导后续决策。工具优先读取本地快照，缺失时回退到 REST 轮询。

---

## 🤖 Agent 客户端接入

XMLX-VLM 专为**编程 Agent 和 AI 助手的本地后端**而设计。由于 Agent 客户端每轮都会重发完整对话历史，强烈建议启用 APC 磁盘持久化（`APC_DISK_PATH`）——它能在首次昂贵的预填充后消除重复预填充开销。

### Claude Code（Anthropic 兼容）

创建 `~/.local/bin/claude-xmlx`：

```bash
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${XMLX_ANTHROPIC_BASE_URL:-http://127.0.0.1:5118}"
export ANTHROPIC_AUTH_TOKEN="${XMLX_API_KEY:-x123456}"
export ANTHROPIC_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"

export ANTHROPIC_CUSTOM_MODEL_OPTION="mlx-community/diffusiongemma-26B-A4B-it-4bit"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="XMLX-VLM Local Qwen3.6"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="本地 MLX 推理 via xmlx_vlm"

export ANTHROPIC_DEFAULT_SONNET_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"
export ANTHROPIC_DEFAULT_OPUS_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"
export CLAUDE_CODE_SUBAGENT_MODEL="mlx-community/diffusiongemma-26B-A4B-it-4bit"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

启动带磁盘 KV 缓存的服务：

```bash
APC_ENABLED=1 APC_DISK_PATH=/tmp/xmlx-apc ./service.sh start
```

### Cline / Continue.dev（OpenAI 兼容）

在 VS Code 设置或 `~/.continue/config.json` 中：

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

### Aider（OpenAI 兼容）

```bash
export OPENAI_API_BASE=http://localhost:5118/v1
export OPENAI_API_KEY=x123456
aider --model openai/mlx-community/diffusiongemma-26B-A4B-it-4bit
```

### Cursor（OpenAI 兼容）

在 Cursor 设置 → Models → Add Model：
- **Base URL**: `http://localhost:5118/v1`
- **API Key**: `x123456`
- **Model**: `mlx-community/diffusiongemma-26B-A4B-it-4bit`

### Pi（pi.dev）

Pi 是一款本地优先的编程 Agent，与 XMLX-VLM 配合良好。在 `~/.pi/agent/models.json` 中添加 provider：

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
          "id": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
          "name": "DiffusionGemma 26B A4B 4bit (XMLX-VLM local)",
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
          "contextWindow": 128000,
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

然后在 `~/.pi/agent/settings.json` 设为默认：

```json
{
  "defaultProvider": "xmlx-local",
  "defaultModel": "mlx-community/diffusiongemma-26B-A4B-it-4bit"
}
```

### 推荐的 Agent 服务端启动参数

```bash
# 完整 Agent 栈：磁盘 APC + 工具加速 + 长上下文
APC_ENABLED=1 \
APC_DISK_PATH=/tmp/xmlx-apc \
XMLX_VLM_ENABLE_TOOL_LOGITS_BIAS=1 \
./service.sh start --ctx 100000
```

> **提示**：Agent 客户端的初始系统提示通常在 10k–30k token 之间。启用 `APC_DISK_PATH` 后，该前缀在首次预填充时写入 SSD，后续会话（即使服务器重启后）也能瞬间恢复。

---

## 🛠 运维与可观测性

```bash
# 查看健康状态、已加载模型、PID、端口
./service.sh status

# 实时 tail 日志
./service.sh logs server
./service.sh logs chat

# 零停机重启
./service.sh restart
```

- **PID 追踪** + orphan 进程兜底
- **端口冲突**自动解决
- **`/health` 健康端点**
- **结构化日志**，rotation-friendly

---

## 🧩 支持模型

- **Qwen-VL / Qwen2-VL / Qwen3.6-VL**（推荐，CJK 文档友好）
- **LLaVA 1.5 / 1.6 / NeXT**
- **Phi-3 / Phi-4 Vision**
- **InternVL2**
- **MiniCPM-V**
- **DeepSeek-VL**
- …以及任何有 MLX community 端口的 Hugging Face 模型

---

## 🏛 致谢与传承

XMLX-VLM 是一个**hard-fork**，有意识地建立在多个杰出开源项目之上：

| 项目 | 我们借鉴了什么 | 我们增加了什么 |
|------|-------------|---------------|
| [**Blaizzy/mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) | 核心 VLM 模型加载、权重转换、MLX 生成原语 | 生产级 server、投机解码、结构化输出、工具调用、MCP、embedding/rerank 引擎 |
| [**vllm-mlx**](https://github.com/vllm-project/vllm)（社区模式）| Metrics 设计、模型注册表模式、硬件检测概念 | SSD 持久化 APC 缓存、Apple Silicon 专用内存预算、统一 CLI |
| [**Rapid-MLX**](https://github.com/raullenchai/Rapid-MLX) | Tool-call 自动修复、jump-forward logits bias、DeltaNet state snapshot | 适配了自动修复和 jump-forward 解码；启发了混合缓存架构路线图 |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp) | 混合量化策略（Q4_K_M 风格） | 集成到 MLX 转换流水线 |
| [**Hugging Face Transformers**](https://github.com/huggingface/transformers) | Tokenizer 工具、sampling 逻辑、AutoModel 加载 | MLX 原生权重转换、batch 流式、thinking-aware processor |

我们深深感谢这些项目的作者和社区。XMLX-VLM 因他们打下的基础而存在。

---

## 🤝 社区与路线图

- [x] 双协议 REST API（OpenAI + Anthropic/Claude）
- [x] 投机解码（DFlash + MTP）
- [x] KV-cache 量化
- [x] 工具调用 & MCP
- [x] Embedding & Rerank 引擎
- [x] SSD 持久化的自动前缀缓存
- [x] Thinking-aware 结构化生成
- [x] Tool-call 自动修复 + Jump-Forward 解码
- [x] LoRA 训练与 adapter 加载
- [~] LoRA 热切换 serving（训练与加载已可用；运行时切换 adapter 需 server API）
- [~] Tensor / Pipeline Parallelism（utils 层已支持 `mx.distributed`；server 集成待完成）
- [x] 内置 benchmark（TTFT / TPOT / TPS / 内存）
- [ ] 跨引擎 benchmark 套件（欢迎贡献！）

**License:** MIT  
**起源:** Hard-fork 自 [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) —— 为生产负载重建。

---

<p align="center">
  <strong>你的数据。你的模型。你的隐私。</strong><br>
  如果 XMLX-VLM 保护了你的敏感流水线，请给我们点一颗 ⭐
</p>
