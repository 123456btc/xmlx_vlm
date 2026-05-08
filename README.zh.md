# XMLX-VLM

<p align="center">
  <strong>Apple Silicon 上的生产级视觉语言推理引擎</strong>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform" src="https://img.shields.io/badge/平台-macOS%20(Apple%20Silicon)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> | <strong>🇨🇳 中文</strong>
</p>

---

## 🚀 为什么选 XMLX-VLM？

**XMLX-VLM** 不是又一个模型加载器。我们面向的是需要在 Apple Silicon 上**真正跑通生产环境**的团队——那些不满足于"笔记本上能跑"，而是需要**可部署、可观测、可运维**的完整视觉语言栈的团队。

我们站在两个优秀开源项目的肩膀上——[**Blaizzy/mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) 提供核心 VLM 加载能力，[**vllm-mlx**](https://github.com/vllm-project/vllm) 社区提供 serving 基础设施——然后补上了从"模型加载器"到"生产系统"之间缺失的每一层：可扩展的 API Server、投机解码、结构化输出、工具调用、Embedding & Rerank 引擎、SSD 持久化的前缀缓存，以及内置的运维套件。

> 如果你在 Mac Studio、Mac Pro 或成规模的 M4 Max 集群上运行 VLM——这是唯一能帮你从"本机能跑"跨越到"生产可用"的完整栈。

---

## 🎯 产品能力

| 能力 | 你得到什么 |
|------|-----------|
| **OpenAI + Claude 双协议 Server** | 一个 Server 同时说两种协议。OpenAI 端点（`/v1/chat/completions`、`/v1/embeddings`、`/v1/rerank`）和 Anthropic 端点（`/v1/messages`），支持流式 SSE、tool-use、reasoning blocks——全部开箱即用。 |
| **SDK 零改动接入** | OpenAI SDK 或 Claude SDK 直接指向 `http://localhost:8080`——无需改代码。Cursor、Claude Code、LangChain、PydanticAI 全部原生支持。 |
| **Gradio Chat UI** | 一条命令启动（`--chat`），自带 polished Web 界面，适合 demo、QA 和内部工具。 |
| **Service Manager** | `service.sh` 一键守护进程化，含健康检查、日志轮转、端口管理、零停机重启。 |
| **API Key 认证** | 通过环境变量轮换密钥。无需代理即可实现企业级访问控制。 |
| **模型仓库 & 转换** | 一行命令把 Hugging Face 模型转成 MLX 格式。原生支持 Qwen、LLaVA、Phi-vision 等数十种模型。 |
| **Batch 推理** | 单次处理多张图片 / 多条 prompt，智能内存调度。 |
| **视觉缓存** | 智能特征缓存，重复的视觉 prompt 不再重复编码图片。 |

---

## ⚡ 技术优势

### 1. Thinking-Aware 约束生成（Logits 层生命周期管理）

这是目前**没有任何其他开源推理引擎**具备的架构级能力。

现代推理模型（Qwen3、DeepSeek-R1、Gemma4 等）会在 `<think>...</think>` 标签内输出思维链，然后才给出最终答案。业界处理这类模型的结构化输出只有两种**错误的**做法：

| 做法 | 问题 |
|------|------|
| **全程约束** | JSON Schema 在 thinking 阶段就生效。模型无法正常推理——它被迫在自我独白中就开始编造结构。 |
| **后处理解析** | 模型无约束生成，然后你祈祷正则能从尾巴里抠出合法 JSON。脆弱、非确定性、工具调用完全不可用。 |

XMLX-VLM 在 **token 生成阶段**用四阶段状态机解决了这个问题：

```
IDLE ──► THINKING ──► TRANSITIONING ──► CONTENT
```

- **THINKING** — 模型完全自由。JSON Schema、正则掩码、任何约束都不生效。我们只计数 `thinking_budget`。
- **TRANSITIONING** — 预算耗尽（或模型自然输出了 end token）时，我们通过 logits masking **强制输出精确的 end-token 序列**（除目标 token 外全部 `-inf`）。保证 thinking 标签干净、确定性地闭合——没有半开标签，没有漂移。
- **CONTENT** — thinking 标签闭合的**瞬间**，控制权交给内层 logits processor（JSON Schema、正则或 tool-parser）。第一个 content token 就已经受约束。

关键实现细节：
- **BoundedSuffixMatcher** 以 O(1) 均摊时间检测 `<think>` / `</think>` token 序列，支持重叠前缀恢复。
- **Snapshot/Rollback** 支持投机解码拒绝和动态 batching 下的状态回滚，不会 desync。
- **Content-phase mask** 防止 `<think>` token 在 transition 后泄露回最终输出。
- **Retirement signal** — 进入 CONTENT 且无内层约束时，processor 通知引擎丢弃自身并重新启用 MTP。

结果：你的模型可以思考 512 个 token，然后吐出一个完美合法的 JSON 对象或 tool call——**零后处理**。

### 2. 多格式推理解析器 + Tool-Call 提升

六个专用流式解析器处理后处理侧：

| 解析器 | 格式 | 特殊处理 |
|--------|------|---------|
| **Qwen3** | `<think>...</think>` | 隐式推理（prompt 中预注入 `<think>`） |
| **DeepSeek-R1** | `<think>...</think>` | 缺失 start-tag 检测 |
| **Gemma4** | `<start_of_thought>...<end_of_thought>` | 多轮 thought 块 |
| **GLM4** | `<|channel>thought...<channel|>` | Channel-based 推理 |
| **GPT-OSS** | 自定义分隔符 | OSS 推理轨迹格式 |
| **Harmony** | 结构化 thinking | 多步推理链 |

每个解析器实现流式状态机（`pre_think → thinking → content`），实时把 delta chunk 拆分为 `reasoning` 和 `content` 流。Server 通过 OpenAI 兼容的 `reasoning_content` 和 Anthropic 兼容的 `thinking` 块同时暴露两者。

**Tool-Call Promotion**：当 `<tool_call>` 块出现在 thinking 阶段时，解析器自动将其提升到 content 流。闭合的 tool call 通过正则提取并追加到最终 content；未闭合的 call 在流结束时 flush 并告警。这意味着推理模型可以"思考要不要调用工具"然后真的调用——而不会把 tool XML 泄露到 reasoning 通道。

### 3. SSD 持久化的自动前缀缓存（APC）

我们把 vLLM 的块级前缀缓存概念移植到了 MLX 运行时，然后加了**磁盘层**：
- KV cache 拆分为 16-token 块，每块通过链式哈希（`H(prev_hash, token_slice, image_hash)`）标识。
- LRU + 引用计数淘汰，热块常驻内存。
- 当设置 `APC_DISK_PATH` 时，完整块写入分片 SSD 文件，**进程重启后仍能恢复**。
- 相同 prompt 从毫秒级 warm-start，而不是秒级——即使 server 重启后也是如此。

**RNN State 支持**：APC 的 `exact` 模式现已支持 BatchKVCache、BatchRotatingKVCache 及所有 `_BaseCache` 子类。对于 Qwen3.5 Gated DeltaNet、Nemotron-H 等混合 SSM/attention 模型，recurrent state 随 KV tensor 一同被 deep-copy 和持久化——不是只有 KV 有效。

### 4. 大规模投机解码

XMLX-VLM 自带**两族投机 draft 模型**：
- **DFlash** — 超轻量 draft 模型，提前预测 2–3 个 token，几乎零开销。
- **MTP**（Multi-Token Prediction）— 高熵 prompt 下的并行 draft 路径。

配合自适应接受阈值，投机路径可显著削减长文本生成的首 token 时间。

### 5. Tool-Call 自动修复 + Jump-Forward 解码

量化模型在生产环境里的经典问题：多轮 tool call 后开始输出畸形 XML 标签或截断 JSON。XMLX-VLM 增加两层互补防御：

- **Auto-Recovery** — 主解析器失败后，启发式修复层会闭合未闭合标签、平衡截断 JSON 的大括号、从混乱文本中提取裸 JSON 对象。闭合的 tool call 提升为结构化输出；未闭合的 flush 并告警。
- **Jump-Forward Logits Bias**（`--enable-tool-logits-bias`）— 对 tool-related token ID（`<tool_call>`、`{`、`"name"` 等）施加加性偏置，更快推入结构化格式，不改变 temperature 或 top-p。

### 6. KV-Cache 量化

统一内存架构上，内存是瓶颈。我们支持：
- **Uniform quantization**（4-bit、3.5-bit、8-bit）
- **TurboQuant** — 自适应策略，在关键位置保留 attention 精度，在无关位置压缩。

### 7. MoE Top-K 覆盖

MoE 模型是新标准，但默认路由浪费 cycle。XMLX-VLM 暴露**动态 top-k 覆盖**，让你在延迟敏感场景用极小的精度损失换取巨大的延迟收益。

### 8. 工具调用 & MCP（Model Context Protocol）

- 从模型 processor 自动推断 tool-parser
- 可插拔 tool 模块
- 内置 **MCP Manager**，通过 stdio 或 SSE 连接外部数据源、IDE 和 agent 框架

### 9. Embedding & Rerank 引擎

**单一进程**同时 serving **视觉语言对话**、**文本 Embedding** 和 **Rerank**。无需运行独立的 embedding 微服务——减少网络跳转和上下文切换开销。

### 10. Apple Silicon 原生优化

- Flash Attention 通过 `mx.fast.scaled_dot_product_attention`
- Metal kernel fusion 加速视觉编码器
- 硬件感知内存预算（M1 → M4 Ultra 配置文件内置）
- CPU 预处理与 GPU 推理之间的统一内存零拷贝

---

## 🏗 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端层                               │
│  (OpenAI SDK、LangChain、curl、Gradio UI、Agent 框架)        │
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
│                     推理核心                                   │
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
│         (Apple Silicon 统一内存 & GPU 核心)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚦 快速开始

### 安装

```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 启动服务

```bash
# 基础服务
./service.sh start

# 服务 + Chat UI
./service.sh start --chat

# 投机解码 + KV 量化
./service.sh start --chat \
  --draft-model mlx-community/Qwen3.6-35B-A3B-DFlash \
  --draft-kind dflash \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant

# 启用 tool-call 加速
./service.sh start --chat --enable-tool-logits-bias
```

### 调用 API

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/qwen3.6-35B-A3B-4bit",
    "messages": [
      {"role": "user", "content": "描述这张图片"}
    ],
    "stream": true
  }'
```

---

## 🛠 运维与可观测性

```bash
# 查看健康状态、已加载模型、PID、端口
./service.sh status

# 实时 tail 日志
./service.sh logs server
./service.sh logs chat

# 零停机重启
./service.sh restart --chat
```

- **PID 追踪** + orphan 进程兜底
- **端口冲突**自动解决
- **`/health` 健康端点**，可直接对接负载均衡
- **结构化日志**，rotation-friendly

---

## 🧩 支持模型

- **Qwen-VL / Qwen2-VL / Qwen3.6-VL**（推荐）
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
  <strong>Built for teams who ship.</strong><br>
  如果 XMLX-VLM 加速了你的视觉流水线，请给我们点一颗 ⭐
</p>
