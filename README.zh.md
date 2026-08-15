# XMLX-VLM: 本地私有 AI 量化交易操作系统 & 终端

<p align="center">
  <strong>全球首个 Apple Silicon 原生、完全私有化的自主 AI 量化交易操作系统</strong>
</p>

<p align="center">
  <em>数据零外泄 · 零 Token 账单 · 微秒级内存行情中枢 · 四角色智能体战队与机构级风控护栏 —— 100% 运行在你的 Mac 本地。</em>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="https://github.com/123456btc/xmlx_vlm/releases/tag/v1.0.0"><img alt="Release: v1.0.0" src="https://img.shields.io/badge/Release-v1.0.0%20Latest-brightgreen"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform: macOS" src="https://img.shields.io/badge/平台-macOS%20(Apple%20Silicon%20M1--M5)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> | <strong>🇨🇳 中文</strong> | <a href="README.ja.md">🇯🇵 日本語</a> | <a href="README.ko.md">🇰🇷 한국어</a>
</p>

---

## ⚡ 为什么选择本地私有 AI 交易操作系统？

市面上绝大多数传统云端 AI 炒股/交易机器人（Cloud-based AI AutoTraders）都高度依赖**云端大模型 API**（OpenAI、Anthropic、DeepSeek）。在真实量化实盘中，依赖云端存在三大致命缺陷：

| 核心维度 | 传统云端 AI 交易 Bot (Cloud AI Trading Bots) | **XMLX-VLM 本地 AI 量化交易操作系统** |
| :--- | :--- | :--- |
| **🛡️ 策略与私钥安全** | API Key、私钥签名、开仓意图和内部 Alpha 策略必须打包发送到云端，面临第三方数据窃听与泄露风险。 | **100% 气隙隔离与数据主权**<br>模型权重与执行完全运行在本地 Apple Silicon。密钥存储在本地 KMS 保险箱，**0 字节离开你的电脑**。 |
| **💰 7x24 运行成本** | 24 小时高频轮询盯盘，每月云端 Token 费用高达 **$300 ~ $3,000+ 美元**。 | **$0 Token 成本**<br>无限量本地硬件推理。7x24 小时监控 30+ 币种，无需支付任何 API 调用费用。 |
| **⏱️ 延迟与限流风险** | 极端行情剧烈波动时，极易遭遇云端 `429 Too Many Requests` 限流或数秒网络超时，导致错失止损爆仓。 | **微秒级内存直达与零限流**<br>专为 MLX 优化的连续批处理（Continuous Batching），零外部网络依赖与限流阻塞。 |
| **🏛️ 运行时执行治理** | 单模型简单循环容易陷入死循环重复下单，或在震荡市中冲动过度交易（Overtrading）。 | **大模型提方案，本地 Runtime 强风控**<br>纯函数护栏、再入场冷静期（30m）、小时开仓上限与 4 角色多智能体协同流水线。 |

---

## 🚀 XMLX-VLM 的 5 大核心支柱

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            XMLX-VLM 交易操作系统核心架构                                 │
│                                                                                          │
│  [ Hyperliquid 常驻 WebSocket 行情流 ]                                                   │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 内存列式行情中枢 (In-Memory Columnar Engine)                                    │  │
│  │    • 实时订阅 Top 30 成交额永续合约，本地维护状态机                                │  │
│  │    • 微秒级指标计算：L2 深度失衡、多周期 CVD、筹码分布 Volume Profile、ATR/ADX     │  │
│  │    • Point-in-Time (`as_of`) 时间旅行：严格杜绝大模型回测与复盘中的“未来函数”      │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │ (高优先级警报触发 / 零网络延迟内存快照)                                 │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 2. 4 角色多智能体自主看板战队 (Kanban Fleet)                                       │  │
│  │    [ Scout (行情侦察) ] ──▶ [ Analyst (多周期点位分析) ]                            │  │
│  │                                       │                                            │  │
│  │    [ Executor (撮合执行) ] ◀── [ Risk Officer (风控与审批门禁) ]                   │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. 企业级 Agent 核心与运行护栏 (Guardrails)                                        │  │
│  │    • ThinkScrubber：思考流 `<think>` 提取与结构化决策隔离                          │  │
│  │    • ToolCallGuardrails：纯函数死循环熔断、重复下单拦截、无进展状态防护            │  │
│  │    • 防过度交易节流阀：再入场 30 分钟冷静期 (Re-entry Cooldown) 与单小时开仓上限   │  │
│  │    • ContextCompressor：带反劫持前缀的 Token 预算压缩（急停平仓指令优先）          │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 4. Apple Silicon MLX 原生推理加速                                                  │  │
│  │    • TurboQuant 3.5b/4b 混合量化与连续批处理                                       │  │
│  │    • 内存 + SSD 分层持久化前缀缓存 (APC)，跨会话秒级启动                           │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. 机构级内存列式行情基础设施 (Hedge-Fund Grade Columnar Engine)
- **WebSocket 常驻连接**：直连 `wss://api.hyperliquid.xyz/ws`，自动监听 24h 成交额前 30 名主流合约。
- **微秒级本地指标聚合**：在 RAM 中动态计算订单簿失衡度（Imbalance Ratio）、累积成交量差（CVD）、Volume Profile 价值区（POC/VAH/VAL）与 ATR/ADX。
- **Point-in-Time (`as_of`) 时间旅行**：支持查询历史任意时刻的截面快照，完全杜绝 Lookahead Bias（未来函数）。

### 2. 大模型提方案，本地 Runtime 强风控
- **死循环与重复下单熔断**：相同参数连续失败自动阻断，防止极端行情下反复下单导致爆仓。
- **防过度交易纪律（Anti-Overtrading）**：提示词注入量化心理学约束（2-4 笔/日，期望持仓 45-90 分钟），Runtime 强制执行 **平仓后 30 分钟冷静期** 与 **单小时最大开仓限制**。
- **防指令劫持上下文压缩**：带有 `SUMMARY_PREFIX` 权威约束，确保用户下达的“一键清仓/急停”指令永远拥有最高执行权。

### 3. 四角色多智能体自主战队
- **Scout（侦察员）**：监控突破、大单集群、波动率异动与资金费率翻转。
- **Analyst（分析师）**：进行多周期趋势分析与出入场点位规划。
- **Risk Officer（风控官）**：核算保证金使用率（< 50%）与杠杆限制。
- **Executor（执行员）**：调用本地 OMS 执行模拟或带签名实盘订单。

### 4. Apple Silicon 原生 MLX 加速
- 原生支持 **Apple M1 / M2 / M3 / M4 / M5** 全系列芯片。
- **Continuous Batching**：多智能体并发推理互不阻塞。
- **分层 APC 前缀缓存**：系统提示词持久化至 SSD，毫秒级温启动。

### 5. 原生现代化量化 Web 终端
- 一键启动本地暗黑风格交易终端：`http://localhost:5119`。
- **KMS 本地加密**：API 密钥经 AES-256-GCM 保护，永不上云。
- **交互式审批门禁**：实盘下单支持手动一键核准，亦可切换全自动巡航。

---

## ⚡ 极速起步（30 秒）

### 1. 克隆与安装
```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
pip install -e .
```

### 2. 一键启动本地 AI 交易操作系统
```bash
# 启动推理引擎与量化交易终端
./service.sh start
```
- 🧠 **OpenAI / Anthropic 双协议推理 API**: `http://localhost:5118`
- 🖥️ **AI Trader 量化交易 Web 终端**: `http://localhost:5119`

### 3. 查看状态与停止
```bash
./service.sh status
./service.sh stop
```

---

## 🧪 完整自动化测试套件

XMLX-VLM 拥有严格的自动化测试保障体系：

```bash
PYTHONPATH=. pytest tests/test_agent_core.py \
                     tests/test_skills_curator.py \
                     tests/test_kanban_board.py \
                     tests/test_ai_trader_agent_core.py \
                     tests/test_columnar_market_store.py \
                     tests/test_throttle_guardrails.py -v
```
> **测试状态**：`35 / 35 全部通过（100% 绿灯）` ✅

---

## 🧬 AFRE 量化因子研究基座

XMLX-VLM 同时作为 **AFRE (AI Factor Research Engine)** 因子谱系研究平台的私有化 AI 大脑，赋能因子溯源、假设检验与防过拟合治理。

---

## 📄 开源许可证

本项目采用 [MIT 许可证](LICENSE)。
