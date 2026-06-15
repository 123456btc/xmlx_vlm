# AI Trader — 本地 AI 量化助手

基于 XMLX-VLM 的本地开源 AI 量化交易助手。启动后通过自然语言即可完成实时行情分析、K 线图绘制、模拟交易等操作。所有行情数据通过 **Hyperliquid WebSocket 常驻连接** 推送到本地内存状态机，AI 在调用工具时直接读取毫秒级快照。

> **⚠️ 资金安全警告**：`trading` 工具默认使用 **纸盘（paper）** 模式，不会动用真实资金。实盘交易需要显式配置环境变量与 API 凭证，并自行承担风险。

## 快速开始

### 1. 安装依赖

```bash
cd /Users/hongjianjia/xmlx_vlm
.venv/bin/python -m pip install -r requirements.txt
```

> `websockets` 已写入 `requirements.txt`，用于 Hyperliquid 实时行情流。
> 若需 Hyperliquid 本地私钥签名，请额外安装 `eth-account`：
> ```bash
> .venv/bin/python -m pip install eth-account
> ```

### 2. 启动服务（复用 service.sh 默认模型）

```bash
./service.sh start
```

这会启动 `xmlx_vlm.server` 并加载默认模型。AI Trader 默认连接 `http://localhost:5118`。

### 3. 启动 AI Trader

```bash
# 连接 service.sh 启动的服务（推荐，纸盘模式）
xmlx_vlm.ai-trader

# 本地加载模型（不需要 server，需自行指定模型路径）
xmlx_vlm.ai-trader --local --model <your-model-path>

# 禁用常驻行情服务，完全回退到 REST 轮询
XMLX_VLM_AI_TRADER_WS=0 xmlx_vlm.ai-trader
```

### 4. 示例对话

```text
你: BTC 现在多少钱？
你: 画一张 BTC 1小时 K 线图
你: 分析一下 BTC 走势
你: 模拟买入 0.01 BTC
你: 我现在持仓多少？
你: 急停
```

## CLI 参数

```bash
xmlx_vlm.ai-trader [--server SERVER] [--local] [--model MODEL]
                   [--temperature TEMP] [--max-tokens N]
                   [--prompt PROMPT]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--server` | 推理服务地址 | `http://localhost:5118` |
| `--local` | 不连接 server，本地加载模型 | `False` |
| `--model` | 本地模型路径 / HuggingFace repo id | 无（连接 server 时由 server 决定；`--local` 时必填） |
| `--temperature` | 采样温度 | `0.3` |
| `--max-tokens` | 最大生成长度 | `2048` |
| `--prompt` | 非交互模式执行一句指令 | 无 |

## 核心能力

### 行情基建（机构级实时流）

- **常驻 WebSocket 连接**：`xmlx_vlm.ai_trader.market_service.MarketDataService` 长期连接 Hyperliquid WS，自动重连、订阅恢复、断线回补。
- **自动监控 Top 30**：启动时按 24h 名义成交额自动订阅前 30 名永续合约，每小时刷新一次监控列表。
- **内存状态机**：每个币种维护 tick、L2 订单簿、逐笔成交、资金费率、持仓量快照；实时聚合 1m/5m/15m/1h/4h K 线。
- **实时技术指标**：EMA、RSI、ATR(14)、ADX/DMI、Volume Profile（POC/VAH/VAL）、VWAP、多窗口 CVD。
- **事件总线 + 警报引擎**：原始行情事件经阈值判断后产出高阶警报，避免 AI 被每笔 tick 淹没。
  - `price_breakout`：价格突破近期高低点且成交量放大
  - `oi_spike`：OI 1h/24h 变化率超阈值
  - `large_order_cluster`：短时间内大单集群同向吃单
  - `funding_flip`：资金费率正负反转
  - `book_imbalance_spike`：盘口深度失衡超阈值
  - `volatility_expansion`：5m 波幅相对 ATR 异常扩张
- **REST 回退**：当常驻服务未启用或数据缺失时，`market_data` 工具自动回退到 Hyperliquid REST API。

### 工具调用

| 工具 | 说明 |
|------|------|
| `market_data` | 查 ticker、OHLCV、L2 订单簿、市场摘要、多周期分析、资金费率、持仓量 |
| `render_chart` | 使用 Pillow 本地生成带 EMA/成交量/RSI 的 K 线图 |
| `trading` | 查询持仓、模拟下单、平仓、紧急停止（默认纸盘） |

### 风控与交易

- 默认纸盘模式，不触及真实资金。
- 实盘模式需同时满足环境变量与凭证配置（详见 `oms/` 文档）。
- 交易工具内部通过 OMS 进行风控检查、审计日志、熔断与急停。

## 配置

### 环境变量

复制示例文件并根据需要填写：

```bash
cp xmlx_vlm/ai_trader/.env.example .env
source .env
```

关键环境变量：

| 变量 | 说明 | 是否必填 |
|------|------|----------|
| `XMLX_VLM_AI_TRADER_WS` | 设置为 `0` 禁用常驻 WebSocket 服务 | 否，默认启用 |
| `AI_TRADER_LIVE` | 设置为 `1` 启用实盘 | 实盘必填 |
| `HL_API_WALLET_ADDRESS` | Hyperliquid 钱包地址 | 实盘必填 |
| `HL_API_PRIVATE_KEY` | Hyperliquid 私钥（本地签名） | 实盘二选一 |
| `HL_SIGNER_ENDPOINT` | 外部签名器 URL（生产推荐） | 实盘二选一 |
| `HL_TESTNET` | 设置为 `1` 使用测试网 | 否 |

### 警报阈值

在代码中创建 `MarketDataService` 时传入 `AlertConfig` 即可调整阈值，例如：

```python
from xmlx_vlm.ai_trader.market_service import MarketDataService, AlertConfig

svc = MarketDataService(
    alert_config=AlertConfig(
        oi_1h_change_threshold_pct=3.0,
        large_trade_notional=30_000.0,
        book_imbalance_threshold=0.55,
    )
)
svc.start()
```

## 目录结构

```
xmlx_vlm/ai_trader/
├── cli.py                    # 聊天入口
├── config.py                 # 默认配置
├── market_service/           # 机构级行情服务
│   ├── models.py             # 领域模型
│   ├── events.py             # 事件总线与事件定义
│   ├── state.py              # 内存状态机
│   ├── indicators.py         # 技术指标计算
│   ├── ws_client.py          # Hyperliquid WebSocket 客户端
│   ├── market_info.py        # 成交量排名等静态信息
│   ├── alerts.py             # 阈值警报引擎
│   ├── service.py            # MarketDataService 编排器
│   └── tests/                # 单元测试
├── oms/                      # 订单管理系统
│   ├── core/                 # OMS 引擎、订单、持仓、账户
│   ├── risk/                 # 风控规则
│   ├── execution/            # 执行适配器（paper / hyperliquid）
│   ├── audit/                # 审计日志
│   ├── circuit/              # 熔断与急停
│   └── events/               # OMS 内部事件总线
├── tools/
│   ├── market.py             # 行情数据工具（优先读本地服务，可回退 REST）
│   ├── chart.py              # K 线图工具
│   ├── trading.py            # 交易执行工具
│   └── registry.py           # 工具注册表
├── data/                     # 历史数据与图表
└── logs/                     # 日志与审计
```

## 安全提示

- **默认纸盘**：不配置 API Key 时所有 `place_order` 都是模拟成交。
- **实盘双重确认**：必须同时满足 `AI_TRADER_LIVE=1` 环境变量与相应凭证配置。
- **风控硬限制**：`trading` 工具内部强制调用 OMS 风控，无法绕过。
- **API Key 最小权限**：实盘交易时只配置交易权限，不要开启提现、划转权限。
- **生产推荐外部签名器**：使用 `HL_SIGNER_ENDPOINT` 让私钥不落地，避免写入任何文件。
- **本地推理**：模型、数据、账户信息均保存在本地。
- **审计留痕**：所有订单意图、风控决策、成交、熔断写入 `logs/oms_audit/`。

## 实盘上线阶段

机构资金全自动不允许一步到位，必须按阶段验证：

| 阶段 | 模式 | 目标 | 建议时间 |
|------|------|------|----------|
| Phase 0 | 纯纸盘 | 验证策略信号、OMS 状态机、风控逻辑 | ≥2 周 |
| Phase 1 | 只读实盘 | OMS 接入 Hyperliquid 仅查询余额/持仓/订单，不下单 | ≥3 天 |
| Phase 2 | 小单半自动 | 每单需人工确认，单笔名义金额极小 | ≥2 周 |
| Phase 3 | 小单全自动 | 仓位、频率、单笔限制在极低水平 | ≥1 月 |
| Phase 4 | 生产规模 | 经统计验证夏普、最大回撤、盈亏比达标后逐步放大 | 持续 |

## 后续规划

- [x] Hyperliquid 实时行情服务（WebSocket + 内存状态机）
- [x] 自动监控 24h 成交额 Top 30 币种
- [x] 事件总线与阈值警报引擎
- [x] 行情工具优先本地服务、REST 回退
- [x] Hyperliquid 实盘交易模式
- [ ] AI Agent 主动订阅警报并自动播报
- [ ] 多交易所行情聚合
- [ ] Web Dashboard 展示实时监控与警报
- [ ] 交易复盘与策略学习
