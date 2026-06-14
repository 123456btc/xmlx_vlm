# AI Trader — 聊天即交易

基于 XMLX-VLM 的本地开源 AI 量化交易助手。用户启动模型、进入聊天界面后，通过自然语言即可完成行情分析、K 线图绘制、模拟交易等操作。

## 快速开始

### 1. 安装依赖

```bash
cd /Users/hongjianjia/xmlx_vlm
.venv/bin/python -m pip install -r requirements.txt
```

> 若使用 Python 3.14 且 `matplotlib`/`ccxt` 安装失败，本项目已内置公开 REST API 回退和 Pillow 绘图，不强制依赖它们。

### 2. 启动服务（复用 service.sh 默认模型）

```bash
./service.sh start
```

这会启动 `xmlx_vlm.server` 并加载默认模型。AI Trader 默认连接 `http://localhost:8080`。

### 3. 启动 AI Trader

```bash
# 连接 service.sh 启动的服务（推荐）
xmlx_vlm.ai-trader

# 或本地加载模型（不需要 server）
xmlx_vlm.ai-trader --local --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
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
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--server` | 推理服务地址 | `http://localhost:8080` |
| `--local` | 不连接 server，本地加载模型 | `False` |
| `--model` | 本地模型路径 / HuggingFace repo id | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| `--temperature` | 采样温度 | `0.3` |
| `--max-tokens` | 最大生成长度 | `2048` |

> 建议：配合 `service.sh start` 使用 `--server` 模式，可以复用已加载模型、降低显存占用。

## 核心能力

- **行情查询（机构级）**：统一使用 Hyperliquid 公开 API，支持 L1 ticker、历史 K 线、**5m/15m/1h 多周期聚合分析**、L2 订单簿深度、逐笔成交流、资金费率、持仓量、综合市场摘要。
- **K 线图绘制**：使用 Pillow 本地绘制蜡烛图、EMA、成交量、RSI。
- **视觉分析**：把 K 线图直接喂给本地 VLM，让模型看图说话。
- **模拟交易**：默认纸盘模式，记录虚拟持仓和 PnL。
- **风控检查**：下单前自动检查单日亏损、仓位暴露等限制。
- **工具调用**：模型自主决定何时查行情、画图、下单。

## 配置

当前配置集中在 `xmlx_vlm/ai_trader/config.py`：

- `DEFAULT_EXCHANGE`：默认交易所（固定为 hyperliquid）
- `DEFAULT_SYMBOL`：默认交易对（如 BTC/USDC）
- `DEFAULT_TIMEFRAME`：默认 K 线周期
- `DEFAULT_RISK`：默认风控阈值

## 工具列表

| 工具 | 说明 |
|------|------|
| `market_data` | 查 ticker / OHLCV |
| `render_chart` | 生成 K 线图 |
| `trading` | 查询持仓、下单、平仓、急停 |

## 安全提示

- **默认纸盘**：不配置 API Key 时所有 `place_order` 都是模拟成交。
- **风控硬限制**：`trading` 工具内部强制调用风控检查，无法绕过。
- **API Key 最小权限**：实盘交易时只配置交易权限，不要开启提现权限。
- **本地推理**：模型、数据、账户信息均保存在本地。

## 架构

```
xmlx_vlm/ai_trader/
├── cli.py          # 聊天入口
├── config.py       # 默认配置
├── tools/          # 工具实现
│   ├── market.py   # 行情数据
│   ├── chart.py    # K 线图
│   ├── trading.py  # 交易执行
│   └── registry.py # 工具注册表
├── data/           # 历史数据与图表
└── logs/           # 交易日志
```

## 后续规划

- [ ] 实盘交易模式（ccxt 真实下单）
- [ ] 多交易所账户余额/持仓查询
- [ ] Computer-Use 截图交易所网页
- [ ] 语音交互
- [ ] Web Dashboard
- [ ] 交易复盘与策略学习
