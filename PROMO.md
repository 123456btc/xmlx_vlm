# XMLX-VLM 宣传文案库

> 适用于 X (Twitter)、LinkedIn、即刻、微信公众号等渠道。
> 中文帖每条控制在 280 字符内（X 标准），英文帖同理。

---

## 方案 A：完整 Twitter Thread（8 条，中文）—— 推荐首发

【1/8】钩子帖
你的研报、病历、卷宗，正在变成别人的训练数据。

每次调用云 API，你都在签署一份看不见的《数据赠与协议》。

本地 AI 不是可选项，是刚需。

🧵

【2/8】定位帖
XMLX-VLM = Apple Silicon 上的私有 Vision-Language 大脑

✅ 零网络调用
✅ 零数据外泄
✅ 零云订阅

你的 Mac 就是你的数据中心。

【3/8】性能帖
性能？不是妥协，是优势。

DFlash 投机解码 + MTP 并行 draft = 长文档毫秒级响应
TurboQuant KV 压缩 = 70B 模型在 128GB Mac 上流畅运行

本地 ≠ 慢

【4/8】技术壁垒帖
推理模型 + 结构化输出 = 行业难题。

模型在 <think> 里自由推理，然后瞬间切换到 JSON Schema 约束。
我们叫这 Thinking-Aware Constrained Generation。

别人还在用后处理解析，我们在 logits 层解决。

【5/8】场景帖
金融研究员：本地解析研报，生成因子定义 JSON
律师：扫描文档结构化分析，零云端暴露
医生：病历影像本地推理，HIPAA by 架构
官员：涉密文件本地处理，完整审计轨迹

一套引擎，所有敏感场景。

【6/8】兼容性帖
OpenAI SDK？Claude SDK？都支持。

把 base URL 改成 localhost:5118，一行代码不改。
Cursor、Claude Code、LangChain、PydanticAI 全部接入。

本地大脑，熟悉的接口。

【7/8】功能清单帖
Speculative decoding ✅
Tool-call auto-recovery ✅
SSD-persistent prefix cache ✅
Dual-protocol API ✅
MCP tool calling ✅
Embedding + Rerank ✅
AI Trader（本地量化助手）✅

【8/8】CTA 帖
开源。MIT。

git clone → pip install → ./service.sh start
默认自带 chat UI + DFlash 投机解码 + API 认证

5 分钟，你的数据不再流浪。

https://github.com/123456btc/xmlx_vlm

---

## 方案 B：浓缩单帖版（适合单独发）

XMLX-VLM：Apple Silicon 上的本地私有化 Vision-Language AI

你的研报、病历、卷宗——全部在本地 Mac 上解析、推理、结构化输出。零云调用，零数据外泄。

DFlash 投机解码 + Thinking-Aware 约束生成 + 双协议 API（OpenAI/Claude）

git clone 即用，5 分钟部署。

https://github.com/123456btc/xmlx_vlm

## 方案 C：金融垂直版（精准打击量化圈）

量化人的噩梦：你把 proprietary 因子喂给 GPT-4，它在下一个版本里"巧合地"学会了类似的策略。

XMLX-VLM 解法：
→ 本地解析研报 PDF 和图表
→ Thinking 模式推理因子失效假设
→ JSON Schema 输出标准化因子定义
→ MCP 调用本地回测引擎
→ AI Trader：本地行情 WS 常驻，单请求 Bull/Bear 多空对抗辩论规避单边偏见，平仓自动反思复盘并写入本地 SQLite 记忆环进行自适应闭环学习

全程零网络调用。你的 alpha，永远只属于你。

https://github.com/123456btc/xmlx_vlm

---

## 方案 D：隐私安全版（精准打击法律/政务/医疗）

云 AI 的隐私承诺 = "我们相信他们不会偷看"

XMLX-VLM 的隐私架构 = "物理上就不可能外泄"

Apple Silicon 本地运行
Vision-Language 全栈推理
结构化输出 + Tool Calling + MCP
数据主权是设计，不是功能

律师、医生、官员——处理敏感信息的人，需要的不是更长的隐私政策，而是本地硬件。

https://github.com/123456btc/xmlx_vlm

---

## 方案 E：英文版 Thread（8 条，X 全球受众）

【1/8】
Your research reports, medical records, legal files — they're becoming someone else's training data.

Every cloud API call is an invisible data donation agreement.

Local AI isn't optional. It's mandatory.

🧵

【2/8】
XMLX-VLM = Private Vision-Language AI on Apple Silicon

✅ Zero network calls
✅ Zero data leakage
✅ Zero cloud subscriptions

Your Mac is your data center.

【3/8】
Performance? Not a compromise. An advantage.

DFlash speculative decoding + MTP parallel drafting = millisecond response on long docs
TurboQuant KV compression = 70B models on 128GB Mac

Local ≠ Slow

【4/8】
Reasoning + structured output has been an industry nightmare.

Model thinks freely inside <think>, then instantly switches to JSON Schema constraints.
We call it Thinking-Aware Constrained Generation — solved at the logits layer, not post-processing.

【5/8】
Finance: Parse research PDFs locally, emit structured factor definitions
Legal: Analyze scanned documents, zero cloud exposure
Healthcare: Local inference on medical imaging, HIPAA by architecture
Government: Classified document processing, full audit trail

One engine. Every sensitive domain.

【6/8】
OpenAI SDK? Claude SDK? Both supported.

Change base URL to localhost:5118. Zero code changes.
Cursor, Claude Code, LangChain, PydanticAI — all connected.

Local brain. Familiar interface.

【7/8】
Speculative decoding ✅
Tool-call auto-recovery ✅
SSD-persistent prefix cache ✅
Dual-protocol API ✅
MCP tool calling ✅
Embedding + Rerank ✅
AI Trader (local quant assistant) ✅

【8/8】
Open source. MIT.

git clone → pip install → ./service.sh start
Chat UI + DFlash speculative decoding + API auth — all defaults.

5 minutes. Your data stops wandering.

https://github.com/123456btc/xmlx_vlm

---

## 方案 F：英文浓缩单帖版

XMLX-VLM: Privacy-first local Vision-Language AI on Apple Silicon.

Parse documents, reason through hypotheses, emit structured JSON — all on your Mac. Zero cloud. Zero leakage.

DFlash speculative decoding + Thinking-Aware constrained generation + OpenAI/Claude dual-protocol API.

git clone and run in 5 minutes.

https://github.com/123456btc/xmlx_vlm

---

## 方案 G：犀利对比版（中文单帖）

云 API：
❌ 数据上传 = 不可撤销的赠与
❌ 模型推理 = 在别人的 GPU 上裸奔
❌ 隐私政策 = "我们相信他们不会偷看"

XMLX-VLM：
✅ 物理隔离 = 网线拔掉照样跑
✅ 本地推理 = 你的数据连 kernel 都不出
✅ 开源可审计 = 每一行代码你都能读

本地 AI 不是复古，是降维打击。

https://github.com/123456btc/xmlx_vlm

---

## 方案 H：技术人视角版（中文单帖）

开源社区终于有人把"本地推理"做到生产级了。

XMLX-VLM 不是玩具：
- DFlash 投机解码（比标准生成快 1.5-2.3x）
- Thinking-Aware JSON Schema（推理模型也能结构化输出）
- APC SSD 持久缓存（重启后照样毫秒级 warm-start）
- Tool-call auto-recovery（量化模型崩了也能修）

git clone，pip install，./service.sh start。
5 分钟，你的 Mac 变成私有 AI 数据中心。

https://github.com/123456btc/xmlx_vlm

---

## 方案 I：即刻/朋友圈短文案（更口语化）

一个冷知识：你每次把文件丢给 ChatGPT，都是在做慈善——免费帮 OpenAI 扩充训练语料。

XMLX-VLM 让你把 Vision-Language 模型完整跑在 Mac 上。研报、病历、卷宗、图纸——全部本地解析，零上传。

而且性能不比云端慢。DFlash 投机解码 + TurboQuant，70B 模型在 Mac Studio 上跑得飞起。

开源，免费，5 分钟装好。

https://github.com/123456btc/xmlx_vlm


---

## 方案 K：AI Trader 本地量化助手版（中文 Thread，10 条）

【1/10】钩子帖
你把 K 线图、盘口、资金费率截图发给云端 AI 分析，

它不光"看见"了你的策略思路，还可能把它变成下一版模型的"常识"。

量化人的数据主权，从本地行情大脑开始。

🧵

【2/10】定位帖
XMLX-VLM 新增 AI Trader

= Apple Silicon 上的私有化量化分析助手

✅ 统一 Hyperliquid 数据源
✅ 5m / 15m / 1h 多周期聚合
✅ 单请求 Bull/Bear 多空对抗辩论规避单边偏见
✅ 平仓自动反思复盘，写入本地 SQLite 记忆环进行自适应闭环学习

你的 Mac = 行情终端 + 研究员 + 辩论委员会 + 风控台。

【3/10】数据源帖
不是"接了个交易所 API"那么简单。

AI Trader 从 Hyperliquid 实时拉取：
→ 最新价 / 24h 高低点 / 成交量
→ L2 买卖深度、spread、深度失衡率
→ 主动买/卖压力、大单识别
→ 资金费率、持仓量、premium

机构级数据，零网络泄露。

【4/10】多周期帖
只看 1h 就做判断？那是散户行为。

AI Trader 默认同时分析：
→ 5m：短线情绪
→ 15m：中短线结构
→ 1h：趋势结构

三周期共振才出信号，分歧就喊观望。

【5/10】工具调用帖
你说"分析一下 ETH"，它不是聊天回复，而是：

1. 调用 get_multi_timeframe_summary
2. 调用 get_market_summary
3. 本地渲染 K 线图
4. 输出结构化分析 + 交易建议

模型自己决定查什么、画什么、算什么。

【6/10】多空对抗辩论帖
怎么让本地小模型像顶尖分析师一样冷静？

我们引入了“单请求对抗辩论（Adversarial Debate）”：
- Bull Analyst 寻找做多逻辑，Bear Analyst 寻找做空隐患
- 在同一个推理周期内自我反思与驳斥，得出最理性的共识决策
- 既保留了辩论深度，又避免了多次本地 LLM 调用的高延迟！

【7/10】闭环反思记忆帖
亏损了？本地大脑不光写日志，还会学习。

当一笔模拟/实盘订单平仓时：
- 后台自动触发 Reflection Task 评估实际盈亏（PnL）
- 总结进场、止损与行情误差，写入本地 SQLite 记忆库
- 下次类似行情触发警报时，自动把历史教训注入 Context，实现真正的增量闭环学习。

【8/10】隐私安全帖
行情数据、持仓逻辑、策略偏好——

在云 API 里跑一遍，就是一份不可撤销的数据赠与。

AI Trader 全部本地推理：
→ 行情不过第三方节点
→ 图表不在云端生成
→ 纸盘状态存在本地日志

你的 alpha，连路由器都不经过。

【9/10】使用帖
```bash
./service.sh start
xmlx_vlm.ai-trader
```

然后直接聊：
- "BTC 现在多少钱？"
- "画一张 ETH 1小时 K 线图"
- "看看盘口买卖失衡吗"
- "模拟买入 0.01 BTC"

5 分钟，你的 Mac 变成私有量化工作站。

【10/10】CTA 帖
开源。MIT。

git clone → pip install → ./service.sh start → xmlx_vlm.ai-trader

本地行情、本地分析、本地决策。

https://github.com/123456btc/xmlx_vlm

---

## 方案 L：AI Trader English Thread（10 posts）

【1/10】
Every chart, order-book screenshot, and funding-rate question you send to a cloud AI is a data donation.

Your edge becomes their training data.

Local quant intelligence isn't nostalgia. It's survival.

🧵

【2/10】
Meet AI Trader on XMLX-VLM.

= Private quant assistant running on Apple Silicon

✅ Unified Hyperliquid data feed
✅ 5m / 15m / 1h multi-timeframe analysis
✅ Single-request Bull/Bear adversarial debate to eliminate confirmation bias
✅ Automated post-trade reflection logged to local SQLite for adaptive closed-loop learning

Your Mac = terminal + researcher + debate panel + risk desk.

【3/10】
Not just "an exchange API wrapper".

AI Trader pulls from Hyperliquid in real time:
→ ticker, 24h range, volume
→ L2 depth, spread, depth imbalance
→ buy/sell pressure, whale-trade detection
→ funding rate, open interest, premium

Institutional-grade data. Zero network leakage.

【4/10】
Analyzing on a single timeframe is amateur.

AI Trader defaults to three:
→ 5m: short-term sentiment
→ 15m: intraday structure
→ 1h: trend structure

Only aligned signals get published. Divergence = wait.

【5/10】
Say "analyze ETH" and it doesn't just chat.

It calls:
1. get_multi_timeframe_summary
2. get_market_summary
3. render_chart locally
4. emits structured analysis + trade suggestion

The model decides what to query, plot, and compute.

【6/10】
How do we make local LLMs make rational decisions like a wall street desk?

We introduced "Single-Request Adversarial Debate":
- A Bull Analyst defends the long setup, a Bear Analyst counters with short/flat risks.
- They debate and reconcile to form a consensus in one single LLM round-trip.
- Maximizes critical thinking depth while keeping local latency at a minimum!

【7/10】
Losses happen. But local brains learn from mistakes.

Upon order closure:
- A non-blocking background task starts a post-trade reflection.
- Analyzes entry/exit offset and realized PnL, then writes lessons-learned into SQLite.
- Feeds recent lessons back into the model's system prompt during future signals.

【8/10】
Your market data, your position logic, your strategy bias —

Running it through a cloud API is an irreversible data gift.

AI Trader runs everything locally:
→ no third-party market-data node
→ no cloud chart generation
→ paper state stored in local logs

Your alpha never touches a router.

【9/10】
```bash
./service.sh start
xmlx_vlm.ai-trader
```

Then just chat:
- "What's BTC at?"
- "Draw ETH 1h chart"
- "Is order book imbalanced?"
- "Paper trade 0.01 BTC"

5 minutes. Your Mac becomes a private quant workstation.

【10/10】
Open source. MIT.

git clone → pip install → ./service.sh start → xmlx_vlm.ai-trader

Local data. Local analysis. Local decisions.

https://github.com/123456btc/xmlx_vlm

---

## 方案 J：AFRE 生态版（量化研究垂直 Thread，10 条）

【1/10】钩子帖
量化研究有个 dirty secret：

你把 proprietary 因子逻辑喂给 GPT-4 做分析，
六个月后 OpenAI 的"新能力"里"巧合地"出现了相似的推理路径。

这不是阴谋论。这是数据赠与的必然结果。

🧵

【2/10】问题帖
传统量化研究三大痛点：

① 研报是图片/PDF，云端 OCR 把内容泄露了
② 因子假设靠直觉，没有结构化追溯
③ 回测和推理割裂， agent 只能"建议"不能"执行"

云 AI 让研究员变成了数据慈善家。

【3/10】AFRE 方法论帖
AFRE = AI Factor Research Engine

不是回测玩具，是研究操作系统：
→ 因子谱系：研究因子为什么被发明、如何传播、为何失效
→ 发明者思维：模拟创造者当时的约束、激励和知识栈
→ 假设驱动：每个新因子必须来自特定的失效假设
→ 反过拟合治理：walk-forward、regime split、多重检验惩罚

这是方法论。但它需要一个本地大脑来执行。

【4/10】XMLX-VLM 定位帖
XMLX-VLM 就是 AFRE 的私有化 AI 大脑。

Apple Silicon 本地运行
零网络调用
零数据外泄

你的 Mac = 数据中心 + 推理引擎 + 安全堡垒

【5/10】视觉解析帖
AFRE 要读研报。研报不是纯文本——是 PDF、是图表、是截图。

XMLX-VLM 的 Vision 能力：
✅ 本地解析研报图片和扫描件
✅ 从 K 线截图提取结构化特征
✅ 财报表格的视觉理解
✅ 零 OCR SaaS，零云端暴露

研报内容连路由器都不经过。

【6/10】推理帖
AFRE 要模拟发明者思维。这需要深度推理。

XMLX-VLM 的 Thinking 模式：
→ 模型在 <think> 里自由分析因子失效原因
→ 然后瞬间切换到 JSON Schema 约束
→ 输出带假设追溯的标准化因子定义

别人让模型"说说想法"，我们让模型"结构化地证明假设"。

【7/10】工具链帖
AFRE 要跑实验。实验不是聊天，是计算。

XMLX-VLM 通过 MCP 调用：
→ 本地回测引擎
→ 信号生成器
→ 风险指标计算器
→ 组合优化器

Agent 不仅能"建议做多"，还能"执行回测并返回夏普比率"。

【8/10】治理帖
AFRE 的核心是反过拟合治理。

XMLX-VLM 的结构化输出强制执行：
✅ walk-forward 参数窗口
✅ out-of-sample 验证标记
✅ regime split 标识
✅ turnover penalty 系数
✅ multiple testing penalty

不是"感觉这个因子不错"。是"机器可审计的假设-验证闭环"。

【9/10】性能帖
本地 ≠ 慢。

DFlash 投机解码 = 长文档分析 1.5-2.3x 加速
TurboQuant KV 压缩 = 70B 模型在 128GB Mac 流畅运行
APC SSD 缓存 = 重复查询毫秒级响应，重启后照样生效

Mac Studio 上的本地推理，延迟比云端 API 还低。

【10/10】CTA 帖
AFRE 是方法论框架（开源）。
XMLX-VLM 是私有化推理引擎（开源）。

两者结合 = 完整的本地量化研究操作系统。

git clone → pip install → ./service.sh start
5 分钟，你的 alpha 不再流浪。

https://github.com/123456btc/xmlx_vlm

---

## 方案 M：AI Trader 日本語 Thread（10 posts）

【1/10】
チャート、板情報、資金調達率（ファンディングレート）のスクリーンショットをクラウドAIに送信するたびに、あなたの戦略データは無償で寄付されています。

あなたのエッジが、彼らのトレーニングデータになってしまうのです。

ローカルでのクオンツインテリジェンスは、単なる懐古趣味ではありません。生き残りのための必須条件です。

🧵

【2/10】
XMLX-VLM に「AI Trader」が新登場。

= Apple Silicon 上で動作するプライベートなローカル量化（クオンツ）アシスタント

✅ Hyperliquid リアルタイムデータソース統合
✅ 5分足 / 15分足 / 1時間足のマルチタイムフレーム分析
✅ 単一リクエストでの Bull/Bear 対抗討論によるバイアスの排除
✅ ローカル SQLite 記憶ループによるポジションクローズ後の自動反省・学習

あなたの Mac ＝ 情報端末 ＋ 研究員 ＋ 討論パネル ＋ リスク管理デスク。

【3/10】
単なる「取引所 API ラッパー」ではありません。

AI Trader は Hyperliquid からリアルタイムにデータを取得します：
→ 最新価格 / 24時間高安 / 出来高
→ L2 板の深さ、スプレッド、インバランス率
→ 指値買い/売り圧力、大口注文（クジラ）検知
→ ファンディングレート、未決済建玉（OI）、プレミアム

機関投資家グレードのデータを、外部への漏洩ゼロで。

【4/10】
単一の時間軸だけで判断するのは素人です。

AI Trader はデフォルトで3つの時間軸を同時に分析します：
→ 5分足：短期のセンチメント
→ 15分足：日中の構造
→ 1時間足：トレンド構造

シグナルが一致した時のみ実行し、不一致の場合は「待機」を選択。

【5/10】
「ETH を分析して」と指示すると、単にテキストで答えるだけでなく、以下のツールを呼び出します：

1. get_multi_timeframe_summary
2. get_market_summary
3. ローカルで K 線チャートをレンダリング（Pillow）
4. 構造化された分析結果 ＋ 交易提案を出力

モデル自身が何を調べ、何を描き、何を計算すべきかを判断します。

【6/10】
ローカルの小型モデルをウォール街のプロのように冷静に判断させるには？

「単一リクエストでの Bull/Bear 対抗討論（Adversarial Debate）」を導入しました：
- Bull Analyst が買いの根拠を探し、Bear Analyst が空売りのリスクを指摘
- 単一の推論サイクル内で自己批判と反論を行い、最も合理的な合意形成へ
- 討論の深さを維持しつつ、ローカル LLM の複数回呼び出しによる高遅延を回避！

【7/10】
損失は発生します。しかし、ローカルの脳は失敗から学びます。

ポジションがクローズされると：
- バックグラウンドタスクが自動でポストトレード反省（Reflection）を実行
- エントリー/エグジットのズレと実現損益（PnL）を分析し、ローカル SQLite に教訓を保存
- 次回のシグナル発生時に、過去の教訓をシステムプロンプトのコンテキストへ自動注入。

【8/10】
市場データ、ポジション管理ロジック、戦略の偏り——

これらをクラウド API で実行することは、取り返しのつかないデータの贈与です。

AI Trader はすべてをローカルで推論します：
→ サードパーティ의 데이터노드を経由しない
→ クラウドでチャートを生成しない
→ ペーパートレードのログはローカルに保存

あなたの Alpha は、ルーターすら通過しません。

【9/10】
```bash
./service.sh start
xmlx_vlm.ai-trader
```

あとはチャットするだけ：
- 「BTCの価格は？」
- 「ETHの1時間足チャートを描いて」
- 「板情報のインバランスは？」
- 「0.01 BTC をデモ取引」

5分で、あなたの Mac が完全プライベートなクオンツワークステーションになります。

【10/10】
オープンソース。MITライセンス。

git clone → pip install → ./service.sh start → xmlx_vlm.ai-trader

ローカルデータ、ローカル分析、ローカル意思決定。

https://github.com/123456btc/xmlx_vlm

---

## 方案 N：AI Trader 한국어 Thread（10 posts）

【1/10】
차트, 호가창(L2), 펀딩비 스크린샷을 클라우드 AI에 보낼 때마다, 당신의 독점 전략 데이터는 무상으로 기부되고 있습니다.

당신의 엣지(Edge)가 그들의 학습 데이터가 되는 것입니다.

로컬 퀀트 인텔리전스는 과거의 향수가 아닙니다. 생존의 필수 조건입니다.

🧵

【2/10】
XMLX-VLM에 "AI Trader"가 새롭게 추가되었습니다.

= Apple Silicon에서 실행되는 나만의 사설 퀀트 트레이딩 비서

✅ 통합 Hyperliquid 실시간 데이터 피드
✅ 5분 / 15분 / 1시간 다중 프레임 통합 분석
✅ 단일 요청 내 Bull/Bear 상호 토론으로 편향 극복
✅ 포지션 청산 후 실시간 복기 및 로컬 SQLite 기반 자가 학습 루프

당신의 Mac ＝ 시세 단말기 ＋ 연구원 ＋ 토론 패널 ＋ 리스크 관리 부서.

【3/10】
단순히 "거래소 API 래퍼"가 아닙니다.

AI Trader는 Hyperliquid에서 직접 실시간 데이터를 가져옵니다:
→ 현재가 / 24시간 고가·저가 / 거래량
→ L2 호가 깊이, 스프레드, 매수/매도 불균형 비율
→ 시장가 매수/매도 압력, 고래 거래 감지
→ 펀딩비, 미결제약정(OI), 프리미엄

기관급 데이터를 외부 노출 전혀 없이 받아보세요.

【4/10】
단일 시간대만 보고 판단하는 것은 초보적인 방식입니다.

AI Trader는 기본적으로 세 가지 프레임을 동시 분석합니다:
→ 5분: 단기 심리
→ 15분: 당일 구조
→ 1시간: 추세 구조

세 프레임의 시그널이 일치할 때만 실행하며, 의견이 갈릴 때는 '관망'을 제시합니다.

【5/10】
"ETH 분석해줘"라고 입력하면 단순 텍스트 답변이 아닌, 다음과 같은 도구들을 직접 실행합니다:

1. get_multi_timeframe_summary
2. get_market_summary
3. 로컬 K라인 차트 렌더링 (Pillow)
4. 구조화된 분석 ＋ 트레이딩 제안 출력

모델 스스로 무엇을 조회하고, 그리고, 계산할지 결정합니다.

【6/10】
로컬 소형 모델이 월스트리트 전문가처럼 침착하게 판단하도록 만드는 법은 무엇일까요?

우리는 "단일 요청 내 대항적 토론(Adversarial Debate)"을 도입했습니다:
- Bull Analyst가 매수 근거를 찾고, Bear Analyst가 숏/관망 위험을 지적합니다.
- 단 한 번의 LLM 요청 내에서 스스로 반박하고 합의점에 도달합니다.
- 토론의 깊이는 유지하면서, 로컬 LLM의 다중 호출로 인한 지연 시간(Latency)을 최소화했습니다!

【7/10】
손실은 발생할 수 있습니다. 하지만 로컬의 두뇌는 실수로부터 배웁니다.

포지션이 청산될 때:
- 백그라운드 태스크가 자동으로 사후 복기(Reflection)를 실행합니다.
- 진입/청산 오차와 실현 손익(PnL)을 분석하고 로컬 SQLite에 학습 기록을 저장합니다.
- 다음 신호 발생 시, 과거의 교훈을 시스템 프롬프트에 자동으로 주입하여 지식을 업데이트합니다.

【8/10】
시장 데이터, 포지션 로직, 전략적 편향——

이러한 데이터를 클라우드 API에 입력하는 것은 영구적인 데이터 기부입니다.

AI Trader는 모든 연산을 로컬에서 수행합니다:
→ 제3자 데이터 노드를 거치지 않음
→ 클라우드 차트 생성 없음
→ 모의 거래 상태를 로컬 로그에만 저장

당신의 Alpha는 인터넷 공유기조차 통과하지 않습니다.

【9/10】
```bash
./service.sh start
xmlx_vlm.ai-trader
```

그 후 대화를 시작하세요:
- "BTC 지금 얼마야?"
- "ETH 1시간 봉 차트 그려줘"
- "호가창 매수 불균형 확인해줘"
- "0.01 BTC 모의 거래 주문"

5분 만에 당신의 Mac이 완전한 프라이빗 퀀트 워크스테이션으로 변신합니다.

【10/10】
오픈소스. MIT 라이선스.

git clone → pip install → ./service.sh start → xmlx_vlm.ai-trader

로컬 데이터, 로컬 분석, 로컬 의사결정.

https://github.com/123456btc/xmlx_vlm

