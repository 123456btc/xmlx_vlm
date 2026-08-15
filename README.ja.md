# XMLX-VLM

<p align="center">
  <strong>プライバシー優先のローカルVision-Language AI & 機関投資家向け完全非公開型ローカルモデル量化取引システム</strong>
</p>

<p align="center">
  <em>Apple Siliconネイティブ推論エンジン。外部流出ゼロ。機密文書の解析や、セキュリティに特化したローカルAIアルゴリズム・クオンツ取引向けに設計。</em>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform: macOS" src="https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> | <a href="README.zh.md">🇨🇳 中文</a> | <strong>🇯🇵 日本語</strong> | <a href="README.ko.md">🇰🇷 한국어</a>
</p>

---

## 🚀 なぜXMLX-VLMなのか？

**機密データを扱うプロフェッショナルにとって、プライバシーは機能ではなく——前提条件である。**

法律文書、医療記録、政府文書、独自の研究、トレーディングアルゴリズム——それらがクラウドAPIを通過した瞬間、あなたは管理を失う。それらは学習データとなり、ログに記録され、召喚状の対象となる。

**XMLX-VLM**は、**ローカルファーストでプロダクショングレードのVision-Language推論エンジン**であり、Apple Silicon上で完全に動作する。文書を読み込み、画像を解析し、複雑な問題を推論し、構造化された出力を生成する——**ネットワーク通信を一切行わずに**。

クラウドサブスクリプションなし。データ保持ポリシーなし。第三者の利用規約なし。あなたのMacと、あなたのデータと、あなたのモデルのみ。

> **データ主権がアーキテクチャである。他のすべてはその上に構築される。**

### 🧬 AFREエコシステム

XMLX-VLMは、**AFRE（AI Factor Research Engine）**エコシステムの**プライベートAIブレイン**である。AFREは、DDD（ドメイン駆動設計）、ヘキサゴナルアーキテクチャ、クリーンアーキテクチャを基盤に構築された、ドメイン特化型でエージェント強化型の定量研究プラットフォームである。

AFREは**マーケットファクターの系譜**を研究する：なぜ発明されたのか、どのように普及したのか、なぜ減衰したのか、そしてどのような破綻した仮説が現代の仮説を生み出せるのか。XMLX-VLMは、以下を提供することでAFREのエージェントランタイムを駆動する：

| AFREの能力 | XMLX-VLMがローカルで実現すること |
|-----------------|------------------------------|
| **ファクター系譜インテリジェンス** | 研究PDFやチャート画像を解析し、視覚的文書から構造化されたファクター履歴を抽出する |
| **発明者思考シミュレーション** | 深い推論（`<think>`モード）により、ファクター創作者の制約、インセンティブ、知識スタックをシミュレートする |
| **仮説中心の研究** | JSON Schema制約付き出力により、生成されたすべてのファクター派生型に検証可能な仮説と破綻仮説の追跡が付与される |
| **再現可能な実験** | ツール呼び出し＋MCPにより、ローカルのバックテスターやシグナル生成器と連携。実験はあなたのハードウェア上で実行され、設計上監査可能 |
| **オーバーフィッティング防止ガバナンス** | 構造化出力により、ウォークフォワードパラメータ、レジーム分割、ターンオーバー罰則を機械可読なスキーマとして強制する |
| **知識の進化** | Embedding＋Rerankにより、検証済みの発見を統制された照会可能なローカル知識ベースにインデックス化する |
| **マルチエージェント並列研究** | 継続的バッチング＋推測的デコーディングにより、複数のAIワーカーがレイテンシ崩壊なしに独立して推論できる |

**AFREが方法論である。XMLX-VLMがそれを可能にするプライベート推論層である。**

AFREはXMLX-VLMの定量的ファイナンスにおける代表的実装を体現しているが、同じローカルプライバシーアーキテクチャは、法務、医療、政府、企業のR&D分野でも同様に機能する。

---

## 🎯 対象ユーザー

| 分野 | 機密データ | XMLX-VLMがローカルで行うこと |
|--------|---------------|---------------------------|
| **定量的ファイナンス** | 独自ファクター、内部研究レポート、アルファシグナル | PDFレポートやチャート画像を解析。ファクター仮説を推論。構造化ファクター定義を出力。MCP経由でローカルバックテスターを呼び出す。AI TraderでHyperliquidの市場データをローカル分析・取引 |
| **法務** | 案件ファイル、契約書、開示文書、クライアント通信 | 文書画像やスキャンを分析。構造化条項を抽出。法的議論を推論。変更追跡付きサマリーを生成 |
| **政府** | 機密ブリーフィング、政策草案、市民記録、諜報画像 | 機密画像やスキャン文書を処理。諜報レポート用の構造化出力。ローカルハードウェア上で完全な監査証跡 |
| **医療** | 患者記録、医療画像、臨床メモ、検査結果 | 医療文書画像を解析。鑑別診断を推論。臨床サマリー用の構造化出力。アーキテクチャ上HIPAA準拠 |
| **企業R&D** | 営業秘密、特許草案、実験データ、内部文書 | 技術図面のVision-Language理解。研究仮説の推論。実験設計用の構造化出力 |

---

## 🎯 コア機能

| 機能 | 得られるもの |
|------------|-------------|
| **ローカル文書インテリジェンス** | PDF、スキャン文書、スクリーンショット、画像重視のレポートを直接モデルに投入。OCR SaaS不要。クラウドVision API不要。文書はlocalhostから一切外出しない。 |
| **推論付き構造化出力** | `thinking`モードを有効化して深い推論を行い、その後最終出力にJSON Schema制約を適用。監査対応レポート、ファクター定義、臨床サマリーに最適。 |
| **デュアルプロトコルAPI** | 1つのサーバーがOpenAI（`/v1/chat/completions`）とAnthropic（`/v1/messages`）の両方のプロトコルを話す。Cursor、Claude Code、LangChain、PydanticAIのバックエンドとして差し込み可能——すべてのトラフィックは`localhost:5118`上に留まる。 |
| **ローカルツール呼び出し＆MCP** | MCP経由で、ローカルデータベース、バックテスター、EHRシステム、案件管理ツール、文書パイプラインに接続。モデルがあなたのツールを呼び出す。あなたのデータはマシンから出ない。 |
| **プライベート知識用Embedding＆Rerank** | 内部文書、研究ノート、案件ファイル、患者履歴をインデックス化。独自の知識ベースに対するセマンティック検索——クラウド露出ゼロ。 |
| **AI Trader（ローカル定量アシスタント）** | HyperliquidのL1/L2/派生商品データと5m/15m/1hのマルチタイムフレーム分析に基づき、ローカルでチャートを描画しシミュレーション取引ができるAIアシスタントと対話。 |
| **SSD永続化プリフィックスキャッシュ** | 同一文書やシステムプロンプトの繰り返し分析が、サーバー再起動後でもミリ秒でウォームスタート。キャッシュは誰かのサーバーではなく、あなたのSSD上に存在する。 |
| **AI Trader Chat UI** | 1つのコマンド（`--chat`、ポート `5119`）で起動。セキュアな KMS 認証情報の保管庫を統合し、Hyperliquid の資産、ポジション、取引履歴をリアルタイムで監視。 |
| **サービスマネージャー** | `service.sh`がデーモン化、ヘルスチェック、ログローテーション、ポート管理、ゼロダウンタイム再起動を処理する。 |
| **API Key認証** | 環境変数経由でキーをローテーション。プロキシなしでエンタープライズグレードのアクセス制御。 |

---

## ⚡ 技術的優位性

### 1. Thinking-Aware Constrained Generation（思考認識型制約生成）

多くの推論モデルは、`<think>...</think>`タグ内で思考の連鎖を出力する。標準的な構造化出力エンジンは、思考中に破損したりJSONを破壊したりする。XMLX-VLMは、トークンレベルで**4段階のロジット状態マシン**（`IDLE → THINKING → TRANSITIONING → CONTENT`）を管理する：

- **THINKING** — モデルは自由に推論する。JSON制約なし。仮説、境界ケース、矛盾を探索できる。
- **TRANSITIONING** — バジェットが尽きると、**ロジットマスキングにより終了トークンシーケンスを強制する**（対象以外は`-inf`）。クリーンで決定論的な終了。
- **CONTENT** — 思考が閉じた瞬間、制御は内部JSON Schemaプロセッサーに移る。最初のコンテンツトークンはすでに制約されている。

結果：モデルは法律上の議論や医療上の鑑別診断について512トークン思考し、その後完全に有効な構造化JSONを出力できる——後処理ゼロ。

### 2. 自動プリフィックスキャッシュ＆SSD永続化（APC）

同一文書やシステムプロンプトを反復処理する際、XMLX-VLMはリクエスト間でKVキャッシュを再利用する。ハイブリッドSSM/Attentionモデル（Qwen3.5 DeltaNet、Nemotron-H）の場合、**再帰状態もスナップショット化されSSDに永続化される**：

- ブロックレベルのKVキャッシュ＆連鎖ハッシュ
- LRU＋参照カウントによる退避
- `APC_DISK_PATH`が完全なブロックをシャーディングされたSSDファイルに書き込む——**プロセス再起動後も存続**
- 同一プロンプトがミリ秒でウォームスタート、サーバー再起動後も

### 3. マルチフォーマット推論パーサー＋ツール呼び出し昇格

6つのストリーミングパーサーが、Qwen3、DeepSeek-R1、Gemma4、GLM4、GPT-OSS、Harmonyの推論抽出を処理する。思考段階内に`<tool_call>`ブロックが出現した場合、**自動的にコンテンツストリームに昇格**される——モデルが「ツールを呼び出すことを考え」、実際に呼び出せるようになる。

### 4. ツール呼び出し自動復旧＋Jump-Forwardデコーディング

量子化モデルは複数回のツールラウンド後に劣化する。XMLX-VLMは2つの防御機構を追加する：

- **自動復旧** — 閉じられていないXMLタグを修復し、途切れたJSONの中括弧をバランスさせ、破損した出力から裸のJSONオブジェクトを抽出する。
- **Jump-Forward Logits Bias**（`--enable-tool-logits-bias`） — ツール関連トークンIDへの加算的バイアスが、モデルを構造化フォーマットにより早く押し込み、ツール初トークンまでの時間を短縮する。

### 5. スケールでの推測的デコーディング

- **DFlash** — 超軽量ドラフトモデルが2～3トークン先を予測
- **MTP**（Multi-Token Prediction） — 高エントロピープロンプト用の並列ドラフトパス

長文書分析と推論タスクのレイテンシを削減する。

### 6. KVキャッシュ量子化

- **Uniform**（4-bit、3.5-bit、8-bit）
- **TurboQuant** — 重要な場所でAttention精度を維持する適応型スキーム

128 GBのMac Studioで70BクラスのVisionモデルを、長コンテキスト文書用の余裕を持って実行できる。

### 7. MoE Top-Kオーバーライド

動的なtop-kオーバーライドにより、インタラクティブ分析セッションでわずかな精度をトレードしてレイテンシの改善を実現できる。

### 8. Apple Siliconネイティブ最適化

- `mx.fast.scaled_dot_product_attention`によるFlash Attention
- Visionエンコーダー向けMetalカーネル融合
- ハードウェア対応メモリ予算（M1 → M5 Maxプロファイルを内蔵）
- CPU前処理とGPU推論間のユニファイドメモリゼロコピー

---

## 🏗 アーキテクチャ概要

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
│   AI Trader — ローカル定量アシスタント)                     │
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

## 🚦 クイックスタート

### ワンクリックインストール（macOS Apple Silicon）

新しいMacや開発ツールが未インストールのユーザー向け：

```bash
curl -fsSL https://raw.githubusercontent.com/123456btc/xmlx_vlm/master/install.sh | bash
```

このスクリプトは自動的に以下を実行する：
- ✅ Apple Silicon（M1/M2/M3/M4/M5）を確認
- ✅ Xcode Command Line Toolsをインストール（未インストールの場合）
- ✅ Homebrewをインストール（未インストールの場合）
- ✅ Python 3.12をインストール（3.10未満の場合）
- ✅ `uv`（高速Pythonパッケージマネージャー）をインストール
- ✅ リポジトリをクローンし、仮想環境を作成
- ✅ MLX、XMLX-VLM、およびすべての依存関係をインストール
- ✅ デフォルトAPIキー（`x123456`）と環境変数を設定
- ✅ 必要に応じてデフォルトモデルを事前ダウンロード（約20GB）
- ✅ サーバーを起動

**想定時間：** 新しいMacで10〜20分（主にモデルのダウンロード）。

### 手動インストール

手動セットアップを希望する場合：

```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### サーバーの起動

```bash
# デフォルト起動 — サーバーのみ（ヘッドレス、Chat UIなし）
./service.sh start

# Chat UIも併用して起動
./service.sh start --chat

# デフォルトAPIキーとKV量子化を上書きして本番ワークロード用に起動
XMLX_VLM_API_KEY=mykey ./service.sh start --kv-bits 3.5 --kv-quant-scheme turboquant

# MCP重視のワークフローでツール呼び出し加速を有効化
./service.sh start --enable-tool-logits-bias

# 推測的デコーディングを完全に無効化（標準生成にフォールバック）
XMLX_VLM_DRAFT_MODEL="" XMLX_VLM_DRAFT_KIND="" ./service.sh start
```

### APIの呼び出し（ローカルのみ）

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

### AI Traderの起動（ローカル定量アシスタント）

```bash
# まずサーバーを起動
./service.sh start

# ローカル取引アシスタントと対話
xmlx_vlm.ai-trader

# または単一プロンプトを実行
xmlx_vlm.ai-trader --prompt "BTCの動向を分析して"
```

AI TraderはHyperliquidのデータソースを統一し、5m/15m/1hのマルチタイムフレーム分析、L2 オーダーブック深度、取引フロー、資金調達率、未決済建玉をサポート。ローカルの意思決定を最適化するため、**単一リクエストでの Bull/Bear 対抗討論（Adversarial Debate）**によりバイアスを排除し、ポジションクローズ時の**自動ポストトレード反省（Reflection）**をローカル SQLite に保存して自律的な学習閉環を実現します。すべてローカルで完結する。

---

## 🤖 エージェントクライアント統合

XMLX-VLMは、**コーディングエージェントやAIアシスタントのローカルバックエンド**として設計されている。エージェントクライアントは毎ターン完全な会話履歴を再送信するため、APCディスク永続化（`APC_DISK_PATH`）の有効化を強く推奨する——最初の高コストなウォームアップ後、繰り返されるprefillオーバーヘッドを排除する。

### Claude Code（Anthropic互換）

`~/.local/bin/claude-xmlx`を作成：

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

ディスクKVキャッシュ付きでサーバーを起動：

```bash
APC_ENABLED=1 APC_DISK_PATH=/tmp/xmlx-apc ./service.sh start
```

### Cline / Continue.dev（OpenAI互換）

VS Code設定または`~/.continue/config.json`で：

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

### Aider（OpenAI互換）

```bash
export OPENAI_API_BASE=http://localhost:5118/v1
export OPENAI_API_KEY=x123456
aider --model openai/mlx-community/diffusiongemma-26B-A4B-it-4bit
```

### Cursor（OpenAI互換）

Cursor Settings → Models → Add Modelで：
- **Base URL**: `http://localhost:5118/v1`
- **API Key**: `x123456`
- **Model**: `mlx-community/diffusiongemma-26B-A4B-it-4bit`

### Pi（pi.dev）

Piは、XMLX-VLMと相性の良いローカルファーストのコーディングエージェントである。`~/.pi/agent/models.json`にプロバイダーを追加：

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

次に、`~/.pi/agent/settings.json`でデフォルトとして設定：

```json
{
  "defaultProvider": "xmlx-local",
  "defaultModel": "mlx-community/Qwen3.8-27B-4bit"
}
```

### 推奨エージェントサーバーフラグ

```bash
# 完全なエージェントスタック：ディスクAPC＋ツール加速＋長コンテキスト
APC_ENABLED=1 \
APC_DISK_PATH=/tmp/xmlx-apc \
XMLX_VLM_ENABLE_TOOL_LOGITS_BIAS=1 \
./service.sh start --ctx 100000
```

> **ヒント**：エージェントクライアントは、初期システムプロンプトとして10k〜30kトークンを送信することが多い。`APC_DISK_PATH`を使用すると、このプリフィックスは最初のprefill中にSSDに書き込まれ、以降のセッションで即座に復元される——サーバー再起動後でも。

---

## 🛠 運用＆可観測性

```bash
# ヘルス、読み込みモデル、PID、ポートを確認
./service.sh status

# ライブログを追尾
./service.sh logs server
./service.sh logs chat

# ゼロダウンタイム再起動
./service.sh restart
```

- **PID追跡** — 孤児プロセスのフォールバック付き
- **ポート衝突** — 自動解決
- **ヘルスエンドポイント** — `/health`
- **構造化ログ** — ローテーション対応の出力

---

## 🧩 サポートモデル

- **Qwen-VL / Qwen2-VL / Qwen3.6-VL**（CJK文書に推奨）
- **LLaVA 1.5 / 1.6 / NeXT**
- **Phi-3 / Phi-4 Vision**
- **InternVL2**
- **MiniCPM-V**
- **DeepSeek-VL**
- …および、MLXコミュニティポートを持つすべてのHugging Faceモデル。

---

## 🏛 謝辞＆系譜

XMLX-VLMは、いくつかの優れたオープンソースプロジェクトを意識的に基盤とする**ハードフォーク**である：

| プロジェクト | 借用したもの | 追加したもの |
|---------|-----------------|---------------|
| [**Blaizzy/mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) | コアVLMモデル読み込み、重み変換、MLX生成プリミティブ | プロダクションサーバー、推測的デコーディング、構造化出力、ツール呼び出し、MCP、Embedding/Rerankエンジン |
| [**vllm-mlx**](https://github.com/vllm-project/vllm)（コミュニティパターン） | メトリクス設計、モデルレジストリパターン、ハードウェア検出コンセプト | SSD永続化APCキャッシュ、Apple Silicon特化メモリ予算、統一CLI |
| [**Rapid-MLX**](https://github.com/raullenchai/Rapid-MLX) | ツール呼び出し自動復旧、jump-forwardロジットバイアス、DeltaNet状態スナップショット | 自動復旧とjump-forwardデコーディングの適用。ハイブリッドキャッシュアーキテクチャロードマップに着想 |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp) | 混合量子化述語（Q4_K_Mスタイル戦略） | MLX変換パイプラインへの統合 |
| [**Hugging Face Transformers**](https://github.com/huggingface/transformers) | トークナイザーユーティリティ、サンプリングロジック、AutoModel読み込み | MLXネイティブ重み変換、バッチストリーミング、thinking-awareプロセッサー |

これらのプロジェクトの作者とコミュニティに深く感謝している。XMLX-VLMは、彼らが基盤を築いてくれたからこそ存在する。

---

## 🤝 コミュニティ＆ロードマップ

- [x] デュアルプロトコルREST API（OpenAI＋Anthropic/Claude）
- [x] 推測的デコーディング（DFlash＋MTP）
- [x] KVキャッシュ量子化
- [x] ツール呼び出し＆MCP
- [x] Embedding＆Rerankエンジン
- [x] SSD永続化による自動プリフィックスキャッシュ
- [x] Thinking-aware構造化生成
- [x] ツール呼び出し自動復旧＋Jump-Forwardデコーディング
- [x] LoRA学習＆アダプター読み込み
- [~] LoRAホットスワップサービング（学習と読み込みは動作。ランタイムアダプター切り替えにはサーバーAPIが必要）
- [~] テンソル/パイプライン並列化（ユーティリティ層は`mx.distributed`をサポート。サーバー統合は保留中）
- [x] 組み込みベンチマーク（TTFT / TPOT / TPS / メモリ）
- [ ] クロスエンジンベンチマークスイート（コントリビューション歓迎！）

**ライセンス：** MIT  
**起源：** [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)からのハードフォーク——プロダクションワークロード向けに再構築。

---

<p align="center">
  <strong>あなたのデータ。あなたのモデル。あなたのプライバシー。</strong><br>
  XMLX-VLMがあなたの機密パイプラインを守るなら、Star ⭐ をお願いします。
</p>
