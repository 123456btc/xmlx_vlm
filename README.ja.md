# XMLX-VLM: ローカル完全プライベート AI クオンツ取引 OS & ターミナル

<p align="center">
  <strong>世界初 Apple Silicon ネイティブ · 完全プライベート自律型 AI 量化取引 OS</strong>
</p>

<p align="center">
  <em>データ流出ゼロ · Token 費用ゼロ · マイクロ秒単位のインメモリ市場基盤 · 4役マルチエージェントと機関投資家級リスク管理 —— 100% お手元の Mac で動作。</em>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="https://github.com/123456btc/xmlx_vlm/releases/tag/v1.0.0"><img alt="Release: v1.0.0" src="https://img.shields.io/badge/Release-v1.0.0%20Latest-brightgreen"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform: macOS" src="https://img.shields.io/badge/プラットフォーム-macOS%20(Apple%20Silicon%20M1--M5)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> | <a href="README.zh.md">🇨🇳 中文</a> | <strong>🇯🇵 日本語</strong> | <a href="README.ko.md">🇰🇷 한국어</a>
</p>

---

## ⚡ なぜローカルプライベート AI 取引 OS なのか？

既存のクラウド型 AI 取引ボット（Cloud-based AI AutoTraders）の多くは、**クラウド LLM API**（OpenAI、Anthropic、DeepSeek など）に依存しています。実運用のクオンツ取引において、クラウド依存は致命的なリスクを伴います：

| 重要項目 | 従来のクラウド AI ボット (Cloud AI Trading Bots) | **XMLX-VLM ローカル AI 取引 OS** |
| :--- | :--- | :--- |
| **🛡️ 戦略と秘密鍵の機密性** | API キー、署名、注文の意図、独自のアルファ戦略がクラウドプロバイダーや中間者攻撃に晒されます。 | **100% エアギャップ & データ主権**<br>モデル推論と実行はすべて Apple Silicon 上で完結。秘密鍵はローカル KMS Vault に保存され、外部送信は一切ありません。 |
| **💰 24時間365日の運用コスト** | 常時監視と高頻度ポーリングにより、月額 **300〜3,000ドル以上** の Token 費用が発生します。 | **Token 費用 0 円（無料）**<br>無制限のローカルハードウェア推論。API コストを気にせず 30+ 銘柄を 24 時間監視可能。 |
| **⏱️ レイテンシとレート制限** | 急激な価格変動時にクラウドの `429 Rate Limit` やタイムアウト（500ms〜数秒）が発生し、ロスカット失敗につながります。 | **マイクロ秒級のインメモリ実行**<br>MLX 最適化による Continuous Batching により、外部ネットワーク制限ゼロで即時応答。 |
| **🏛️ 実行ガバナンスと規律** | 単一モデルの単純ループは無限リトライや感情的な過剰取引（Overtrading）に陥りがちです。 | **モデルが提案し、ローカル Runtime が統制**<br>純粋関数型ガードレール、再エントリークールダウン（30分）、時間単位の注文上限、4役マルチエージェント協調。 |

---

## 🚀 XMLX-VLM の 5 大コア機能

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            XMLX-VLM 取引 OS アーキテクチャ                               │
│                                                                                          │
│  [ Hyperliquid 常時接続 WebSocket ストリーム ]                                          │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. インメモリ・カラム型市場エンジン (In-Memory Columnar Engine)                    │  │
│  │    • 出来高上位 30 銘柄のリアルタイム自動購読 & RAM 状態マシン                      │  │
│  │    • マイクロ秒指標計算: 板情報不均衡比率、マルチ周期 CVD、Volume Profile、ATR/ADX   │  │
│  │    • Point-in-Time (`as_of`) タイムトラベル: バックテストにおける未来情報の厳格遮断 │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │ (高優先度アラート / レイテンシゼロのメモリ内スナップショット)            │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 2. 4役マルチエージェント自律カンバン艦隊 (Kanban Fleet)                            │  │
│  │    [ Scout (市場異常検知) ] ──▶ [ Analyst (マルチタイムフレーム分析) ]             │  │
│  │                                       │                                            │  │
│  │    [ Executor (執行約定) ] ◀── [ Risk Officer (証拠金 & 承認ゲート) ]              │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. エンタープライズ級 Agent コア & 実行ガードレール (Guardrails)                   │  │
│  │    • ThinkScrubber: 思考過程 `<think>` の分離と構造化 JSON 抽出                    │  │
│  │    • ToolCallGuardrails: 無限ループ遮断、重複注文ブロック、停滞状態防止            │  │
│  │    • 過剰取引防止スロットル: 再エントリー 30 分クールダウン & 1 時間あたり注文上限 │  │
│  │    • ContextCompressor: 指令乗っ取り防止 Token 圧縮（緊急決済指示の最優先保証）   │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 4. Apple Silicon MLX ネイティブ推論アクセラレーション                             │  │
│  │    • TurboQuant 3.5b/4b 混合量子化 & Continuous Batching                           │  │
│  │    • RAM + SSD 階層型永続プレフィックスキャッシュ (APC)                            │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. インメモリ・カラム型市場データ基盤
- **WebSocket 常時接続**: `wss://api.hyperliquid.xyz/ws` に直結し、24時間出来高 Top 30 の無期限先物を追跡。
- **ゼロ待機インメモリ計算**: 板情報インバランス、累積ボリュームデルタ（CVD）、Volume Profile（POC/VAH/VAL）、ATR/ADX をメモリ内で直接算出。
- **Point-in-Time タイムトラベル (`as_of`)**: 過去の任意時点のスナップショットを正確に復元し、バックテスト時のルックアヘッドバイアス（未来関数）を完全排除。

### 2. モデルが提案し、ローカル Runtime が厳格に統制
- **ループ遮断 & 注文保護**: 同一パラメータの失敗を検知して遮断し、暴走注文を防止。
- **過剰取引防止（Anti-Overtrading）**: 1日2〜4回の上質な取引、45〜90分の保有目標をプロンプトに組み込み、決済後 **30分間の再エントリー禁止クールダウン** を強制。
- **指令乗っ取り防止コンテキスト圧縮**: 長期会話を要約する際も `SUMMARY_PREFIX` により「緊急全決済」「即時停止」の最新指示を常に最優先実行。

### 3. 4役の自律型マルチエージェント艦隊
- **Scout（索敵員）**: ボラティリティ急拡大、大口約定クラスター、資金調達率の反転を常時監視。
- **Analyst（アナリスト）**: 複数時間軸のテクニカル分析とエントリー/決済ラインを設計。
- **Risk Officer（リスク管理官）**: 証拠金維持率（< 50%）や許容ドローダウンを厳格に審査。
- **Executor（執行員）**: ローカル OMS を通じて模擬または実取引の署名注文を発注。

### 4. Apple Silicon ネイティブ MLX アクセラレーション
- **Apple M1 / M2 / M3 / M4 / M5** 各種チップ（Pro/Max/Ultra）に完全対応。
- **Continuous Batching**: 複数エージェントの並行推論でもレイテンシが崩壊しません。
- **階層型 APC キャッシュ**: プロンプトを SSD に永続キャッシュし、数ミリ秒でウォームスタート。

### 5. ネイティブ Web 取引ターミナル UI
- ワンクリックで起動できるダークテーマの取引画面：`http://localhost:5119`。
- **ローカル KMS 暗号化**: 取引所 API キーを AES-256-GCM でローカル保護。
- **インタラクティブ承認ゲート**: 実注文前に画面上でワンクリック承認、または全自動パイロットモードの選択が可能。

---

## ⚡ クイックスタート（30 秒）

### 1. インストール
```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
pip install -e .
```

### 2. ローカル AI 取引 OS & ターミナルの起動
```bash
# 推論エンジンと Web ターミナルを一括起動
./service.sh start
```
- 🧠 **OpenAI / Anthropic デュアル API**: `http://localhost:5118`
- 🖥️ **AI Trader Web 取引ターミナル**: `http://localhost:5119`

### 3. 停止 & ステータス確認
```bash
./service.sh status
./service.sh stop
```

---

## 🧪 自動テスト検証

```bash
PYTHONPATH=. pytest tests/test_agent_core.py \
                     tests/test_skills_curator.py \
                     tests/test_kanban_board.py \
                     tests/test_ai_trader_agent_core.py \
                     tests/test_columnar_market_store.py \
                     tests/test_throttle_guardrails.py -v
```
> **テスト結果**: `35 / 35 Passed (100% 成功)` ✅

---

## 📄 ライセンス

本プロジェクトは [MIT License](LICENSE) のもとで公開されています。
