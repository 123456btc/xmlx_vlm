# XMLX-VLM

<p align="center">
  <strong>프라이버시 중심의 로컬 비전-언어 AI</strong>
</p>

<p align="center">
  <em>Apple Silicon 네이티브 추론 엔진. 데이터가 기기를 벗어나지 않습니다. 클라우드 API 노출은 제로입니다.</em>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform: macOS" src="https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> | <a href="README.zh.md">🇨🇳 中文</a> | <a href="README.ja.md">🇯🇵 日本語</a> | <strong>🇰🇷 한국어</strong>
</p>

---

## 🚀 왜 XMLX-VLM인가?

**민감한 데이터를 다루는 전문가에게 프라이버시는 기능이 아니라 기본입니다.**

법률 문서, 의료 기록, 정부 파일, 독점 연구, 트레이딩 알고리즘 — 이것들이 클라우드 API를 거치는 순간, 통제권을 잃게 됩니다. 훈련 데이터가 되고, 로그에 남고, 소환장에 회부됩니다.

**XMLX-VLM**은 **로컬 우선, 프로덕션급 비전-언어 추론 엔진**으로, Apple Silicon에서 완전히 실행됩니다. 문서를 읽고, 이미지를 파싱하며, 복잡한 문제를 추론하고 구조화된 출력을 생성합니다 — **단 하나의 네트워크 호출 없이**.

클우드 구독도, 데이터 보존 정책도, 제3자 서비스 약관도 없습니다. 오직 당신의 Mac, 당신의 데이터, 그리고 당신의 모델뿐입니다.

> **데이터 주권이 아키텍처의 기반입니다. 나머지는 그 위에 세워집니다.**

### 🧬 AFRE 생태계

XMLX-VLM은 **AFRE (AI Factor Research Engine)** 생태계의 **프라이빗 AI 브레인**입니다 — DDD, 헥사고날 아키텍처, 클린 아키텍처 기반의 도메인 중심, 에이전트 강화 퀀트 리서치 플랫폼입니다.

AFRE는 **팩터의 계보(genealogy)**를 연구합니다: 왜 발명되었는지, 어떻게 확산되었는지, 왜 쇠퇴했는지, 그리고 어떤 깨진 가정에서 현대적 가설이 나올 수 있는지. XMLX-VLM은 AFRE의 에이전트 런타임을 다음과 같이 지원합니다:

| AFRE 기능 | XMLX-VLM이 로컬에서 가능하게 하는 것 |
|-----------------|--------------------------------|
| **팩터 계보 인텔리전스** | 연구 PDF와 차트 이미지를 파싱하고, 시각적 문서에서 구조화된 팩터 역사를 추출 |
| **발명자 사고 시뮬레이션** | 깊은 추론(`<think>` 모드)을 통해 팩터 창시자의 제약, 인센티브, 지식 스택을 시뮬레이션 |
| **가설 중심 연구** | JSON-Schema 제약 출력으로 생성된 모든 팩터 변형에 검증 가능한 가설과 깨진 가정 추적을 포함 |
| **재현 가능한 실험** | 도구 호출 + MCP로 로컬 백테스터와 신호 생성기에 연결; 실험은 사용자 하드웨어에서 실행되며 설계상 감사 가능 |
| **과적합 방지 거버넌스** | 구조화된 출력이 워크포워드 파라미터, 체제 분할, 회전 패널티를 머신 리더블 스키마로 강제 |
| **지식 진화** | 임베딩 + 리랭크로 검증된 결과를 거버넌스된 로컬 쿼리 가능 지식 베이스에 색인 |
| **멀티 에이전트 병렬 연구** | 연속 배칭 + 추측 디코딩으로 여러 AI 워커가 지연 붕괴 없이 독립적으로 추론 |

**AFRE는 방법론입니다. XMLX-VLM은 이를 가능하게 하는 프라이빗 추론 계층입니다.**

AFRE는 XMLX-VLM의 퀀트 금융 분야 대표 구현 사례지만, 동일한 로컬 프라이버시 아키텍처는 법률, 헬스케어, 정부, 기업 R&D 분야에서도 동등하게 활용됩니다.

---

## 🎯 누구를 위한 것인가?

| 도메인 | 민감한 데이터 | XMLX-VLM이 로컬에서 하는 일 |
|--------|---------------|---------------------------|
| **퀀트 금융** | 독점 팩터, 내부 연구 보고서, 알파 신호 | PDF 보고서와 차트 이미지 파싱; 팩터 가설 추론; 구조화된 팩터 정의 출력; MCP를 통한 로컬 백테스터 호출; AI Trader로 Hyperliquid 시장 데이터를 로컬에서 분석·거래 |
| **법률** | 사건 파일, 계약서, 디스커버리 문서, 고객 통신 | 문서 이미지와 스캔 분석; 구조화된 조항 추출; 법적 논증 추론; 레드라인 요약 생성 |
| **정부** | 기밀 브리핑, 정책 초안, 시민 기록, 인텔리전스 영상 | 민감한 영상과 스캔 문서 처리; 인텔리전스 보고서용 구조화된 출력; 로컬 하드웨어에서 완전한 감사 추적 |
| **헬스케어** | 환자 기록, 의료 영상, 임상 노트, 검사 결과 | 의료 문서 이미지 파싱; 감별 진단 추론; 임상 요약용 구조화된 출력; 아키텍처상 HIPAA 준수 |
| **기업 R&D** | 영업 비밀, 특허 초안, 실험 데이터, 내부 메모 | 기술 도면의 비전-언어 이해; 연구 가설 추론; 실험 설계용 구조화된 출력 |

---

## 🎯 핵심 기능

| 기능 | 제공되는 것 |
|------------|-------------|
| **로컬 문서 인텔리전스** | PDF, 스캔 문서, 스크린샷, 이미지 중심 보고서를 모델에 직접 입력. OCR SaaS 불필요. 클라우드 비전 API 불필요. 문서가 localhost를 벗어나지 않습니다. |
| **추론이 포함된 구조화된 출력** | 깊은 추론을 위해 `thinking` 모드를 활성화한 후 최종 출력에 JSON-Schema 제약을 적용. 감사 준비 보고서, 팩터 정의, 임상 요약에 적합합니다. |
| **듀얼 프로토콜 API** | 하나의 서버가 OpenAI(`/v1/chat/completions`)와 Anthropic(`/v1/messages`) 프로토콜을 모두 지원. Cursor, Claude Code, LangChain, PydanticAI의 백엔드로 바로 연결 — 모든 트래픽이 `localhost:8080`에 머무릅니다. |
| **로컬 도구 호출 및 MCP** | 로컬 데이터베이스, 백테스터, EHR 시스템, 사건 관리 도구, 문서 파이프라인을 MCP로 연결. 모델이 도구를 호출하고, 데이터는 기기를 벗어나지 않습니다. |
| **프라이빗 지식을 위한 임베딩 및 리랭크** | 내부 문서, 연구 노트, 사건 파일, 환자 이력을 색인. 독점 지식 베이스 위의 시맨틱 검색 — 클라우드 노출 제로. |
| **AI Trader（로컬 퀀트 어시스턴트）** | Hyperliquid의 L1/L2/파생상품 데이터와 5m/15m/1h 멀티 타임프레임 분석을 기반으로 로컬에서 차트를 렌더링하고 모의 거래할 수 있는 AI 어시스턴트와 대화. |
| **SSD 영구 프리픽스 캐시** | 동일한 문서나 시스템 프롬프트를 반복 분석할 때 밀리초 내로 웜 스타트, 서버 재시작 후에도. 캐시는 타인의 서버가 아닌 당신의 SSD에 존재합니다. |
| **Gradio Chat UI** | 한 번의 명령(`--chat`)으로 로컬 데모, 내부 검토 세션, 보안 내부 도구를 실행. |
| **서비스 매니저** | `service.sh`이 데몬화, 헬스 체크, 로그 로테이션, 포트 관리, 무중단 재시작을 처리. |
| **API 키 인증** | 환경 변수를 통해 키 로테이션. 프록시 없이도 엔터프라이즈급 접근 제어. |

---

## ⚡ 기술적 우위

### 1. Thinking-Aware 제약 생성

대부분의 추론 모델은 `<think>...</think>` 태그 안에 사고 사슬을 출력합니다. 표준 구조화된 출력 엔진은 사고 중에 깨지거나 JSON을 손상시킵니다. XMLX-VLM은 토큰 수준에서 **4단계 로짓 상태 머신**(`IDLE → THINKING → TRANSITIONING → CONTENT`)을 관리합니다:

- **THINKING** — 모델이 자유롭게 추론합니다. JSON 제약 없음. 가정, 엣지 케이스, 모순을 탐색할 수 있습니다.
- **TRANSITIONING** — 예산이 만료되면 **로짓 마스킹**을 통해 종료 토큰 시퀀스를 강제합니다(목표 외에는 `-inf`). 깨끗하고 결정적인 종료.
- **CONTENT** — 사고가 닫히는 즉시 제어가 내부 JSON-Schema 프로세서로 넘어갑니다. 첫 번째 콘텐츠 토큰은 이미 제약되어 있습니다.

결과: 모델이 법적 논증이나 의학적 감별 진단에 대해 512 토큰 동안 사고한 후, 완벽하게 유효한 구조화된 JSON을 출력할 수 있습니다 — 후처리 제로.

### 2. SSD 영속성을 가진 자동 프리픽스 캐싱(APC)

동일한 문서나 시스템 프롬프트를 반복할 때 XMLX-VLM은 요청 간 KV 캐시를 재사용합니다. 하이브리드 SSM/어텐션 모델(Qwen3.5 DeltaNet, Nemotron-H)의 경우 **순환 상태도 스냅샷되어 SSD에 영속화**됩니다:

- 체인 해싱이 적용된 블록 레벨 KV 캐시
- LRU + 참조 카운트 기반 퇴출
- `APC_DISK_PATH`가 전체 블록을 샤딩된 SSD 파일에 기록 — **프로세스 재시작 후에도 유지**
- 동일한 프롬프트가 서버 재시작 후에도 밀리초 내로 웜 스타트

### 3. 멀티 포맷 추론 파서 + 도구 호출 프로모션

6개의 스트리밍 파서가 Qwen3, DeepSeek-R1, Gemma4, GLM4, GPT-OSS, Harmony에 대한 추론 추출을 처리합니다. 사고 단계에 `<tool_call>` 블록이 나타나면 **자동으로 콘텐츠 스트림으로 프로모션**되어 모델이 "도구 호출을 생각"하고 실제로 호출할 수 있습니다.

### 4. 도구 호출 자동 복구 + 점프 포워드 디코딩

양자화된 모델은 여러 도구 라운드 후에 품질이 저하됩니다. XMLX-VLM은 두 가지 방어 메커니즘을 추가합니다:

- **자동 복구** — 닫히지 않은 XML 태그를 복구하고, 잘린 JSON 중괄호를 균형 맞추며, 엉망인 출력에서 베어 JSON 객체를 추출.
- **점프 포워드 로짓 바이어스**(`--enable-tool-logits-bias`) — 도구 관련 토큰 ID에 가산 바이어스를 적용하여 모델을 구조화된 형식으로 더 빠르게 밀어넣어 첫 도구 토큰까지의 시간을 단축.

### 5. 대규모 추측 디코딩

- **DFlash** — 초경량 드래프트 모델이 2~3 토큰 앞을 예측
- **MTP** (Multi-Token Prediction) — 높은 엔트로피 프롬프트를 위한 병렬 드래프트 경로

장문 문서 분석과 추론 작업에서 지연 시간을 단축합니다.

### 6. KV-Cache 양자화

- **Uniform** (4-bit, 3.5-bit, 8-bit)
- **TurboQuant** — 중요한 곳에서 어텐션 정밀도를 보존하는 적응형 방식

128GB Mac Studio에서 70B급 비전 모델을 장문 컨텍스트 문서를 위한 여유 공간과 함께 실행.

### 7. MoE Top-K 오버라이드

동적 top-k 오버라이드로 대화형 분석 세션에서 일부 정확도를 희생하여 지연 시간 이득을 얻을 수 있습니다.

### 8. Apple-Silicon 네이티브 최적화

- `mx.fast.scaled_dot_product_attention`을 통한 Flash Attention
- 비전 인코더를 위한 Metal 커널 퓨전
- 하드웨어 인식 메모리 버젯팅(M1 → M5 Max 프로파일 내장)
- CPU 전처리와 GPU 추론 간 통합 메모리 제로 카피

---

## 🏗 아키텍처 개요

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
│   AI Trader — 로컬 퀀트 어시스턴트)                          │
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

## 🚦 빠른 시작

### 원클릭 설치(macOS Apple Silicon)

개발 도구가 없는 신규 Mac 사용자를 위해:

```bash
curl -fsSL https://raw.githubusercontent.com/123456btc/xmlx_vlm/master/install.sh | bash
```

이 스크립트는 자동으로:
- ✅ Apple Silicon(M1/M2/M3/M4/M5) 확인
- ✅ Xcode Command Line Tools 설치(없는 경우)
- ✅ Homebrew 설치(없는 경우)
- ✅ Python 3.12 설치(3.10 미만인 경우)
- ✅ `uv` 설치(빠른 Python 패키지 매니저)
- ✅ 레포지토리 클론 및 가상 환경 생성
- ✅ MLX, XMLX-VLM 및 모든 의존성 설치
- ✅ 기본 API 키(`x123456`) 및 환경 변수 설정
- ✅ 선택적 기본 모델 사전 다운로드(~20GB)
- ✅ 서버 시작

**예상 시간:** 신규 Mac에서 10~20분(대부분 모델 다운로드).

### 수동 설치

수동 설정을 선호하는 경우:

```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 서버 시작

```bash
# 기본 시작 — 서버만(헤드리스, Chat UI 없음)
./service.sh start

# 서버와 함께 Chat UI 활성화
./service.sh start --chat

# 기본 API 키 재정의 + 프로덕션 워크로드용 KV 양자화
XMLX_VLM_API_KEY=mykey ./service.sh start --kv-bits 3.5 --kv-quant-scheme turboquant

# MCP 중심 워크플로우용 도구 호출 가속화 활성화
./service.sh start --enable-tool-logits-bias

# 추측 디코딩 완전 비활성화(표준 생성으로 폴백)
XMLX_VLM_DRAFT_MODEL="" XMLX_VLM_DRAFT_KIND="" ./service.sh start
```

### API 호출(로컬 전용)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
    "messages": [
      {"role": "user", "content": "Analyze the attached document and extract structured findings"}
    ],
    "stream": true
  }'
```

### AI Trader 실행（로컬 퀀트 어시스턴트）

```bash
# 먼저 서버 시작
./service.sh start

# 로컬 트레이딩 어시스턴트와 대화
xmlx_vlm.ai-trader

# 또는 단일 프롬프트 실행
xmlx_vlm.ai-trader --prompt "BTC 동향 분석"
```

AI Trader는 Hyperliquid 데이터 소스를 통합하여 5m/15m/1h 멀티 타임프레임 분석, L2 오더북 깊이, 거래 흐름, 자금 조달 비율 및 미결제 약정을 지원하며, 모두 로컬에서 완료됩니다.

---

## 🤖 에이전트 클라이언트 통합

XMLX-VLM은 **코딩 에이전트와 AI 어시스턴트를 위한 로컬 백엔드**로 설계되었습니다. 에이전트 클라이언트가 매 턴마다 전체 대화 기록을 재전송하므로, APC 디스크 영속성(`APC_DISK_PATH`) 활성화를 강력히 권장합니다 — 첫 번째 비싼 웜업 후 반복적인 프리필 오버헤드를 제거합니다.

### Claude Code(Anthropic 호환)

`~/.local/bin/claude-xmlx` 생성:

```bash
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${XMLX_ANTHROPIC_BASE_URL:-http://127.0.0.1:8080}"
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

디스크 KV 캐시로 서버 시작:

```bash
APC_ENABLED=1 APC_DISK_PATH=/tmp/xmlx-apc ./service.sh start
```

### Cline / Continue.dev(OpenAI 호환)

VS Code 설정 또는 `~/.continue/config.json`에서:

```json
{
  "models": [
    {
      "title": "XMLX-VLM Local",
      "provider": "openai",
      "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "x123456"
    }
  ]
}
```

### Aider(OpenAI 호환)

```bash
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=x123456
aider --model openai/mlx-community/diffusiongemma-26B-A4B-it-4bit
```

### Cursor(OpenAI 호환)

Cursor 설정 → 모델 → 모델 추가:
- **Base URL**: `http://localhost:8080/v1`
- **API Key**: `x123456`
- **Model**: `mlx-community/diffusiongemma-26B-A4B-it-4bit`

### Pi(pi.dev)

Pi는 XMLX-VLM과 잘 어울리는 로컬 우선 코딩 에이전트입니다. `~/.pi/agent/models.json`에 프로바이더를 추가:

```json
{
  "providers": {
    "xmlx-local": {
      "name": "XMLX-VLM (local)",
      "baseUrl": "http://localhost:8080/v1",
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

그런 다음 `~/.pi/agent/settings.json`에서 기본값으로 설정:

```json
{
  "defaultProvider": "xmlx-local",
  "defaultModel": "mlx-community/diffusiongemma-26B-A4B-it-4bit"
}
```

### 권장 에이전트 서버 플래그

```bash
# 전체 에이전트 스택: 디스크 APC + 도구 가속화 + 장문 컨텍스트
APC_ENABLED=1 \
APC_DISK_PATH=/tmp/xmlx-apc \
XMLX_VLM_ENABLE_TOOL_LOGITS_BIAS=1 \
./service.sh start --ctx 100000
```

> **팁**: 에이전트 클라이언트는 초기 시스템 프롬프트로 종종 10k~30k 토큰을 전송합니다. `APC_DISK_PATH`가 있으면 이 프리픽스가 첫 프리필 중 SSD에 기록되고 후속 세션에서 즉시 복원됩니다 — 서버 재시작 후에도.

---

## 🛠 운영 및 가시성

```bash
# 헬스, 로드된 모델, PID, 포트 확인
./service.sh status

# 실시간 로그 테일
./service.sh logs server
./service.sh logs chat

# 무중단 재시작
./service.sh restart
```

- **PID 추적** 및 고아 프로세스 폴백
- **포트 충돌** 자동 해결
- `/health`의 **헬스 엔드포인트**
- 로테이션 친화적 출력의 **구조화된 로그**

---

## 🧩 지원 모델

- **Qwen-VL / Qwen2-VL / Qwen3.6-VL** (CJK 문서에 권장)
- **LLaVA 1.5 / 1.6 / NeXT**
- **Phi-3 / Phi-4 Vision**
- **InternVL2**
- **MiniCPM-V**
- **DeepSeek-VL**
- ... 및 MLX 커뮤니티 포트가 있는 모든 Hugging Face 모델.

---

## 🏛 감사의 말 및 계보

XMLX-VLM은 여러 우수한 오픈소스 프로젝트를 의식적으로 기반으로 하는 **하드 포크**입니다:

| 프로젝트 | 차용한 것 | 추가한 것 |
|---------|-------------|---------------|
| [**Blaizzy/mlx-vlm**](https://github.com/Blaizzy/mlx-vlm) | 핵심 VLM 모델 로딩, 가중치 변환, MLX 생성 기본 요소 | 프로덕션 서버, 추측 디코딩, 구조화된 출력, 도구 호출, MCP, 임베딩/리랭크 엔진 |
| [**vllm-mlx**](https://github.com/vllm-project/vllm) (커뮤니티 패턴) | 메트릭스 설계, 모델 레지스트리 패턴, 하드웨어 감지 개념 | SSD 영속 APC 캐시, Apple-Silicon 특화 메모리 버젯팅, 통합 CLI |
| [**Rapid-MLX**](https://github.com/raullenchai/Rapid-MLX) | 도구 호출 자동 복구, 점프 포워드 로짓 바이어스, DeltaNet 상태 스냅샷 | 자동 복구 및 점프 포워드 디코딩 적용; 하이브리드 캐시 아키텍처 로드맵에 영감 |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp) | 혼합 양자화 조건(Q4_K_M 스타일 전략) | MLX 변환 파이프라인에 통합 |
| [**Hugging Face Transformers**](https://github.com/huggingface/transformers) | 토크나이저 유틸리티, 샘플링 로직, AutoModel 로딩 | MLX 네이티브 가중치 변환, 배치 스트리밍, thinking-aware 프로세서 |

이러한 프로젝트의 저자와 커뮤니티에게 깊은 감사를 드립니다. XMLX-VLM은 그들이 토대를 다졌기에 존재합니다.

---

## 🤝 커뮤니티 및 로드맵

- [x] 듀얼 프로토콜 REST API (OpenAI + Anthropic/Claude)
- [x] 추측 디코딩 (DFlash + MTP)
- [x] KV-cache 양자화
- [x] 도구 호출 및 MCP
- [x] 임베딩 및 리랭크 엔진
- [x] SSD 영속성을 가진 자동 프리픽스 캐싱
- [x] Thinking-aware 구조화된 생성
- [x] 도구 호출 자동 복구 + 점프 포워드 디코딩
- [x] LoRA 훈련 및 어댑터 로딩
- [~] LoRA 핫 스왑 서빙 (훈련 및 로딩은 작동; 런타임 어댑터 전환은 서버 API 필요)
- [~] 텐서 / 파이프라인 병렬화 (유틸리티 레이어가 `mx.distributed` 지원; 서버 통합 대기 중)
- [x] 내장 벤치마크 (TTFT / TPOT / TPS / 메모리)
- [ ] 크로스 엔진 벤치마크 스위트 (기여 환영!)

**라이선스:** MIT  
**기원:** [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)에서 하드 포크 — 프로덕션 워크로드를 위해 재구축.

---

<p align="center">
  <strong>당신의 데이터. 당신의 모델. 당신의 프라이버시.</strong><br>
  XMLX-VLM이 민감한 파이프라인을 보호한다면 ⭐를 눌러주세요.
</p>
