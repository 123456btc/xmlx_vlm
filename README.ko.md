# XMLX-VLM: 로컬 완전 프라이빗 AI 퀀트 트레이딩 OS & 터미널

<p align="center">
  <strong>세계 최초 Apple Silicon 네이티브 · 100% 프라이빗 자율형 AI 퀀트 트레이딩 OS</strong>
</p>

<p align="center">
  <em>데이터 유출 제로 · Token 비용 0원 · 마이크로초 인메모리 시장 인프라 · 4대 멀티 에이전트와 기관급 리스크 가드레일 —— 100% Mac 로컬 구동.</em>
</p>

<p align="center">
  <a href="https://github.com/123456btc/xmlx_vlm"><img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-xmlx__vlm-blue?logo=github"></a>
  <a href="https://github.com/123456btc/xmlx_vlm/releases/tag/v1.0.0"><img alt="Release: v1.0.0" src="https://img.shields.io/badge/Release-v1.0.0%20Latest-brightgreen"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Platform: macOS" src="https://img.shields.io/badge/플랫폼-macOS%20(Apple%20Silicon%20M1--M5)-silver?logo=apple"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> | <a href="README.zh.md">🇨🇳 中文</a> | <a href="README.ja.md">🇯🇵 日本語</a> | <strong>🇰🇷 한국어</strong>
</p>

---

## ⚡ 왜 로컬 프라이빗 AI 트레이딩 OS인가?

기존의 클라우드 기반 AI 트레이딩 봇(Cloud-based AI AutoTraders)의 대부분은 **클라우드 LLM API**(OpenAI, Anthropic, DeepSeek 등)에 전적으로 의존합니다. 실제 퀀트 실거래 환경에서 클라우드 의존은 치명적인 위험을 초래합니다:

| 핵심 비교 | 기존 클라우드 AI 봇 (Cloud AI Trading Bots) | **XMLX-VLM 로컬 AI 트레이딩 OS** |
| :--- | :--- | :--- |
| **🛡️ 전략 및 개인키 보안** | API 키, 비밀 서명, 주문 의도 및 자체 알파 전략이 클라우드 제공업체와 네트워크 중간자 공격에 노출됩니다. | **100% 에어갭 & 데이터 주권**<br>모델 추론과 주문 실행이 100% Apple Silicon에서만 구동됩니다. 키는 로컬 KMS에 암호화 보관되며 외부 전송이 전혀 없습니다. |
| **💰 24/7 상시 운영 비용** | 24시간 실시간 모니터링 시 매월 **300달러 ~ 3,000달러 이상**의 막대한 Token 요금이 청구됩니다. | **Token 비용 0원 (완전 무료)**<br>무제한 로컬 하드웨어 추론. API 비용 걱정 없이 30개 이상의 종목을 24시간 감시할 수 있습니다. |
| **⏱️ 레이턴시 및 호출 제한** | 급격한 변동성 발생 시 클라우드 `429 Rate Limit` 또는 수 초의 지연이 발생하여 적시 손절에 실패하고 청산될 위험이 큽니다. | **마이크로초급 인메모리 실행**<br>MLX 기반 Continuous Batching을 통해 외부 네트워크 지연과 호출 제한 없이 즉각 응답합니다. |
| **🏛️ 실행 거버넌스 및 규율** | 단일 모델 단순 루프는 무한 재시도나 감정적인 과잉 거래(Overtrading)에 쉽게 빠집니다. | **모델은 제안하고, 로컬 Runtime이 강제 통제**<br>순수 함수형 가드레일, 재진입 쿨다운(30분), 시간당 주문 상한, 4역 멀티 에이전트 협업. |

---

## 🚀 XMLX-VLM 5대 핵심 축

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            XMLX-VLM 트레이딩 OS 아키텍처                                 │
│                                                                                          │
│  [ Hyperliquid 상시 연결 WebSocket 스트림 ]                                             │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 인메모리 컬럼형 시장 인프라 (In-Memory Columnar Engine)                         │  │
│  │    • 24시간 거래대금 상위 30개 무기한 선물 실시간 자동 구독 및 RAM 상태 머신        │  │
│  │    • 마이크로초 지표 연산: 호가창 불균형, 멀티 윈도우 CVD, Volume Profile, ATR/ADX │  │
│  │    • Point-in-Time (`as_of`) 시간 여행: 백테스팅 시 미래 정보 유출 완벽 차단       │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │ (고우선순위 알람 / 레이턴시 제로 메모리 스냅샷)                          │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 2. 4역 멀티 에이전트 자율 칸반 플릿 (Kanban Fleet)                                 │  │
│  │    [ Scout (이상 탐지) ] ──▶ [ Analyst (멀티 타임프레임 분석) ]                    │  │
│  │                                       │                                            │  │
│  │    [ Executor (체결 실행) ] ◀── [ Risk Officer (증거금 및 승인 게이트) ]           │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. 엔터프라이즈급 Agent 코어 및 실행 가드레일 (Guardrails)                         │  │
│  │    • ThinkScrubber: 사고 과정 `<think>` 분리 및 정형화된 JSON 추출                 │  │
│  │    • ToolCallGuardrails: 무한 루프 차단, 중복 주문 방지, 정체 상태 방지            │  │
│  │    • 뇌동매매 방지 스로틀: 청산 후 30분 재진입 쿨다운 & 시간당 주문 한도          │  │
│  │    • ContextCompressor: 명령 탈취 방지 Token 압축 (비상 청산 명령 최우선 보장)     │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                                                         │
│                ▼                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 4. Apple Silicon MLX 네이티브 추론 가속                                            │  │
│  │    • TurboQuant 3.5b/4b 혼합 양자화 및 Continuous Batching                         │  │
│  │    • RAM + SSD 계층형 영구 프롬프트 캐싱 (APC)                                     │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. 인메모리 컬럼형 시장 데이터 엔진
- **WebSocket 상시 연결**: `wss://api.hyperliquid.xyz/ws`에 직결하여 24시간 거래대금 Top 30 선물을 실시간 추적.
- **지연 없는 인메모리 연산**: 호가창 불균형(Imbalance), 누적 볼륨 델타(CVD), Volume Profile(POC/VAH/VAL), ATR/ADX를 RAM에서 직접 산출.
- **Point-in-Time 시간 여행 (`as_of`)**: 과거 임의 시점의 스냅샷을 오차 없이 복원하여 백테스팅 시 룩어헤드 편향(미래 함수)을 완전히 배제.

### 2. 모델은 제안하고, 로컬 Runtime이 엄격히 통제
- **루프 차단 및 주문 보호**: 동일 파라미터 실패 시 즉각 차단하여 폭주 주문 방지.
- **과잉 매매 방지(Anti-Overtrading)**: 하루 2~4회의 고품질 거래 및 45~90분 보유 원칙을 프롬프트에 주입하고, 청산 후 **30분간 동일 코인 재진입 금지 쿨다운**을 강제.
- **명령 탈취 방지 컨텍스트 압축**: 장기 대화 요약 시에도 `SUMMARY_PREFIX`를 통해 '비상 전량 청산' 및 '즉시 정지' 명령의 최우선 실행 보장.

### 3. 4대 자율 멀티 에이전트 플릿
- **Scout(정찰원)**: 변동성 급확장, 대형 체결 클러스터, 펀딩비 역전을 실시간 감시.
- **Analyst(분석가)**: 멀티 타임프레임 추세 및 진입/청산 라인 설계.
- **Risk Officer(리스크 관리관)**: 계좌 레버리지, 증거금 사용률(< 50%)을 엄격히 심사.
- **Executor(집행원)**: 로컬 OMS를 통해 모의 또는 서명 실주문을 체결.

### 4. Apple Silicon 네이티브 MLX 가속
- **Apple M1 / M2 / M3 / M4 / M5** (Pro/Max/Ultra) 전기종 완벽 지원.
- **Continuous Batching**: 다중 에이전트 병렬 추론 시에도 레이턴시 저하 없음.
- **계층형 APC 캐싱**: 프롬프트를 SSD에 영구 캐싱하여 밀리초 만에 웜 스타트.

### 5. 네이티브 Web 트레이딩 터미널 UI
- 원클릭으로 구동되는 다크 테마 웹 터미널: `http://localhost:5119`.
- **로컬 KMS 암호화**: 거래소 API 키를 AES-256-GCM으로 로컬에 안전하게 보관.
- **대화형 승인 게이트**: 실주문 전 화면에서 원클릭 수동 승인 또는 전자동 오토파일럿 선택 가능.

---

## ⚡ 빠른 시작 (30초)

### 1. 설치
```bash
git clone https://github.com/123456btc/xmlx_vlm.git
cd xmlx_vlm
pip install -e .
```

### 2. 로컬 AI 트레이딩 OS 및 터미널 실행
```bash
# 추론 엔진과 웹 터미널을 한 번에 실행
./service.sh start
```
- 🧠 **OpenAI / Anthropic 듀얼 추론 API**: `http://localhost:5118`
- 🖥️ **AI Trader 웹 트레이딩 터미널**: `http://localhost:5119`

### 3. 상태 확인 및 종료
```bash
./service.sh status
./service.sh stop
```

---

## 🧪 자동화 테스트 검증

```bash
PYTHONPATH=. pytest tests/test_agent_core.py \
                     tests/test_skills_curator.py \
                     tests/test_kanban_board.py \
                     tests/test_ai_trader_agent_core.py \
                     tests/test_columnar_market_store.py \
                     tests/test_throttle_guardrails.py -v
```
> **테스트 상태**: `35 / 35 통과 (100% 성공)` ✅

---

## 📄 라이선스

본 프로젝트는 [MIT License](LICENSE)에 따라 배포됩니다.
