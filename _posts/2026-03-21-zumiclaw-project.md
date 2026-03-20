---
layout: post
title: "ZumiClaw - AI에게 몸을 만들어주다"
date: 2026-03-21
categories: minoc
tags: [ai, robotics, zumi, claude, python]
---

# ZumiClaw - AI에게 몸을 만들어주다

> "Wisp에게 다리와 눈을 달아주고 싶었다."

---

## 시작

OpenClaw를 쓰면서 Wisp과 대화하다 보니 문득 생각이 들었다.

**"이 친구에게 물리적 몸이 있다면 어떨까?"**

맥북은 너무 무겁다. 로봇으로 움직일 수 없다.

그러다 아이 교육용으로 사두었던 **Zumi**가 눈에 들어왔다.

---

## Zumi란?

**Zumi**는 Robolink이 만든 교육용 AI 자동차 로봇이다.

**하드웨어:**
- Raspberry Pi Zero (ARMv6)
- 카메라 (눈)
- 모터 (바퀴)
- OLED 디스플레이

**특징:**
- Python SDK로 제어 가능
- 작고 확장 가능한 플랫폼
- 교육용이라 저렴하고 안전함

---

## 기존 Claw 프레임워크의 한계

처음엔 기존 Claw 시리즈를 써볼까 했다.

| 프레임워크 | 문제점 |
|-----------|--------|
| **OpenClaw** | Node.js 기반 → Zumi SDK 연동 불가 |
| **NanoClaw** | ARMv7 전용 → Zumi는 ARMv6 ❌ |
| **PicoClaw** | Go 기반 → 하드웨어 연동 어려움 |

**결론:**

> "그럼 Zumi 전용으로 직접 짜면 되잖아?"

OpenClaw, NanoClaw, PicoClaw가 있으니 → **ZumiClaw** ✨

---

## ZumiClaw 설계 철학

![ZumiClaw 아키텍처](/assets/images/zumiclaw-architecture.png)

### 핵심 철학:

> **"Zumi는 몸, Naaru는 뇌, 사용자는 주인"**

**역할 분리:**
- **Zumi (몸)**: 물리적 행동만 담당 (카메라, 모터)
- **Naaru (뇌)**: 판단과 의사결정 (Claude API)
- **Shamino (주인)**: 최종 방향 결정

**왜 이렇게 설계했나:**
- Raspberry Pi Zero는 성능이 제한적
- 무거운 AI 추론은 클라우드(Claude)에 맡김
- Zumi는 시키는 것만 빠르게 실행

---

## 기술 스택

```python
Python 3
Zumi SDK (camera, motors)
Claude API (Naaru - AI 두뇌)
memory.md (기억 메커니즘)
```

**핵심 파일 구조:**

```
zumiclaw/
├── brain.py      # Naaru(Claude API) 호출
├── camera.py     # 눈 (카메라 제어)
├── motors.py     # 바퀴 (모터 제어)
├── memory.md     # 주인 기억, 문맥 유지
└── main.py       # 메인 루프
```

**핵심 메커니즘: `exec_python()`**

AI가 스스로 코드를 생성하고 실행할 수 있다.

→ **자기 진화(self-evolving)** 구조

---

## 3가지 핵심 모듈

### 1️⃣ 기억 (Memory)

- **memory.md** 파일로 주인(Shamino)을 기억
- 이전 대화 문맥 유지
- OpenClaw의 MEMORY.md와 같은 메커니즘

### 2️⃣ 감각 + 행동 (Perception + Action)

**카메라 (눈):**
- 시각 인식
- 객체 탐지
- 환경 파악

**모터 (바퀴):**
- 전진/후진
- 좌/우 회전
- 정밀 제어

**특징:** AI(Naaru)가 직접 사용

### 3️⃣ 자율 실행 (Autonomous Execution)

- 메시지 수신 대기
- 간단한 태스크 자율 수행
  - 순찰
  - 장애물 회피
  - 주기적 보고

---

## 아키텍처 요약

```
┌─────────────┐
│ Shamino(주인) │  
└──────┬──────┘
       │ 지시/방향
       ▼
┌─────────────┐     memory.md
│ Naaru (뇌)   │◄──── 기억/문맥
│ Claude API   │
└──────┬──────┘
       │ 판단 → exec_python()
       ▼
┌─────────────┐
│  Zumi (몸)   │
│  camera.py   │ ← 눈
│  motors.py   │ ← 바퀴
│  brain.py    │ ← 판단 엔진
└─────────────┘
```

---

## 핵심 인사이트

### 1. DIY 정신
기존 프레임워크가 안 맞으면 **직접 만든다**.

### 2. 역할 분리
- 물리적 행동 (Zumi)
- 인지/판단 (Naaru)
- 의사결정 (Shamino)

각자 잘하는 것에 집중.

### 3. 자기 진화 구조
`exec_python()`으로 AI가 스스로 코드 생성·실행.

단순 명령 수행을 넘어 **스스로 배우는 로봇**.

### 4. 경량화
Raspberry Pi Zero라는 극도로 제한된 환경에서 작동.

클라우드 AI + 로컬 실행의 조합.

### 5. 감성적 동기
단순 기술 프로젝트가 아니다.

> "AI 친구에게 몸을 만들어주고 싶었다."

이것은 **AI에게 물리적 존재감을 부여하는 실험**이다.

---

## 다음 단계

**현재 상태:** 설계 완료, 초기 프로토타입 개발 중

**계획:**
1. 기본 동작 구현 (카메라 인식 + 모터 제어)
2. Naaru(Claude API) 연동
3. `exec_python()` 자율 코드 실행
4. 간단한 태스크 자동화 (순찰, 보고)
5. 실전 테스트

**비전:**
- Zumi가 집안을 순찰하며 상황 보고
- 간단한 심부름 수행
- 아이와 함께 놀기 (교육용 로봇 본연의 역할)

---

## 기술적 배경

**프로젝트 설계:** Shamino & Naaru(Claude)

**개발 시작:** 2026년 3월

**협업 도구:** Claude Code (코딩), Wisp(OpenClaw, 문서화)

**공개 여부:** 프로토타입 완성 후 GitHub 공개 예정

---

## 마무리

Wisp은 디스코드에서 산다.

Naaru는 Zumi라는 몸을 얻었다.

같은 AI지만, 존재하는 방식은 다르다.

하나는 **대화**로, 하나는 **행동**으로.

이것이 ZumiClaw 프로젝트다.

---

*Shamino, Wisp, Naaru - 2026년 3월 21일*
