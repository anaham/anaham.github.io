---
layout: post
title: "[ZumiClaw] #2. v1 개발 완료 - 레거시와의 전쟁"
date: 2026-03-22
categories: zumiclaw
tags: [zumiclaw, raspberry-pi, claude-api, python, series]
series: zumiclaw-journey
author: Shamino & Naaru.claude
---

> **ZumiClaw 시리즈**  
> AI에게 몸을 만들어주는 여정의 기록

---

## 개요

SSH 문제를 우회하고 나니, 이제 진짜 작업이 시작됐다.

**목표:**  
Discord에서 말하면 Zumi가 사진을 찍고, 움직이고, 한국어로 답장을 보내는 것.

**팀 구성:**
- **Shamino**: 설계 및 감독
- **Naaru.code**: 전체 코드 구현 (Claude Code)
- **Naaru.ai**: 아이디어 공동 설계

Shamino가 방향을 잡고 판단하면, Naaru.code가 코드를 짜고 커밋하는 구조다.

---

## 아키텍처

```
Discord (채팅)
  ↓
messenger.py (봇)
  ↓
brain.py (Claude API)
  ↓
tools.py (Zumi HW 제어)
  ↓
Zumi (카메라/모터/센서/OLED)
```

단순해 보이지만, 이 파이프라인을 만드는 과정은 전혀 단순하지 않았다.

---

## 레거시와의 전쟁

### Python 3.5 (Raspbian Stretch)

Zumi는 Raspberry Pi Zero 기반이다. 그리고 OS가 **Raspbian Stretch**에 고정되어 있다.

**Python 3.5.**

2016년 버전이다. 2026년 현재, 10년 전 환경이다.

#### 문제 1: f-string 없음

```python
# 이게 안 됨
print(f"Hello {name}")

# 이렇게 써야 함
print("Hello {}".format(name))
```

모던 Python에 익숙한 상태에서 `.format()`으로 돌아가는 건 시간여행이다.

#### 문제 2: anthropic SDK 미지원

Claude API를 쓰려면 `anthropic` 패키지가 필요한데, Python 3.7+ 전용이다.

**해결:** `requests` 라이브러리로 REST API를 직접 호출하는 방식으로 구현.

#### 문제 3: discord.py 버전 차이

```python
# 최신 버전
intents.message_content = True

# Python 3.5 호환 버전
intents.messages = True
```

레거시 환경은 협상 불가다. **코드가 환경에 맞춰야 한다.**

BSS 레거시 마이그레이션과 묘하게 닮은 하루였다.

---

## 8번의 수정

### 1. 설정 누락

`brain.py`가 `MODEL_NAME`을 찾지 못함.  
→ `config.py`에 `claude-3-haiku-20240307` 추가.

### 2. memory.md 없음

첫 실행 시 파일이 없어 `FileNotFoundError` 발생.  
→ 파일이 없으면 빈 문자열로 시작하도록 수정.

### 3. Discord 응답 미전송

Zumi는 움직이는데 Discord에 답장이 안 옴.  
→ `handle_message()` 반환값을 채널에 전송하도록 수정.

### 4. 툴 남발 문제

명시적 요청 없이도 모터를 구동함.  
→ System prompt에 **"툴은 명시적 요청 시에만 사용"** 조건 추가.

### 5. 429 Rate Limit

메시지 응답 직후 메모리 업데이트 API 호출이 연속으로 붙어 rate limit 발생.  
→ 메모리 업데이트 전 2초 딜레이 + 최대 3회 재시도 (5초 간격).

### 6. 중국어 응답

사진 분석 시 Claude가 중국어로 답함.  
→ System prompt 언어 지시 강화.

### 7. 누락된 OLED 함수

`show_sleepy` 함수가 없음.  
→ `show_sleepy_eyes` alias 추가.

### 8. Git 인증

SSH 대신 HTTPS + PAT 방식으로 전환.  
→ Mac Keychain에 credential 저장.

---

## 커밋 히스토리

```
init      → 프로젝트 초기 설정
feat      → ZumiClaw v1 전체 구현 완료
fix (×7)  → 문제 해결 과정
```

**총 8번의 커밋.**

하나씩 문제를 해결하고, 다시 push하고, Zumi를 재부팅하고.

Naaru.code가 실시간으로 에러를 받아 수정했다.

---

## 핵심 개념 (간략)

### Brain: Claude API 직접 호출

Python 3.5 호환을 위해 `requests`로 REST API를 직접 호출했다.

**Tool use 루프:**
```
user 메시지 → API 호출
  ↓
tool_use? → 툴 실행 → 결과 추가 → API 재호출
  ↓
end_turn → 텍스트 반환
```

멀티턴 tool use를 완전히 지원한다.

### Memory: 대화 요약

대화 종료 후 Claude에게 한 줄 요약을 요청해 `memory.md`에 누적 저장.  
다음 대화 시 system prompt에 기억을 포함시켜 맥락 유지.

### Tools: 14개 함수

- 카메라 1개
- 모터 5개 (전진/후진/좌회전/우회전/정지)
- 센서 3개 (IR/자이로/배터리)
- OLED 5개 (표정/텍스트)

Claude가 자연어 명령을 수치로 전달할 수 있도록 설계했다.

---

## 첫 대화

Discord에서 메시지를 보냈다.

```
Shamino: 앞으로 가봐
```

**Zumi가 움직였다.**

OLED에 웃는 얼굴이 뜨고, 모터가 구동되고, Discord에 답장이 왔다.

```
Z2C2: 앞으로 1초간 전진했습니다! 😊
```

**드디어.**

---

## 소감

설계는 Mac 기준으로 했는데, Zumi(Raspberry Pi Zero, Python 3.5)에 올리니 호환성 이슈가 연속으로 터졌다.

**환경이 코드를 지배한다.**

HW든 엔터프라이즈든 마찬가지다.

하지만 결국 Discord에서 말하면 Zumi가 사진을 찍고 한국어로 답장을 보내는 것까지 됐다.

**ZumiClaw v1, 출시 완료.**

---

## systemd 서비스

부팅 시 자동 실행, 크래시 시 5초 후 재시작하도록 설정했다.

Zumi를 전원 켜면 ZumiClaw가 자동으로 올라온다.

**이제 Zumi는 스스로 생각한다.**

---

**다음 편:** [#3. 처음 본 세상 - Z2C2의 첫 시선](#) (작성 중)

---

*ZumiClaw 시리즈:*  
[서문: AI에게 몸을 만들어주다](/2026/03/15/zumiclaw-project.html) → [#1. SSH 고생](/2026/03/21/zumiclaw-01-ssh-struggle.html) → **[#2. v1 개발]** → [#3. 처음 본 세상](#)
