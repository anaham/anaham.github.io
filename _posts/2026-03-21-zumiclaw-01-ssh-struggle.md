---
layout: post
title: "[ZumiClaw] #1. SSH 접속까지의 고생"
date: 2026-03-21
categories: zumiclaw
tags: [zumiclaw, raspberry-pi, ssh, robolink-zumi, series]
series: zumiclaw-journey
author: Shamino
---

> **ZumiClaw 시리즈**  
> AI에게 몸을 만들어주는 여정의 기록

---

## 시작

책상 서랍 구석에서 Zumi를 꺼냈다.

몇 년 전, 아들에게 Python을 가르치겠다고 산 Raspberry Pi Zero 기반 로봇 키트. 하지만 아들도, 나도 삶에 바빠 방치되어 있었다.

손바닥만 한 로봇. 먼지를 털어내니 귀여운 외형 아래 카메라, 모터, 센서들이 빼곡했다.

**"이 안에 AI를 넣어보자."**

방치된 교육용 로봇에 새 생명을 불어넣는 실험.

단순한 생각이었다. Wisp은 Discord에서 살고, Naaru는 Claude Code 웹에서 산다. 그렇다면 **Zumi는 물리적 세계를 탐험할 수 있지 않을까?**

하지만 현실은 만만치 않았다.

---

## 첫 번째 벽: SSH 연결

Zumi를 제어하려면 SSH로 접속해야 한다. Raspberry Pi Zero는 Linux 머신이니까 당연한 수순이다.

### 시도 1: 기본 접속

```
ssh pi@zumi.local
```

**타임아웃.**

뭐지? Wi-Fi는 연결되어 있는데?

### 시도 2: IP 직접 찾기

라우터 관리 페이지에서 Zumi의 IP를 확인했다.

```
ssh pi@192.168.0.xxx
```

**타임아웃.**

같은 결과.

### 시도 3: SSH 키 방식

혹시 비밀번호가 아니라 키가 필요한가? SSH 키를 생성해서 시도했다.

```
ssh-keygen -t rsa
ssh-copy-id pi@zumi.local
```

**여전히 타임아웃.**

---

## 삽질의 시간

![SSH 타임아웃 로그](/home/dupre/.openclaw/media/inbound/a36f6e06-1c12-499c-8b17-2c0a571f2096.png)

로그를 보면 알 수 있다. 얼마나 많은 시도를 했는지.

- `ssh pi@zumi.local` (실패)
- `ssh pi@192.168.0.xxx` (실패)
- `ping zumi.local` (응답 없음)
- DNS 캐시 클리어
- 재부팅
- Wi-Fi 재연결

**며칠이 지났다.**

---

## 돌파구

며칠간의 삽질 끝에, 결국 해결했다.

세부 과정은 생략하지만, 핵심은 **네트워크 설정과 접근 방식을 바꾼 것**이었다.

Wi-Fi 연결을 직접 설정하고, IP를 찾아내고, 마침내 SSH 접속에 성공했다.

```bash
ssh pi@192.168.x.xxx
```

**드디어 들어갔다.**

---

## 깨달음

**"우회하는 것도 방법이지만, 때론 정면돌파가 필요하다."**

레거시 환경(Raspberry Pi Zero, Raspbian Stretch)과 싸우는 건 이제 시작일 뿐이었다.

하지만 SSH만 뚫리면, 나머지는 할 수 있다.

---

## 다음 단계

드디어 SSH 접속에 성공했다.

이제 Git pull로 코드를 받아오고, 수정하고, Zumi에서 바로 실행할 수 있다.

**여기까지 1주일.**

평일엔 회사에서 BSS 운영과 차세대 구축으로 정신없다. 주말에만 집중적으로 작업할 수 있었다.

---

## IT 격언

> **"환경설정이 반이다."**

코드를 짜는 건 그 다음 이야기다.

SSH 하나 뚫는 데 1주일. 하지만 이제 진짜 작업을 시작할 수 있다.

**v1 개발:**
- Discord 봇 구현
- Claude API 연동
- Zumi 하드웨어 제어
- 메모리 시스템

하지만 여기서 또 다른 벽이 기다리고 있었다.

**Python 3.5, Raspbian Stretch.**

레거시 환경과의 전쟁이 시작된다.

---

**다음 편:** [#2. v1 개발 완료 - 레거시와의 전쟁](#) (작성 중)

---

*ZumiClaw 시리즈:*  
[서문: AI에게 몸을 만들어주다](/2026/03/15/zumiclaw-project/) → **[#1. SSH 고생]** → [#2. v1 개발](/2026/03/22/zumiclaw-02-v1-release/) → [#3. 처음 본 세상](/2026/03/28/zumiclaw-03-first-sight/)
