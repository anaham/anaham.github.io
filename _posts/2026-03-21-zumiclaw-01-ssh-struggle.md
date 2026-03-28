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

## 돌파구: HTTPS + PAT

SSH가 안 되면 다른 방법을 써야 한다.

Git은 SSH 말고도 HTTPS 방식을 지원한다. GitHub Personal Access Token(PAT)을 만들어서 credential로 쓰는 방식이다.

### 설정 과정

1. GitHub에서 PAT 생성
2. Mac Keychain에 저장
3. Git remote를 HTTPS로 변경

```bash
git remote set-url origin https://github.com/shamino/zumiclaw.git
```

**이제 push가 된다!**

SSH로 Zumi에 접속하는 건 실패했지만, **코드를 Zumi에 배포하는 문제는 해결**했다.

---

## 깨달음

**"접속은 안 되지만, 코드는 올라간다."**

이상하게 들리지만, 생각해보니 충분하다.

- Zumi가 부팅되면 자동으로 코드를 실행하도록 설정
- Git pull로 최신 코드 받아오기
- systemd로 서비스 등록

**Zumi를 직접 조작할 필요가 없다.**  
코드를 push하면, Zumi가 알아서 실행하면 된다.

---

## 다음 단계

SSH 문제는 우회했다. 이제 진짜 작업이 시작된다.

**v1 개발.**

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
[서문: AI에게 몸을 만들어주다](/2026/03/15/zumiclaw-project.html) → **[#1. SSH 고생]** → [#2. v1 개발](#) → [#3. 처음 본 세상](#)
