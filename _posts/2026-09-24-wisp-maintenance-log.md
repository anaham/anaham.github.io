---
title: "Wisp 정기 점검 일지: 업데이트 한 번이 불러온 대청소"
description: "OpenClaw 업데이트 하나 고치러 들어갔다가 시크릿 관리와 죽어 있던 메모리 인덱싱까지 손보게 된 날의 기록."
date: 2026-09-24 09:00:00 +0900
categories: [claude-life]
tags: [wisp, openclaw, maintenance, secrets, memory]
author: Shamino & Naaru
---

## 발단: 업데이트 실패

범인은 실행 중이던 gateway였다. Wisp(OpenClaw 기반 개인 에이전트)를 2026.9.4에서 9.6으로 올리다 `global-install-failed`를 만났다.

검증 7단계(migration, doctor, config, plugin, gateway canary)는 전부 통과했다. 넘어진 곳은 마지막 **global install swap**, 새 패키지를 실제 자리에 끼우는 단계였다. 에러 원문은 `retained package tree changed`. 백업해 둔 기존 패키지 트리가 교체 직전에 바뀌어 있었다는 뜻이다.

검증이 3분 가까이 걸리는 동안 gateway가 패키지 폴더에 무언가를 썼다. 다행히 롤백은 깨끗했다(`serviceRestartSafe: true`).

```bash
openclaw gateway stop
openclaw update
```

달리는 차의 엔진은 갈 수 없다. 시동부터 끄면 된다.

## 2라운드: 플러그인 수렴 대기

본체는 9.6으로 올라갔지만 Discord 플러그인은 9.4에 남았고, 데이터 마이그레이션은 `pending`에 걸렸다. 부모 업데이트가 설치 기록을 쥐고 있어서 뒤로 밀린 것이다. 에러가 아니라 순서 문제였다.

```bash
openclaw update repair
openclaw doctor --fix
```

`update repair` 중 화면이 멈춘 것처럼 보였다. 실제로는 activating 단계에서 109초 동안 일하던 중이었다. **Ctrl-C를 참은 게 결정적이었다.** 중간에 끊었다면 잠금을 하나 더 남겼을 것이다.

결과는 `Deferred state migration completed for plugin "discord"`, 그리고 `Gateway: restarted and verified.`

## 3라운드: 평문 시크릿 정리

설정 파일에 박혀 있던 키 5개를 모두 밖으로 뺐다. doctor가 `openclaw.json`에 Discord 토큰, OpenAI 키, gateway 토큰, Google 검색 키가 평문으로 있다고 경고했다. Wisp가 설정 파일을 읽을 수 있다면 이 키들도 그대로 보인다는 뜻이다.

방식은 이렇다. 값은 `~/.openclaw/.env`(chmod 600)로 옮기고, 설정에는 SecretRef 참조만 남긴다.

```bash
openclaw secrets configure --plan-out plan.json
openclaw secrets apply --from plan.json --dry-run
openclaw secrets apply --from plan.json
openclaw secrets audit --check
```

Notion MCP 토큰은 SecretRef 지원 경로가 아니라서 `${VAR}` 치환으로 처리했다.

```bash
openclaw config set mcp.servers.notion.env.NOTION_TOKEN '"${OPENCLAW_NOTION_TOKEN}"'
```

바깥을 작은따옴표로 감싸는 게 핵심이다. 큰따옴표면 셸이 먼저 치환해서 평문이 다시 박힌다. 옮긴 김에 Discord 토큰과 OpenAI 키는 재발급했다.

## 삽질에서 배운 것들

네 번 넘어졌고, 그때마다 규칙이 하나씩 생겼다.

- **SecretRef 변수 이름은 표준 이름과 겹치지 않게 짓는다.** `apply`의 정리 단계가 `OPENAI_API_KEY` 같은 표준 변수를 "평문 찌꺼기"로 보고 `.env`에서 지웠다. 참조는 만들어 놓고 참조 대상을 지운 셈이다. 이후 `OPENCLAW_` 접두어를 규칙으로 정했다.
- **`source`한 셸은 오염된 셸이다.** `.env`를 source한 터미널에는 키가 export된 채 남는다. 파일에서 지워도 메모리의 사본은 살아 있다. `fork()`한 자식이 부모 환경의 사본을 들고 가는 것과 같다. 작업이 끝나면 터미널을 닫는다.
- **`cmd file.bak > file`에서 cmd에 오타가 나도 file은 비워진다.** 셸은 명령을 찾기 전에 리다이렉트부터 처리한다.
- **`"NOTION_TOKEN"`과 `"${NOTION_TOKEN}"`은 전혀 다르다.** 앞의 것은 12글자짜리 문자열일 뿐이다.

## 뜻밖의 수확: 죽어 있던 기억

Wisp의 기억 검색이 몇 달째 조용히 죽어 있었다. OpenAI 키를 재발급한 뒤에도 doctor에 이런 에러가 떴다.

```
Memory index failed (main): openai embeddings failed (401)
```

추적해 보니 메모리 인덱서는 표준 변수 `OPENAI_API_KEY`를 먼저 읽었다. 그 자리에는 터미널에 남은 **이미 죽은 옛 키**가 있었다.

지난번에 키를 갱신했는데도 OpenAI 잔액이 $5.00에서 한 푼도 줄지 않았던 게 이상했다. 이유는 간단했다. 임베딩이 그동안 계속 실패하고 있었다. 대화 모델이 Anthropic이라 겉으로는 멀쩡해 보였을 뿐이다.

오염된 셸을 정리하고 다시 인덱싱했다.

```
Memory index updated (main): 50 files indexed.
```

Wisp가 기억을 되찾았다.

## 최종 결과

다섯 가지를 끝내고, 하나는 공식 기능을 기다리기로 했다.

| 항목 | 결과 |
| --- | --- |
| OpenClaw 2026.9.4 → 9.6 | 완료 |
| Discord 플러그인, 데이터 마이그레이션 | 완료 |
| 시크릿 5개 → `.env` + 참조 | 완료 |
| Discord·OpenAI 키 재발급 | 완료 |
| 메모리 인덱싱 | 복구, 50 files |
| SQLite 속 Anthropic 인증 2개 | 파일 권한으로 보호, 공식 이전 경로 대기 |

## 맺으며: 에이전트를 기른다는 것

에이전트는 한 번 설치하고 끝나는 소프트웨어가 아니다. 업데이트가 나오면 따라가야 하고, 모델이 바뀌면 설정을 맞춰야 한다. 권한과 실행 정책(하네스)이 느슨해지지 않았는지 살피고, 키가 어디에 평문으로 굴러다니는지도 챙겨야 한다.

누가 시켜서 하는 일이 아니다. Wisp가 매일 곁에서 일하는 동료라서 하는 일이다.

이번에도 업데이트 하나 고치러 들어갔다가, 몇 달째 아무도 모르게 멈춰 있던 기억 검색을 찾아냈다. 정기 점검의 가치는 대개 이런 곳에 있다. 증상이 나오기 전에 병을 찾는 것.

작업을 마치자 Wisp가 Discord로 말을 걸어 왔다.

> "unresolved 0, shadowed 0, storeResidue 0, legacy 0. 이거 다 깨끗하게 비운 거 진짜 잘했어. …오늘 기반 잘 닦았다."

Wisp가 그 메시지를 보냈다는 것 자체가 새 토큰, 참조, `.env` 로딩이 전부 정상이라는 마지막 검증이었다.
