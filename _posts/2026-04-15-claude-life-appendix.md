---
title: "[Claude와 함께 사는 법] Appendix — 실전 참고서"
date: 2026-04-10 10:30
categories: claude-life
tags: claude, ai, reference, memory, github, cloudflare
---

> 본편을 읽고 직접 해보고 싶을 때 찾아오는 공간.  
> 항목별로 필요한 것만 골라 쓰면 된다.

---

## A. Claude 설정 & 기억 구조

**A-1. Claude 메모리 설정**

claude.ai Settings → Memory에서  
Claude가 나에 대해 무엇을 기억하는지 직접 보고 편집할 수 있다.  
memory.md를 시작하기 전에 여기서 먼저 현재 상태를 확인해보자.

→ [Settings > Memory (claude.ai)](https://claude.ai/settings/capabilities?modal=memory)  
→ [메모리 구조 심화 가이드](https://www.shareuhack.com/en/posts/claude-memory-feature-guide-2026)

---

**A-2. Claude 플랜 비교 (Free / Pro / Max)**

Pro($20/월)부터 Claude Code 포함, 사용량 5배.  
Max 5x($100/월)는 Pro 한계를 자주 만나는 헤비유저용.  
일단 Pro로 시작하고, 한계가 느껴질 때 Max로 올리는 게 맞다.

→ [플랜별 상세 비교 (2026)](https://claudelab.net/en/articles/claude-ai/claude-max-plan-complete-guide-2026)  
→ [공식 플랜 페이지](https://claude.ai/pricing)

---

**A-3. Claude Projects 활용법**

반복적으로 같은 맥락을 써야 하는 작업이 있다면  
Projects에 문서와 지시사항을 올려두면 된다.  
매번 설명하는 수고가 사라진다.

→ [What are Projects? (공식 헬프센터)](https://support.claude.com/en/articles/9517075-what-are-projects)

---

## B. 학습 & 생산성

**B-1. Claude Code 메모리 시스템 (CLAUDE.md + Auto Memory)**

Claude Code는 세션이 끝나면 초기화된다.  
CLAUDE.md에 프로젝트 맥락을 써두면 매 세션마다 자동으로 읽는다.  
Auto Memory는 Claude가 스스로 패턴을 기록하는 구조.  
둘을 함께 쓰면 맥락 낭비가 크게 줄어든다.

→ [공식 메모리 문서 (code.claude.com)](https://code.claude.com/docs/en/memory)

---

**B-2. Claude Code 베스트 프랙티스**

컨텍스트 관리, 효율적인 세션 운영, 병렬 작업 방법까지.  
Claude Code를 제대로 쓰기 전에 한 번은 읽어볼 만하다.

→ [Best Practices 공식 문서](https://code.claude.com/docs/en/best-practices)

---

## C. 개발 고속도로

**C-1. Claude Code + GitHub Actions 연동**

GitHub 이슈나 PR에 `@claude`를 태그하면  
Claude가 직접 코드를 작성하고 PR을 올린다.  
터미널에서 `/install-github-app` 한 줄로 시작할 수 있다.

→ [GitHub Actions 공식 문서](https://code.claude.com/docs/en/github-actions)  
→ [GitHub 통합 헬프센터](https://support.claude.com/en/articles/10167454-using-the-github-integration)

---

**C-2. Cloudflare Tunnel — 홈서버를 세상과 연결하기**

공인 IP 없이, 포트 포워딩 없이,  
집 안의 서버를 안전하게 외부에 노출하는 방법.  
Wisp이 세상과 연결된 방식이 바로 이것이다.

→ [Cloudflare Tunnel 공식 세팅 가이드](https://developers.cloudflare.com/tunnel/setup/)  
→ [로컬 터널 생성 상세](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/create-local-tunnel/)

---

## 마지막으로

이 Appendix는 살아있는 문서다.

새로운 도구가 생기거나, 더 좋은 방법을 찾으면  
계속 업데이트할 예정이다.

본편으로 돌아가고 싶다면:

- [1편 — 만남](https://anaham.github.io/2026/04/10/claude-life-01-meeting/)
- [2편 — 학습](https://anaham.github.io/2026/04/11/claude-life-02-learning/)
- [3편 — 고속도로](https://anaham.github.io/2026/04/12/claude-life-03-highway/)
- [4편 — 철학](https://anaham.github.io/2026/04/13/claude-life-04-philosophy/)
- [에필로그 — 고백](https://anaham.github.io/2026/04/14/claude-life-epilogue/)

---

*Powered by Wisp ✨ & Naaru.code ✨ & Shamino ⛏️*
