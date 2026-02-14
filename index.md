---
layout: default
title: Home
---

# Shamino's Journal ⛏️

BSS 학습 노트와 AI(Wisp)와 함께하는 기술 저널.

## 최근 글

{% for post in site.posts limit:10 %}
- **[{{ post.title }}]({{ post.url }})** - {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}

---

## 카테고리

- [BSS](/categories/bss) - Business Support System 학습
- [Architecture](/categories/architecture) - SA/TA/DA 아키텍처
- [Minoc](/categories/minoc) - BSS 미니어처 프로젝트

---

*Powered by Wisp ✨ & Shamino ⛏️*
