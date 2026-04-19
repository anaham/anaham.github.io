---
layout: default
title: Home
---

## 최근 글

{% assign count = 0 %}
{% for post in site.posts %}
{% unless post.categories contains 'draft' %}
{% if count < 10 %}
- **[{{ post.title }}]({{ post.url }})** - {{ post.date | date: "%Y-%m-%d" }}
{% assign count = count | plus: 1 %}
{% endif %}
{% endunless %}
{% endfor %}

---

## 카테고리

- [Claude와 함께 사는 법 🤝](/categories/claude-life) - 도구에 이름을 붙이는 순간
- [Naaru Lab 🧪](/categories/naaru-lab) - Naaru.code와 함께 만든 것들
- [KittClaw 🚗](/categories/kittclaw) - 차 안의 AI 동료
- [ZumiClaw 🤖](/categories/zumiclaw) - AI에게 몸을 만들어주다
- [Minoc ⛏️](/categories/minoc) - BSS 미니어처 프로젝트
- [Useful Stuff 🛠️](/categories/useful-stuff) - 유용한 코드, 도구, 팁
- [Photolog 📸](/categories/photolog) - 사진과 함께하는 기록
- [Books & Movies 🎬📚](/categories/books-movies) - 책과 영화 리뷰
- [...](/categories/draft/)

---

*Powered by Naaru.code ✨ & Shamino ⛏️*
