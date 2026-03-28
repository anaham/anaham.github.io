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

- [ZumiClaw 🤖](/categories/zumiclaw) - AI에게 몸을 만들어주다
- [Minoc ⛏️](/categories/minoc) - BSS 미니어처 프로젝트
- [Useful Stuff 🛠️](/categories/useful-stuff) - 유용한 코드, 도구, 팁
- [Photolog 📸](/categories/photolog) - 사진과 함께하는 기록
- [Books & Movies 🎬📚](/categories/books-movies) - 책과 영화 리뷰

---

*Powered by Wisp ✨ & Shamino ⛏️*
