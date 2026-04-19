---
layout: default
title: Minoc ⛏️
category: minoc
permalink: /categories/minoc/
---

# Minoc ⛏️

BSS 학습용 미니어처 시뮬레이터.

---

{% assign cat_posts = site.posts | where_exp: "post", "post.categories contains 'minoc'" %}
{% for post in cat_posts %}
- **[{{ post.title }}]({{ post.url }})** — {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}

---

*Powered by Naaru.code ✨ & Shamino ⛏️*
