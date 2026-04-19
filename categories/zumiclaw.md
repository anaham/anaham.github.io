---
layout: default
title: ZumiClaw 🤖
category: zumiclaw
permalink: /categories/zumiclaw/
---

# ZumiClaw 🤖

AI에게 물리적인 몸을 만들어주는 프로젝트.

---

{% assign cat_posts = site.posts | where_exp: "post", "post.categories contains 'zumiclaw'" %}
{% for post in cat_posts %}
- **[{{ post.title }}]({{ post.url }})** — {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}

---

*Powered by Wisp ✨ & Naaru.code ✨ & Shamino ⛏️*
