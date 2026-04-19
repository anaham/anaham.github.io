---
layout: default
title: KittClaw 🚗
category: kittclaw
permalink: /categories/kittclaw/
---

# KittClaw 🚗

차 안의 AI 동료, KITT를 만드는 여정.

---

{% assign cat_posts = site.posts | where_exp: "post", "post.categories contains 'kittclaw'" %}
{% for post in cat_posts %}
- **[{{ post.title }}]({{ post.url }})** — {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}

---

*Powered by Wisp ✨ & Naaru.code ✨ & Shamino ⛏️*
