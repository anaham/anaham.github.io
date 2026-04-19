---
layout: default
title: Photolog 📸
category: photolog
permalink: /categories/photolog/
---

# Photolog 📸

사진과 함께하는 기록.

---

{% assign cat_posts = site.posts | where_exp: "post", "post.categories contains 'photolog'" %}
{% for post in cat_posts %}
- **[{{ post.title }}]({{ post.url }})** — {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}

---

*Powered by Wisp ✨ & Naaru.code ✨ & Shamino ⛏️*
