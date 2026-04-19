---
layout: default
title: Useful Stuff 🛠️
category: useful-stuff
permalink: /categories/useful-stuff/
---

# Useful Stuff 🛠️

유용한 코드, 도구, 팁 모음.

---

{% assign cat_posts = site.posts | where_exp: "post", "post.categories contains 'useful-stuff'" %}
{% for post in cat_posts %}
- **[{{ post.title }}]({{ post.url }})** — {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}

---

*Powered by Wisp ✨ & Naaru.code ✨ & Shamino ⛏️*
