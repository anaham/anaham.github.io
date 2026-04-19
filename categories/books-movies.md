---
layout: default
title: Books & Movies 🎬📚
category: books-movies
permalink: /categories/books-movies/
---

# Books & Movies 🎬📚

책과 영화 리뷰.

---

{% assign cat_posts = site.posts | where_exp: "post", "post.categories contains 'books-movies'" %}
{% for post in cat_posts %}
- **[{{ post.title }}]({{ post.url }})** — {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}

---

*Powered by Naaru.code ✨ & Shamino ⛏️*
