---
layout: default
title: "Books & Movies 🎬📚"
permalink: /categories/books-movies
---

# Books & Movies 🎬📚

책과 영화 리뷰

{% for post in site.posts %}
{% if post.categories contains "books-movies" %}
- **[{{ post.title }}]({{ post.url }})** - {{ post.date | date: "%Y-%m-%d" }}
{% endif %}
{% endfor %}
