---
layout: default
title: "Minoc ⛏️"
permalink: /categories/minoc
---

# Minoc ⛏️

BSS 미니어처 프로젝트

{% for post in site.posts %}
{% if post.categories contains "minoc" %}
- **[{{ post.title }}]({{ post.url }})** - {{ post.date | date: "%Y-%m-%d" }}
{% endif %}
{% endfor %}
