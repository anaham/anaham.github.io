---
layout: default
title: "Useful Stuff 🛠️"
permalink: /categories/useful-stuff
---

# Useful Stuff 🛠️

유용한 코드, 도구, 팁 모음

{% for post in site.posts %}
{% if post.categories contains "useful-stuff" %}
- **[{{ post.title }}]({{ post.url }})** - {{ post.date | date: "%Y-%m-%d" }}
{% endif %}
{% endfor %}
