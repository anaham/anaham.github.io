---
layout: default
title: "..."
category: draft
permalink: /categories/draft/
---

# ...

작업 중인 글들.

---

{% assign draft_posts = site.posts | where_exp: "post", "post.categories contains 'draft'" %}
{% for post in draft_posts %}
- **[{{ post.title }}]({{ post.url }})** - {{ post.date | date: "%Y-%m-%d" }}
{% endfor %}
