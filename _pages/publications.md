---
layout: page
permalink: /publications/
title: publications
description: Publications and Patents
years: [2022, 2021, 2014, 2012]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
