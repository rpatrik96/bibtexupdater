---
cssclasses:
  - research-note
aliases:
- {{citekey}}
- "{{title}}"
authors:
  {{authors}}
DOI:
  {{DOI}}
citekey:
  {{citekey}}
year:
  {{date|format("YYYY")}}
keywords:
{%- if tags and tags.length > 0 %}
{%- for tag in tags %}
  - "[[{{tag.tag}}]]"
{%- endfor %}
{%- endif %}
related:
---



# {{title}}

> [!abstract] Abstract

> {{abstractNote}}



> [!info]

> - [Select in Zotero]({{select}})

> - [Open online]({{url}})


 ![[Review @{{citekey}}]]

---



{% macro calloutHeader(color) -%}

{%- if color == "#ffd400" -%}

Summary

{%- endif -%}

{%- if color == "#ff6666" -%}

Important

{%- endif -%}

{%- if color == "#5fb236" -%}

Notation

{%- endif -%}

{%- if color == "#2ea8e5" -%}

Technical details/experiments

{%- endif -%}

{%- if color == "#a28ae5" -%}

Contribution

{%- endif -%}

{%- if color == "#e56eee" -%}

Connection to literature

{%- endif -%}

{%- if color == "#f19837" -%}

Assumption

{%- endif -%}

{%- if color == "#aaaaaa" -%}

Wrong?!

{%- endif -%}

{%- endmacro -%}



{% macro genCallout(annotation) -%}

{%- if annotation.imageRelativePath -%}

> [!quote{% if annotation.color %}|{{annotation.color}}{% endif %}] {{calloutHeader(annotation.color)}}  [*(go to image, p.{{annotation.pageLabel}})*](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.pageLabel}}&annotation={{annotation.id}}){%- elif annotation.annotatedText -%}

> [!quote{% if annotation.color %}|{{annotation.color}}{% endif %}] {{calloutHeader(annotation.color)}} [*(go to annotation, p.{{annotation.pageLabel}})*](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.pageLabel}}&annotation={{annotation.id}}){%- elif annotation.comment -%}

> [!note{% if annotation.color %}|{{annotation.color}}{% endif %}] {{calloutHeader(annotation.color)}}  [*(go to comment, p.{{annotation.pageLabel}})*](zotero://open-pdf/library/items/{{annotation.attachment.itemKey}}?page={{annotation.pageLabel}}&annotation={{annotation.id}}){%- endif -%}

{%- endmacro -%}





## Notes



{% if attachments and attachments.length > 0 -%}

{%- set attachments = attachments | filterby("path", "endswith", ".pdf") %}

{%- endif -%}

### Imported on {{importDate | format("YYYY-MM-DD h:mm a")}}

{%for attach_ind in range(0, attachments.length) %}

## [*{{ attachments[attach_ind].title }}*](zotero://open-pdf/library/items/{{ attachments[attach_ind].itemKey}})

{%- for annotation in attachments[attach_ind].annotations %}

{{genCallout(annotation)}}

{% if annotation.imageRelativePath -%}

> ![[{{annotation.imageRelativePath}}]]

{%- if annotation.comment %}

> [!note] Comment
> {{annotation.comment | replace("\n", "\n>> ") }}

{%- endif -%}

{%- elif annotation.annotatedText -%}

> {{annotation.annotatedText}}

{%- if annotation.comment %}

> [!note] Comment
> {{annotation.comment  | replace("\n", "\n>> ") }}

{%- endif -%}

{%- elif annotation.comment -%}

> {{annotation.comment | replace("\n", "\n> ")}}

{%- endif %}

{% endfor -%}

{% endfor %}

---
