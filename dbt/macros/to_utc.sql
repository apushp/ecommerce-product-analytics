{# raw timestamps come in as timestamptz; everything downstream is plain UTC #}
{% macro to_utc(col) -%}
    timezone('UTC', {{ col }})
{%- endmacro %}
