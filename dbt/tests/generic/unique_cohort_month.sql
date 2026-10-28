{% test unique_cohort_month(model) %}
-- each cohort and month offset should appear once
select cohort_month, months_since_first
from {{ model }}
group by 1, 2
having count(*) > 1
{% endtest %}
