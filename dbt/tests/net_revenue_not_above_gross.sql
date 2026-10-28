select order_id
from {{ ref('fct_orders') }}
where net_revenue > gross_revenue + 0.001
