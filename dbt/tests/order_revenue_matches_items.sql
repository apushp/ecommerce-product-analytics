-- order-level net revenue has to add up to the item-level total
select 1
from (select sum(net_revenue) as total from {{ ref('fct_orders') }}) o
cross join (select sum(sale_price) filter (where counts_toward_net) as total from {{ ref('fct_order_items') }}) i
where abs(o.total - i.total) > 0.01
