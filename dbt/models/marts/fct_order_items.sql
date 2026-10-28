-- revenue rules used everywhere downstream:
--   gross revenue = every item that wasn't cancelled
--   net revenue   = gross minus returned items
select
    oi.order_item_id,
    oi.order_id,
    oi.user_id,
    oi.product_id,
    o.ordered_at,
    oi.status,
    p.category,
    p.department,
    p.brand,
    oi.sale_price,
    p.cost,
    oi.status <> 'cancelled' as counts_toward_gross,
    oi.status not in ('cancelled', 'returned') as counts_toward_net,
    case when oi.status not in ('cancelled', 'returned') then oi.sale_price - p.cost else 0 end as net_margin
from {{ ref('stg_order_items') }} oi
join {{ ref('stg_orders') }} o using (order_id)
join {{ ref('stg_products') }} p using (product_id)
