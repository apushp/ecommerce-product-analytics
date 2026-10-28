with items as (
    select
        order_id,
        sum(sale_price) filter (where counts_toward_gross) as gross_revenue,
        sum(sale_price) filter (where counts_toward_net) as net_revenue,
        sum(net_margin) as net_margin,
        count(*) as item_rows
    from {{ ref('fct_order_items') }}
    group by order_id
)

select
    o.order_id,
    o.user_id,
    o.ordered_at,
    o.status,
    o.item_count,
    coalesce(i.gross_revenue, 0) as gross_revenue,
    coalesce(i.net_revenue, 0) as net_revenue,
    coalesce(i.net_margin, 0) as net_margin,
    row_number() over (partition by o.user_id order by o.ordered_at, o.order_id) as order_number,
    date_diff(
        'day',
        lag(o.ordered_at) over (partition by o.user_id order by o.ordered_at, o.order_id),
        o.ordered_at
    ) as days_since_previous_order,
    epoch(o.delivered_at - o.shipped_at) / 86400.0 as days_to_deliver
from {{ ref('stg_orders') }} o
left join items i using (order_id)
