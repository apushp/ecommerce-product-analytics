-- one row per month with the store's core metrics.
-- the north star is active_buyers. revenue breaks down as
--   gross_revenue = active_buyers * orders_per_buyer * avg_order_value
-- so any change in revenue can be traced to one of those three drivers
with orders as (
    select
        *,
        date_trunc('month', ordered_at) as order_month,
        status <> 'cancelled' as is_placed
    from {{ ref('fct_orders') }}
),

items as (
    select
        date_trunc('month', ordered_at) as order_month,
        count(*) filter (where counts_toward_gross) as items_sold,
        count(*) filter (where status = 'returned') as items_returned
    from {{ ref('fct_order_items') }}
    group by 1
),

monthly as (
    select
        order_month,
        count(*) as orders_created,
        count(*) filter (where is_placed) as orders_placed,
        count(*) filter (where status = 'cancelled') as orders_cancelled,
        count(distinct user_id) filter (where is_placed) as active_buyers,
        count(distinct user_id) filter (where is_placed and order_number = 1) as new_buyers,
        sum(gross_revenue) as gross_revenue,
        sum(net_revenue) as net_revenue,
        sum(net_margin) as net_margin
    from orders
    group by 1
)

select
    m.order_month,
    -- the last month is cut off at as_of_date, so it shouldn't be compared to full months
    m.order_month + interval 1 month <= '{{ var("as_of_date") }}' as is_complete_month,
    m.active_buyers,
    m.new_buyers,
    m.active_buyers - m.new_buyers as returning_buyers,
    m.orders_placed,
    m.orders_placed / m.active_buyers as orders_per_buyer,
    m.gross_revenue,
    m.gross_revenue / m.orders_placed as avg_order_value,
    m.net_revenue,
    m.net_margin,
    m.net_margin / nullif(m.net_revenue, 0) as net_margin_pct,
    m.orders_cancelled / m.orders_created as cancel_rate,
    i.items_returned / i.items_sold as item_return_rate
from monthly m
left join items i using (order_month)
