-- one row per customer who has bought something, with the recency, frequency
-- and monetary numbers that segmentation and lifetime value start from.
-- a purchase is an order the customer kept at least one item from: cancelled
-- orders and orders returned in full have no net revenue, so they don't count
with purchases as (
    select user_id, ordered_at, net_revenue
    from {{ ref('fct_orders') }}
    where net_revenue > 0
)

select
    user_id,
    min(ordered_at) as first_purchase_at,
    max(ordered_at) as last_purchase_at,
    date_diff('day', max(ordered_at), timestamp '{{ var("as_of_date") }}') as recency_days,
    count(*) as frequency,
    sum(net_revenue) as monetary_total,
    avg(net_revenue) as monetary_avg,
    date_diff('day', min(ordered_at), timestamp '{{ var("as_of_date") }}') as tenure_days
from purchases
group by user_id
