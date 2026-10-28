with order_stats as (
    select
        user_id,
        min(ordered_at) as first_order_at,
        max(ordered_at) as last_order_at,
        count(*) as lifetime_orders,
        count(*) filter (where status not in ('cancelled', 'returned')) as lifetime_kept_orders,
        sum(net_revenue) as lifetime_net_revenue,
        sum(net_margin) as lifetime_net_margin
    from {{ ref('fct_orders') }}
    group by user_id
)

select
    u.user_id,
    u.age,
    case
        when u.age < 25 then '18-24'
        when u.age < 35 then '25-34'
        when u.age < 45 then '35-44'
        when u.age < 55 then '45-54'
        else '55+'
    end as age_band,
    u.gender,
    u.country,
    u.state,
    u.acquisition_channel,
    u.signed_up_at,
    date_trunc('month', u.signed_up_at) as signup_month,
    s.first_order_at,
    s.last_order_at,
    coalesce(s.lifetime_orders, 0) as lifetime_orders,
    coalesce(s.lifetime_kept_orders, 0) as lifetime_kept_orders,
    coalesce(s.lifetime_net_revenue, 0) as lifetime_net_revenue,
    coalesce(s.lifetime_net_margin, 0) as lifetime_net_margin,
    s.first_order_at is not null as has_ordered
from {{ ref('stg_users') }} u
left join order_stats s using (user_id)
