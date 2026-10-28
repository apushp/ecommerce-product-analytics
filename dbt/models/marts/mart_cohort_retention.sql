-- customers grouped by the month of their first order, and how many of them
-- placed an order again in each later month.
-- months_since_first = 0 is the first-order month itself, so it's always 100%
with first_orders as (
    select user_id, date_trunc('month', ordered_at) as cohort_month
    from {{ ref('fct_orders') }}
    where order_number = 1
),

activity as (
    select distinct
        o.user_id,
        f.cohort_month,
        date_diff('month', f.cohort_month, date_trunc('month', o.ordered_at)) as months_since_first
    from {{ ref('fct_orders') }} o
    join first_orders f using (user_id)
    -- later months only count orders that went through; the first order always
    -- counts so month 0 holds the whole cohort
    where o.status <> 'cancelled' or o.order_number = 1
),

cohort_sizes as (
    select cohort_month, count(*) as cohort_size
    from first_orders
    group by 1
)

select
    a.cohort_month,
    s.cohort_size,
    a.months_since_first,
    count(*) as active_customers,
    count(*) / s.cohort_size as retention_rate
from activity a
join cohort_sizes s using (cohort_month)
group by 1, 2, 3
