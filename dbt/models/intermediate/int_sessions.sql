-- one row per web session, with how far down the funnel it got
with events as (
    select * from {{ ref('stg_events') }}
)

select
    session_id,
    max(user_id) as user_id,
    max(user_id) is not null as is_logged_in,
    any_value(traffic_source) as traffic_source,
    any_value(browser) as browser,
    arg_min(event_type, sequence_number) as landing_page_type,
    min(event_at) as session_started_at,
    max(event_at) as session_ended_at,
    count(*) as event_count,
    bool_or(event_type = 'product') as viewed_product,
    bool_or(event_type = 'cart') as added_to_cart,
    bool_or(event_type = 'purchase') as purchased,
    bool_or(event_type = 'cancel') as cancelled
from events
group by session_id
