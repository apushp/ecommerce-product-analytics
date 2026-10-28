select
    *,
    epoch(session_ended_at - session_started_at) / 60.0 as session_minutes,
    case
        when purchased then 'purchase'
        when added_to_cart then 'cart'
        when viewed_product then 'product'
        else 'browse'
    end as furthest_step
from {{ ref('int_sessions') }}
