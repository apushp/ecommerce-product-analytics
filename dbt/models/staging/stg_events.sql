-- about half the events are anonymous (no user_id), so sessions are the unit for funnels
select
    id as event_id,
    session_id,
    user_id,
    sequence_number,
    lower(event_type) as event_type,
    uri,
    browser,
    traffic_source,
    {{ to_utc('created_at') }} as event_at
from {{ source('thelook', 'events') }}
where {{ to_utc('created_at') }} < '{{ var("as_of_date") }}'
