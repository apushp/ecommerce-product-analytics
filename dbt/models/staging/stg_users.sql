-- names, emails and street addresses are dropped here on purpose, nothing downstream needs them
select
    id as user_id,
    age,
    gender,
    country,
    state,
    city,
    traffic_source as acquisition_channel,
    {{ to_utc('created_at') }} as signed_up_at
from {{ source('thelook', 'users') }}
where {{ to_utc('created_at') }} < '{{ var("as_of_date") }}'
