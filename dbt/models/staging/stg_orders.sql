select
    order_id,
    user_id,
    lower(status) as status,
    num_of_item as item_count,
    {{ to_utc('created_at') }} as ordered_at,
    {{ to_utc('shipped_at') }} as shipped_at,
    {{ to_utc('delivered_at') }} as delivered_at,
    {{ to_utc('returned_at') }} as returned_at
from {{ source('thelook', 'orders') }}
where {{ to_utc('created_at') }} < '{{ var("as_of_date") }}'
