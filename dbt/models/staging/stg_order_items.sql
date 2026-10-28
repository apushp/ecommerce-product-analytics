-- item timestamps in thelook drift up to a few days from their order's timestamp,
-- so the cutoff follows the order date to keep every item attached to an order
select
    id as order_item_id,
    order_id,
    user_id,
    product_id,
    inventory_item_id,
    lower(status) as status,
    sale_price,
    {{ to_utc('created_at') }} as item_created_at,
    {{ to_utc('returned_at') }} as returned_at
from {{ source('thelook', 'order_items') }}
where order_id in (select order_id from {{ ref('stg_orders') }})
