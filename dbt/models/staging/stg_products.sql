select
    id as product_id,
    name as product_name,
    brand,
    category,
    lower(department) as department,
    cost,
    retail_price,
    distribution_center_id
from {{ source('thelook', 'products') }}
