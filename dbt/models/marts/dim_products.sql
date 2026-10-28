select
    product_id,
    product_name,
    brand,
    category,
    department,
    cost,
    retail_price,
    retail_price - cost as unit_margin,
    (retail_price - cost) / nullif(retail_price, 0) as margin_pct
from {{ ref('stg_products') }}
