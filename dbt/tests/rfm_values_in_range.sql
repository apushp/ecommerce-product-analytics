-- recency can't be negative (no purchases after the cutoff), every customer
-- here bought at least once, and the last purchase can't come before the first
select user_id
from {{ ref('mart_customer_rfm') }}
where recency_days < 0
   or frequency < 1
   or monetary_avg <= 0
   or recency_days > tenure_days
