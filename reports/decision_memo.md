# Decision memo: where to put effort next quarter

**From:** Amal Pushp
**Data:** thelook store data to 2026-10-22, Criteo ad experiment (14M users)
**Dashboard:** https://apushp.github.io/ecommerce-product-analytics/ (Tableau version: https://public.tableau.com/app/profile/amal.pushp2875/viz/ecommerce_analytics_17934643228960/E-commerceProductAnalytics)

## Summary

The store is growing fast, and almost all of it comes from more buyers. The
biggest leak is checkout: fewer than half of carts become orders. My three
recommendations are to fix the cart before spending more on traffic, to work
on second purchases, and to target ads by who they actually move, not by
who's likely to buy anyway.

## What the numbers say

| Area | Finding | Confidence |
| --- | --- | --- |
| Growth | Active buyers up 111% in a year; that explains 99% of revenue growth. Order value is flat. | High |
| Funnel | 63% of sessions add to cart, but only 41% of carts become orders. | High |
| Channels | No traffic channel converts better than another (p = 0.76). | High, but the data is synthetic |
| Repeat buying | 25% of customers order again within a year. Repeat buying is rising store-wide. | High |
| Segments | Champions and Loyal are 22% of buyers and 38% of revenue. | High |
| Lifetime value | A Champion is worth about $34 over the next year, a Lost customer about $9. | Low: the model ran 35% low on a holdout |
| Ads | Ads lift visits about 17%, not the 27% a plain comparison shows. | High |
| Ad targeting | The top 10% of users by predicted uplift hold 76% of the extra visits ads cause. | High |
| Next quarter | About $1.62M net revenue, 80% range $1.22M to $2.16M. | Medium: every model ran low in backtests |

## Recommendations

**1. Fix the cart before buying more traffic.** Every extra session hits the
same 41% cart-to-order rate. Raising it to 45% would mean about 10% more
orders from the traffic we already have. I'd test checkout changes (shipping
cost shown earlier, guest checkout, fewer steps) as A/B tests: plan the
sample size up front, check the split and balance, and don't stop early.
Peeking 20 times turned a 5% false-win rate into 26% in my simulations.

**2. Don't move budget between channels based on conversion.** YouTube,
Adwords, Facebook, email and organic traffic all convert at the same rate here. Channel
decisions should come from cost per acquired buyer, which this data doesn't
have.

**3. Work on the second purchase.** Only 25% of customers come back within a
year, and New plus Promising customers (one purchase each) are 38% of the
base. A second-order offer aimed at them is the clearest retention play.
Champions and Loyal customers should get recognition rather than discounts,
since they already buy.

**4. Target ads by uplift when the budget is tight.** With budget for 10% of
users, uplift targeting reaches 76% of the extra visits ads can cause,
against 66% for the usual "likely buyers" targeting. Above 30% of users the
two approaches converge, and ads shown to the bottom half buy almost nothing
(+0.04 points).

**5. Report ad results with the adjustment.** A plain test-vs-control
comparison overstated the lift by more than half (27% vs 17%), because the
groups weren't fully comparable. Comparing users who saw ads with users who
didn't overstates it about twice over. Any ad readout should come with a
balance check.

**6. Plan the quarter on the range, not the point.** $1.62M is the middle.
Every forecasting model under-shot in backtests while growth sped up, so the
upper half of the range is more likely than the lower half if the trend
holds.

## Caveats

- thelook is generated data. Some patterns (equal channel conversion,
  customers stopping at 4 orders, no seasonality) come from the generator, so
  recommendations 2 and 3 would need checking on real data.
- The ad findings come from Criteo's experiment, a different business from
  the store. They show the method and the size of the mistakes it avoids.
- Lifetime value is a rough ranking here, not a forecast.
