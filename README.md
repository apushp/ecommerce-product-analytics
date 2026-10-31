# E-commerce product analytics

Analytics for an online clothing store, built the way a product analytics team would: raw data pulled from BigQuery, modeled and tested in dbt, then analyzed in Python. The question running through it: where does the store lose customers, and what actually brings them back?

**Work in progress.** Stages 1 to 3 (warehouse, metrics, funnel and retention) stage 5 (A/B testing) and stage 6 (uplift modeling) are done. Segmentation, forecasting and a dashboard come next.

## What I found so far

![Active buyers per month](reports/figures/01_north_star.png)

- **Active buyers grew 111% in a year** (2,061 in Sep 2025 to 4,351 in Sep 2026). That alone explains 99% of revenue growth. Order value stayed flat at about $84 to $88, and orders per buyer barely moved.
- **The cart is the biggest leak.** 63% of sessions add something to cart, but only 41% of those carts become orders.
- **No traffic channel converts better than another.** Chi-square p = 0.76, Cramér's V = 0.0017.
- **25% of customers order again within a year** (Kaplan-Meier, which accounts for recent customers who haven't had time to come back). The simple "2+ orders" rate says 37%, and that number mixes long-time customers with brand new ones.
- **Repeat buying is rising store-wide**, from 1.7% to 2.8% of customers ordering again each month over the past year. It shows up along the diagonals of the cohort table, not down its columns.

## A/B test: did Criteo's ads work?

![Lift estimates](reports/figures/09_lift.png)

Criteo held back 15% of 14M users from an ad campaign. I read the test the way an experimentation team would, checking the setup before looking at the result.

- **The split checks out, but the groups aren't quite comparable.** Treatment is 85.000% of users, exactly as designed, so the sample ratio mismatch check passes. Every pre-treatment feature also passes the usual 0.1 standardized-difference rule. But the biggest gap is 65 times what random assignment would produce. The file turns out to be long runs of treated rows and long runs of control rows with different base visit rates, so the control group isn't drawn from the same mix of users.
- **Ads lifted visits by about 17%, not the 27% a plain comparison shows.** Two independent adjustments that only use pre-treatment features agree: CUPAC (predict each user's outcome, then regress it out) gives +16.8% (95% CI 16.0% to 17.5%), and inverse propensity weighting gives +19.1%. Conversions rose about 50 to 55%. CUPAC also cut the variance of the visit estimate by 31%, which is like running the test on 44% more users.
- **The 85/15 split was expensive.** It needs 1.96x the users of a 50/50 test for the same precision. Even so, 14M users could detect a 1.1% lift in visits.
- **Peeking breaks the 5% promise.** In 1,000 A/A tests (control split against itself), 4% came out significant when checked once. Checking 20 times and stopping at the first p < 0.05 gave a false win 26% of the time.
- **Only 3.6% of the treatment group actually saw an ad.** Comparing users who saw one with users who didn't suggests a 38-point jump in visit rate. Using random assignment as an instrument, the real effect of seeing the ad is 17.8 points. The people who got shown ads would have visited 24% of the time anyway, against 3.8% for an average user.

## Uplift: who should see an ad?

![Qini curves](reports/figures/12_qini.png)

The average effect hides who the ads actually move. I trained models on 4M users to predict how much an ad changes each person's chance of visiting, then scored 7M users they never saw.

- **The effect is packed into a few users.** The top 10% by predicted uplift hold 76% of all the extra visits the ads caused, and their visit rate rises 5.8 points. The bottom half gains 0.04 points.
- **Uplift targeting wins when the budget is tight.** Showing ads to the top 10% by uplift captures 76% of the extra visits, against 66% when targeting users most likely to visit (a 10-point gain, bootstrap 95% CI 7 to 13). From 30% of users up, both approaches capture about 94%, because in this data the people ads move are mostly people already likely to visit.
- **The S-learner beat the T-learner.** The control group is only 15% of users, so the T-learner's separate control model had less to learn from and came out noisier.
- Every comparison is weighted by inverse propensity, carrying over the stage 5 fix for the groups not being perfectly comparable.

More charts are in [`reports/figures/`](reports/figures/). Every number above is in the `reports/stage*_numbers.json` files.

## Data

| Dataset | What it is | Used for |
| --- | --- | --- |
| [thelook_ecommerce](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce) | Google's public BigQuery dataset of a fictional clothing store: 97k users, 120k orders, 2.4M web events | warehouse, metrics, funnel, cohorts |
| [Criteo Uplift v2.1](https://huggingface.co/datasets/criteo/criteo-uplift) | a real randomized ad experiment, about 14M users | A/B testing and uplift modeling |

thelook is synthetic, so some patterns are generator artifacts rather than real behavior. Anonymous sessions never buy, and every channel converts at the same rate. I kept these as documented limitations. The A/B testing uses Criteo's real experiment for this reason.

Data is cut off at **2026-10-22**. thelook's daily refresh piles about 13x the normal number of signups into the last week before a pull. Left in, it looks like sales tripled overnight.

## How it's built

```
BigQuery (public data)
   └─ scripts/01_extract_thelook.py   pull 7 tables once, save as parquet
        └─ dbt on DuckDB              staging → intermediate → marts, 49 tests
             └─ scripts/02_...py      metrics, funnel, stats tests, charts

Criteo csv (Hugging Face)
   └─ scripts/03_ab_test_criteo.py    SRM, balance, power, lift, CUPAC, A/A, exposure
        └─ scripts/04_uplift_criteo.py    S/T-learners, Qini curves, bootstrap
```

- `dbt/models/staging/`: one cleaned model per raw table. Timestamps go to UTC, personal data is dropped, and the date cutoff is applied.
- `dbt/models/marts/`: `fct_sessions`, `fct_orders`, `fct_order_items`, `dim_users`, `dim_products`, `mart_monthly_kpis`, `mart_cohort_retention`.
- Revenue rules are defined once in `fct_order_items`: gross revenue excludes cancelled items, and net revenue also excludes returns.
- Two business-rule tests check that order revenue matches item revenue, and that net revenue is never above gross.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

gcloud auth application-default login
python scripts/01_extract_thelook.py --project <your-gcp-project>

cd dbt && dbt build --profiles-dir . && cd ..
python scripts/02_metrics_funnel_retention.py

# Criteo: download criteo-uplift-v2.1.csv.gz from Hugging Face into data/raw/
python scripts/03_ab_test_criteo.py   # about 2 minutes, needs ~4GB of memory
python scripts/04_uplift_criteo.py    # about 5 minutes
```

The BigQuery sandbox is free, and no credit card is needed. Reading the tables uses `list_rows`, which doesn't count against the query quota.
