"""
Stage 4: customer segments and lifetime value for the thelook store.

RFM: score every customer on recency (how long since they last bought),
frequency (how many purchases) and monetary value (how much they spent), then
name segments from the scores. k-means clustering on the same three numbers is
the data-driven comparison.

Lifetime value: a BG/NBD model predicts how many more purchases each customer
will make, and a Gamma-Gamma model predicts how much each will be worth. Both
are fitted by maximum likelihood with scipy. Before trusting it, the model is
fitted on data up to 26 weeks before the cutoff and checked against what
customers actually bought in those 26 weeks.

Reads mart_customer_rfm and fct_orders from data/warehouse.duckdb (run
`dbt build` first). Saves charts to reports/figures/ and every quoted number
to reports/stage4_numbers.json.

usage: python scripts/05_segments_clv.py
"""
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import optimize, special
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import chart_style
from chart_style import ACCENT, GREY, INK, INK_2

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
AS_OF = pd.Timestamp("2026-10-22")  # matches as_of_date in dbt/dbt_project.yml
HOLDOUT_WEEKS = 26
CLV_WEEKS = 52

SEGMENT_ORDER = ["Champions", "Loyal", "New", "Promising", "At risk", "Slipping", "Lost"]


def save(fig, name):
    chart_style.save(fig, FIG_DIR / name)


def rfm_segments(con, out):
    df = con.sql("select * from mart_customer_rfm").df()
    # quintiles for recency and spend. ranking first breaks ties so every
    # quintile holds a fifth of customers. fewer days since a purchase is better
    df["r"] = 5 - pd.qcut(df.recency_days.rank(method="first"), 5, labels=False)
    df["m"] = 1 + pd.qcut(df.monetary_total.rank(method="first"), 5, labels=False)
    # thelook caps customers at 4 orders and 70% have exactly one, so quintiles
    # would split identical customers at random. the count itself is the score
    df["f"] = df.frequency

    def name(r, f):
        if r >= 4:
            return "Champions" if f >= 3 else "Loyal" if f == 2 else "New"
        if r == 3:
            return "Loyal" if f >= 2 else "Promising"
        return "At risk" if f >= 3 else "Slipping" if f == 2 else "Lost"

    df["segment"] = [name(r, f) for r, f in zip(df.r, df.f)]
    seg = df.groupby("segment").agg(customers=("user_id", "size"),
                                    revenue=("monetary_total", "sum"),
                                    recency_days=("recency_days", "median"),
                                    frequency=("frequency", "mean")).reindex(SEGMENT_ORDER)
    seg["customer_share"] = seg.customers / seg.customers.sum()
    seg["revenue_share"] = seg.revenue / seg.revenue.sum()
    out["rfm"] = {
        "customers": int(len(df)),
        "frequency_counts": {int(k): int(v) for k, v in df.frequency.value_counts().sort_index().items()},
        "segments": {s: {"customers": int(r.customers), "customer_share": round(r.customer_share, 3),
                         "revenue_share": round(r.revenue_share, 3),
                         "median_recency_days": int(r.recency_days),
                         "avg_purchases": round(r.frequency, 2)}
                     for s, r in seg.iterrows()},
    }
    return df, seg


def kmeans_compare(df, out):
    # log for recency and spend because both have long right tails, then
    # standardize so no feature wins just by having bigger numbers
    x = StandardScaler().fit_transform(np.column_stack([
        np.log1p(df.recency_days), df.frequency, np.log(df.monetary_total)]))
    rng = np.random.default_rng(4)
    sample = rng.choice(len(x), size=20_000, replace=False)
    sil = {}
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x[sample])
        sil[k] = silhouette_score(x[sample], labels)
    best_k = max(sil, key=sil.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(x)
    df["cluster"] = km.labels_
    profile = df.groupby("cluster").agg(customers=("user_id", "size"),
                                        median_recency_days=("recency_days", "median"),
                                        avg_purchases=("frequency", "mean"),
                                        median_spend=("monetary_total", "median"))
    out["kmeans"] = {
        "silhouette_by_k": {k: round(v, 3) for k, v in sil.items()},
        "best_k": best_k,
        # 1 = clusters match the named segments exactly, 0 = no better than chance
        "adjusted_rand_vs_rfm": round(adjusted_rand_score(df.segment, df.cluster), 3),
        "clusters": {int(c): {"customers": int(r.customers),
                              "median_recency_days": int(r.median_recency_days),
                              "avg_purchases": round(r.avg_purchases, 2),
                              "median_spend": round(r.median_spend, 1)}
                     for c, r in profile.iterrows()},
    }


def segment_chart(seg):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(seg))
    ax.barh(y - 0.19, seg.customer_share, height=0.36, color=GREY, label="Share of customers")
    ax.barh(y + 0.19, seg.revenue_share, height=0.36, color=ACCENT, label="Share of revenue")
    for i, (c, r) in enumerate(zip(seg.customer_share, seg.revenue_share)):
        ax.text(c + 0.005, i - 0.19, f"{c:.0%}", va="center", color=INK_2, fontsize=9)
        ax.text(r + 0.005, i + 0.19, f"{r:.0%}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, seg.index)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    top = seg.loc[["Champions", "Loyal"]]
    chart_style.title(ax, f"Champions and Loyal customers are {top.customer_share.sum():.0%} of buyers "
                          f"and {top.revenue_share.sum():.0%} of revenue",
                      "RFM segments from recency and number of purchases; revenue is lifetime net revenue")
    save(fig, "14_rfm_segments.png")


# ---- BG/NBD: how many more purchases? ----------------------------------------
#
# each customer buys at their own steady rate while "alive", and after every
# purchase has some chance of quietly leaving for good. rates vary between
# customers following a gamma distribution (r, alpha), and leaving chances
# follow a beta distribution (a, b). the model never sees anyone leave, it
# infers it from how long a customer has been quiet compared with their rate.

def bgnbd_neg_ll(log_params, x, t_x, T):
    r, alpha, a, b = np.exp(log_params)
    common = special.gammaln(r + x) - special.gammaln(r) + r * np.log(alpha)
    still_alive = special.betaln(a, b + x) - special.betaln(a, b) - (r + x) * np.log(alpha + T)
    # only customers with a repeat purchase can have left after their last one
    left = np.where(
        x > 0,
        special.betaln(a + 1, np.maximum(b + x - 1, 1e-9)) - special.betaln(a, b)
        - (r + x) * np.log(alpha + t_x),
        -np.inf,
    )
    return -(common + np.logaddexp(still_alive, left)).sum()


def bgnbd_fit(x, t_x, T):
    res = optimize.minimize(bgnbd_neg_ll, np.zeros(4), args=(x, t_x, T), method="L-BFGS-B",
                            bounds=[(-10, 10)] * 4)
    return np.exp(res.x)


def bgnbd_expected(params, t, x, t_x, T):
    # expected purchases in the next t weeks for a customer with this history
    r, alpha, a, b = params
    hyp = special.hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
    top = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
    return top / (1 + (x > 0) * a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x))


def p_alive(params, x, t_x, T):
    r, alpha, a, b = params
    return 1 / (1 + (x > 0) * a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x))


def history(purchases, end):
    # x = repeat purchases, t_x = weeks from first to last purchase, tenure =
    # weeks from first purchase to the end of the window (T in the BG/NBD
    # paper, renamed because df.T means transpose in pandas)
    p = purchases[purchases.ordered_at < end]
    g = p.groupby("user_id").ordered_at.agg(["min", "max", "size"])
    week = pd.Timedelta(weeks=1)
    return pd.DataFrame({
        "x": g["size"] - 1,
        "t_x": (g["max"] - g["min"]) / week,
        "tenure": (end - g["min"]) / week,
    })


# ---- Gamma-Gamma: how much is each purchase worth? ----------------------------
#
# each customer has their own typical order value, and the model pulls a
# customer's observed average toward the store average, more strongly when
# they've bought only once or twice. it assumes order value doesn't depend
# on how often someone buys, which gets checked before fitting.

def gg_neg_ll(log_params, n, m):
    p, q, v = np.exp(log_params)
    ll = (special.gammaln(p * n + q) - special.gammaln(p * n) - special.gammaln(q)
          + q * np.log(v) + (p * n - 1) * np.log(m) + p * n * np.log(n)
          - (p * n + q) * np.log(n * m + v))
    return -ll.sum()


def gg_fit(n, m):
    res = optimize.minimize(gg_neg_ll, np.zeros(3), args=(n, m), method="L-BFGS-B",
                            bounds=[(-10, 10)] * 3)
    return np.exp(res.x)


def gg_expected(params, n, m):
    p, q, v = params
    return p * (v + n * m) / (p * n + q - 1)


def clv(con, rfm, out):
    purchases = con.sql(
        "select user_id, ordered_at, net_revenue from fct_orders where net_revenue > 0"
    ).df()

    # 1. validate: fit on everything before the holdout, predict the holdout
    cal_end = AS_OF - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    cal = history(purchases, cal_end)
    params = bgnbd_fit(cal.x.to_numpy(), cal.t_x.to_numpy(), cal.tenure.to_numpy())
    cal["predicted"] = bgnbd_expected(params, HOLDOUT_WEEKS, cal.x, cal.t_x, cal.tenure)
    hold = purchases[(purchases.ordered_at >= cal_end) & (purchases.ordered_at < AS_OF)]
    cal["actual"] = hold.groupby("user_id").size().reindex(cal.index, fill_value=0)
    # a simple baseline: keep buying at your past repeat rate
    cal["naive"] = cal.x / cal.tenure * HOLDOUT_WEEKS
    by_x = cal.groupby("x")[["actual", "predicted", "naive"]].mean()

    def top_decile_capture(col):
        # share of all holdout purchases made by the 10% of customers this
        # method ranked highest. tests the ranking, not the total
        top = cal[col].rank(method="first", ascending=False) <= len(cal) / 10
        return round(float(cal.actual[top].sum() / cal.actual.sum()), 3)

    out["clv_validation"] = {
        "calibration_end": str(cal_end.date()),
        "holdout_weeks": HOLDOUT_WEEKS,
        "customers": int(len(cal)),
        "bgnbd_params": dict(zip(["r", "alpha", "a", "b"], [round(float(v), 4) for v in params])),
        "actual_total": int(cal.actual.sum()),
        "predicted_total": round(float(cal.predicted.sum()), 1),
        "naive_total": round(float(cal.naive.sum()), 1),
        "mae_model": round(float((cal.predicted - cal.actual).abs().mean()), 4),
        "mae_naive": round(float((cal.naive - cal.actual).abs().mean()), 4),
        "top_decile_capture_model": top_decile_capture("predicted"),
        "top_decile_capture_naive": top_decile_capture("naive"),
        # BG/NBD assumes each customer's buying rate stays put. stage 3 showed
        # repeat buying rising store-wide, so the totals come out low
        "actual_over_predicted": round(float(cal.actual.sum() / cal.predicted.sum()), 3),
        "by_repeat_purchases": {int(k): {c: round(float(v), 4) for c, v in r.items()}
                                for k, r in by_x.iterrows()},
    }

    # 2. refit on all the data and look a year ahead
    full = history(purchases, AS_OF)
    params = bgnbd_fit(full.x.to_numpy(), full.t_x.to_numpy(), full.tenure.to_numpy())
    full["exp_purchases"] = bgnbd_expected(params, CLV_WEEKS, full.x, full.t_x, full.tenure)
    full["p_alive"] = p_alive(params, full.x, full.t_x, full.tenure)

    value = rfm.set_index("user_id")[["frequency", "monetary_avg", "segment"]]
    full = full.join(value)
    repeaters = full[full.frequency >= 2]
    corr = float(np.corrcoef(repeaters.frequency, repeaters.monetary_avg)[0, 1])
    gg = gg_fit(repeaters.frequency.to_numpy(), repeaters.monetary_avg.to_numpy())
    full["exp_value"] = gg_expected(gg, full.frequency, full.monetary_avg)
    # 12 months of net revenue, no discounting over such a short horizon
    full["clv"] = full.exp_purchases * full.exp_value

    top = full.clv.sort_values(ascending=False)
    by_seg = full.groupby("segment").agg(customers=("clv", "size"), avg_clv=("clv", "mean"),
                                         total_clv=("clv", "sum"), p_alive=("p_alive", "mean"),
                                         exp_purchases=("exp_purchases", "mean")).reindex(SEGMENT_ORDER)
    out["clv"] = {
        "horizon_weeks": CLV_WEEKS,
        "bgnbd_params": dict(zip(["r", "alpha", "a", "b"], [round(float(v), 4) for v in params])),
        "gamma_gamma_params": dict(zip(["p", "q", "v"], [round(float(v), 4) for v in gg])),
        "freq_value_corr_repeaters": round(corr, 3),
        "expected_purchases_total": round(float(full.exp_purchases.sum()), 0),
        "expected_revenue_total": round(float(full.clv.sum()), 0),
        # the same total scaled by how far the model fell short in the holdout.
        # only valid if the upward drift carries on at the same pace
        "expected_revenue_total_holdout_scaled": round(
            float(full.clv.sum() * out["clv_validation"]["actual_over_predicted"]), 0),
        "avg_clv": round(float(full.clv.mean()), 2),
        "top_10pct_share_of_clv": round(float(top.iloc[: len(top) // 10].sum() / top.sum()), 3),
        # p_alive is always 1 for one-time buyers: in BG/NBD nobody can leave
        # before their first repeat purchase, so it only means something for
        # segments made of repeat buyers
        "by_segment": {s: {"customers": int(r.customers), "avg_clv": round(r.avg_clv, 2),
                           "share_of_clv": round(r.total_clv / full.clv.sum(), 3),
                           "avg_p_alive": round(r.p_alive, 3),
                           "avg_expected_purchases": round(r.exp_purchases, 3)}
                       for s, r in by_seg.iterrows()},
    }
    return cal, by_x, by_seg


def validation_chart(by_x, out):
    v = out["clv_validation"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(by_x))
    w = 0.26
    ax.bar(x - w, by_x.actual, width=w, color=INK_2, label="Actual")
    ax.bar(x, by_x.predicted, width=w, color=ACCENT, label="BG/NBD prediction")
    ax.bar(x + w, by_x.naive, width=w, color=GREY, label="Past rate continues")
    ax.set_xticks(x, [str(i) for i in by_x.index])
    ax.set_xlabel(f"Repeat purchases before {v['calibration_end']}")
    ax.set_ylabel(f"Purchases per customer, next {HOLDOUT_WEEKS} weeks")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    # nobody with 4 purchases bought again: the generator stops at 4 orders
    ax.text(len(by_x) - 1 - w / 2, 0.012, "thelook caps\ncustomers at\n4 orders", ha="right",
            va="bottom", color=INK_2, fontsize=8.5)
    chart_style.title(ax, f"Past buying barely predicts future buying, and the model ran "
                          f"{1 - 1 / v['actual_over_predicted']:.0%} low",
                      f"Fitted before {v['calibration_end']}, checked on {HOLDOUT_WEEKS} weeks after: "
                      f"{v['predicted_total']:,.0f} predicted vs {v['actual_total']:,} actual purchases")
    save(fig, "15_clv_validation.png")


def clv_chart(by_seg, out):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(by_seg))
    best = by_seg.avg_clv.idxmax()
    colors = [ACCENT if s == best else GREY for s in by_seg.index]
    ax.barh(y, by_seg.avg_clv, color=colors, height=0.6)
    for i, (s, r) in enumerate(by_seg.iterrows()):
        ax.text(r.avg_clv + 0.5, i, f"\\${r.avg_clv:,.0f}   ({int(r.customers):,} customers)",
                va="center", color=INK if s == best else INK_2, fontsize=9.5)
    ax.set_yticks(y, by_seg.index)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, by_seg.avg_clv.max() * 1.45)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"\\${v:,.0f}"))
    ax.set_xlabel("Predicted net revenue per customer, next 12 months")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    c = out["clv"]
    # a $ pair in matplotlib text switches to math mode, so they're escaped
    chart_style.title(ax, f"A {best} customer is worth \\${by_seg.avg_clv.max():,.0f} over the next year, "
                          f"a Lost one \\${by_seg.loc['Lost', 'avg_clv']:,.0f}",
                      f"BG/NBD + Gamma-Gamma. A rough guide only: the model ran "
                      f"{1 - 1 / out['clv_validation']['actual_over_predicted']:.0%} low in the holdout")
    save(fig, "16_clv_by_segment.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
    out = {"as_of_date": str(AS_OF.date())}
    rfm, seg = rfm_segments(con, out)
    kmeans_compare(rfm, out)
    segment_chart(seg)
    cal, by_x, by_seg = clv(con, rfm, out)
    validation_chart(by_x, out)
    clv_chart(by_seg, out)
    (ROOT / "reports" / "stage4_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
