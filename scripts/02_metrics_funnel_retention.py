"""
Stage 2 and 3 analysis: north star trend, what drove revenue growth, the
session funnel, whether traffic channels convert differently, how long
customers take to come back, and cohort retention.

Reads the dbt marts in data/warehouse.duckdb (run `dbt build` first), saves
charts to reports/figures/ and every number quoted in the write-up to
reports/stage3_numbers.json.

usage: python scripts/02_metrics_funnel_retention.py
"""
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
AS_OF = "2026-10-22"  # matches as_of_date in dbt/dbt_project.yml

# one accent, everything else grey, so the eye goes to what the title is about
ACCENT = "#2a78d6"
GREY = "#b4b2ab"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
})


def title(ax, headline, sub):
    ax.set_title(headline, pad=26)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_2, fontsize=10)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)


def north_star(con, out):
    df = con.sql("""
        select order_month, new_buyers, returning_buyers, active_buyers
        from mart_monthly_kpis
        where is_complete_month and order_month >= '2023-01-01'
        order by order_month
    """).df()
    first, last = df.iloc[0], df.iloc[-1]
    year_ago = df.iloc[-13]
    growth = last.active_buyers / year_ago.active_buyers - 1
    out["north_star"] = {
        "latest_month": str(last.order_month.date()),
        "active_buyers": int(last.active_buyers),
        "active_buyers_year_ago": int(year_ago.active_buyers),
        "yoy_growth": round(growth, 3),
        "returning_share_latest": round(last.returning_buyers / last.active_buyers, 3),
        "returning_share_first": round(first.returning_buyers / first.active_buyers, 3),
    }

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = df.order_month
    ax.fill_between(x, 0, df.new_buyers, color=GREY, linewidth=0, step=None)
    ax.fill_between(x, df.new_buyers, df.active_buyers, color=ACCENT, alpha=0.85, linewidth=0)
    ax.text(x.iloc[-1], last.new_buyers / 2, "  New buyers", color=INK_2, va="center")
    ax.text(x.iloc[-1], last.new_buyers + last.returning_buyers / 2, "  Returning buyers",
            color=ACCENT, va="center", fontweight="bold")
    ax.set_xlim(x.iloc[0], x.iloc[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, None)
    ax.set_ylabel("Active buyers per month")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    title(ax, f"Active buyers grew {growth:.0%} in a year",
          f"Customers with at least one non-cancelled order, {x.iloc[0]:%b %Y} to {x.iloc[-1]:%b %Y}")
    fig.subplots_adjust(right=0.82)
    save(fig, "01_north_star.png")


def revenue_drivers(con, out):
    # gross_revenue = active_buyers * orders_per_buyer * avg_order_value, so
    # log changes of the three drivers add up exactly to the log change in revenue
    df = con.sql("""
        select order_month, gross_revenue, active_buyers, orders_per_buyer, avg_order_value
        from mart_monthly_kpis
        where is_complete_month
        order by order_month desc
        limit 13
    """).df()
    now, then = df.iloc[0], df.iloc[-1]
    drivers = {
        "Active buyers": np.log(now.active_buyers / then.active_buyers),
        "Orders per buyer": np.log(now.orders_per_buyer / then.orders_per_buyer),
        "Avg order value": np.log(now.avg_order_value / then.avg_order_value),
    }
    total = np.log(now.gross_revenue / then.gross_revenue)
    shares = {k: v / total for k, v in drivers.items()}
    out["revenue_drivers"] = {
        "from_month": str(then.order_month.date()),
        "to_month": str(now.order_month.date()),
        "revenue_growth": round(now.gross_revenue / then.gross_revenue - 1, 3),
        "driver_change": {k: round(float(np.exp(v) - 1), 3) for k, v in drivers.items()},
        "share_of_growth": {k: round(float(v), 3) for k, v in shares.items()},
    }

    fig, ax = plt.subplots(figsize=(9, 3.4))
    names = list(drivers)[::-1]
    vals = [shares[n] for n in names]
    colors = [ACCENT if n == "Active buyers" else GREY for n in names]
    ax.barh(names, vals, color=colors, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.0%}", va="center",
                ha="left" if v >= 0 else "right", color=INK)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.grid(False)
    ax.set_xticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(min(vals) - 0.15, max(vals) + 0.15)
    rev = now.gross_revenue / then.gross_revenue - 1
    title(ax, f"More buyers explain {shares['Active buyers']:.0%} of revenue growth",
          f"Share of the {rev:.0%} rise in gross revenue, {then.order_month:%b %Y} to {now.order_month:%b %Y} (log decomposition)")
    save(fig, "02_revenue_drivers.png")


def funnel(con, out):
    df = con.sql("""
        select
            count(*) as sessions,
            count(*) filter (where viewed_product) as viewed,
            count(*) filter (where added_to_cart) as carted,
            count(*) filter (where purchased) as bought
        from fct_sessions
    """).df().iloc[0]
    steps = [("Sessions", df.sessions), ("Viewed a product", df.viewed),
             ("Added to cart", df.carted), ("Purchased", df.bought)]
    out["funnel"] = {name: int(n) for name, n in steps}
    out["funnel"]["cart_to_purchase"] = round(df.bought / df.carted, 3)
    out["funnel"]["session_to_purchase"] = round(df.bought / df.sessions, 3)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = [s[0] for s in steps][::-1]
    counts = [s[1] for s in steps][::-1]
    colors = [GREY] * 4
    # the biggest drop is the one worth looking at
    drops = [steps[i + 1][1] / steps[i][1] for i in range(3)]
    worst = int(np.argmin(drops)) + 1
    colors[3 - worst] = ACCENT
    ax.barh(labels, counts, color=colors, height=0.6)
    for i, (lab, n) in enumerate(zip(labels, counts)):
        ax.text(n + df.sessions * 0.01, i, f"{n:,.0f}", va="center", color=INK)
    for i in range(1, 4):
        ax.text(df.sessions * 1.17, 3 - i + 0.5, f"{drops[i - 1]:.0%} carry on",
                va="center", ha="right", color=INK_2, fontsize=9.5)
    ax.grid(False)
    ax.set_xticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, df.sessions * 1.18)
    title(ax, f"Cart to purchase is the biggest drop: {drops[2]:.0%} of carts become orders",
          f"All web sessions up to {AS_OF}")
    save(fig, "03_funnel.png")


def channel_test(con, out):
    df = con.sql("""
        select traffic_source, count(*) as sessions, count(*) filter (where purchased) as purchases
        from fct_sessions
        group by 1
        order by purchases / count(*) desc
    """).df()
    table = np.column_stack([df.purchases, df.sessions - df.purchases])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    # with 680k sessions almost any gap is "significant", so the effect size
    # says whether the gap is big enough to matter
    cramers_v = np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    rate = df.purchases / df.sessions
    se = np.sqrt(rate * (1 - rate) / df.sessions)
    out["channel_test"] = {
        "chi2": round(float(chi2), 2), "dof": int(dof), "p_value": round(float(p), 4),
        "cramers_v": round(float(cramers_v), 4),
        "rates": {s: round(float(r), 4) for s, r in zip(df.traffic_source, rate)},
    }

    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(df))[::-1]
    ax.errorbar(rate, y, xerr=1.96 * se, fmt="o", color=ACCENT, ecolor=ACCENT,
                elinewidth=2, capsize=0, markersize=7)
    overall = df.purchases.sum() / df.sessions.sum()
    ax.axvline(overall, color=INK_2, linestyle="--", linewidth=1)
    ax.text(overall, y.max() + 0.6, f" all sessions {overall:.1%}", color=INK_2, fontsize=9.5)
    ax.set_yticks(y, [f"{s}  ({n:,} sessions)" for s, n in zip(df.traffic_source, df.sessions)])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.set_xlabel("Sessions ending in a purchase (95% confidence interval)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, y.max() + 1)
    title(ax, "No traffic channel converts better than another",
          f"chi-square = {chi2:.1f}, p = {p:.2f}, Cramér's V = {cramers_v:.4f}")
    save(fig, "04_channel_conversion.png")


def time_to_second_order(con, out):
    # every customer is followed from their first order until their second one,
    # or until the data ends if they haven't come back yet (censored)
    df = con.sql(f"""
        with firsts as (
            select user_id, ordered_at as first_at from fct_orders where order_number = 1
        ),
        seconds as (
            select user_id, ordered_at as second_at from fct_orders where order_number = 2
        )
        select
            date_diff('day', f.first_at, coalesce(s.second_at, timestamp '{AS_OF}')) as days,
            s.second_at is not null as came_back
        from firsts f
        left join seconds s using (user_id)
    """).df()

    # Kaplan-Meier: at each day, the share of customers still waiting who come back that day
    days = df.days.to_numpy()
    event = df.came_back.to_numpy()
    times = np.unique(days[event])
    at_risk = np.array([(days >= t).sum() for t in times])
    events = np.array([((days == t) & event).sum() for t in times])
    surv = np.cumprod(1 - events / at_risk)
    # Greenwood's formula for the standard error
    var = surv ** 2 * np.cumsum(events / (at_risk * (at_risk - events)))
    se = np.sqrt(var)
    returned = 1 - surv

    def at(d):
        i = np.searchsorted(times, d, side="right") - 1
        return float(returned[i]) if i >= 0 else 0.0

    naive = event.mean()
    out["time_to_second_order"] = {
        "customers": int(len(df)),
        "came_back": int(event.sum()),
        "naive_repeat_rate": round(float(naive), 3),
        "km_returned_by": {f"{d}_days": round(at(d), 3) for d in (30, 90, 180, 365, 730)},
    }

    fig, ax = plt.subplots(figsize=(9, 4.6))
    keep = times <= 1095
    ax.step(times[keep], returned[keep], where="post", color=ACCENT, linewidth=2)
    ax.fill_between(times[keep], (returned - 1.96 * se)[keep], (returned + 1.96 * se)[keep],
                    step="post", color=ACCENT, alpha=0.15, linewidth=0)
    for d in (90, 365):
        ax.plot(d, at(d), "o", color=ACCENT, markersize=7)
        ax.annotate(f"{at(d):.0%} by day {d}", (d, at(d)), xytext=(8, -14),
                    textcoords="offset points", color=INK)
    ax.axhline(naive, color=INK_2, linestyle="--", linewidth=1)
    ax.text(1095, naive, f"simple repeat rate {naive:.0%}\n(counts recent buyers as gone)  ",
            ha="right", va="bottom", color=INK_2, fontsize=9.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlim(0, 1095)
    ax.set_ylim(0, None)
    ax.set_xlabel("Days since first order")
    ax.set_ylabel("Customers who placed a second order")
    title(ax, f"{at(365):.0%} of customers order again within a year",
          "Kaplan-Meier estimate with 95% band; customers still inside their window are censored, not counted as lost")
    save(fig, "05_time_to_second_order.png")


def cohort_heatmap(con, out):
    df = con.sql(f"""
        select cohort_month, months_since_first, retention_rate, cohort_size
        from mart_cohort_retention
        where cohort_month between '2025-10-01' and '2026-08-01'
          and months_since_first between 1 and 12
    """).df()
    grid = df.pivot(index="cohort_month", columns="months_since_first", values="retention_rate")
    grid = grid.reindex(columns=range(1, 13))
    sizes = df.groupby("cohort_month").cohort_size.first()
    # a cohort can only be measured up to the last full month before the cutoff
    last_full = np.datetime64("2026-09-01")
    for cm in grid.index:
        for k in grid.columns:
            if np.datetime64(cm, "M") + np.timedelta64(k, "M") > last_full:
                grid.loc[cm, k] = np.nan
    # reading the table along its diagonals: every cell on one diagonal is the
    # same calendar month, so a darker diagonal means the whole store had a
    # better month for repeat orders, whatever the cohort
    by_calendar = con.sql("""
        select cohort_month + to_months(months_since_first) as calendar_month,
               sum(active_customers) / sum(cohort_size) as repeat_rate
        from mart_cohort_retention
        where months_since_first >= 1
          and cohort_month + to_months(months_since_first) < '2026-10-01'
        group by 1
        order by 1
    """).df().tail(13)
    vals = grid.to_numpy(dtype=float)
    out["cohorts"] = {
        "repeat_rate_by_calendar_month": {
            f"{m:%Y-%m}": round(float(r), 4) for m, r in zip(by_calendar.calendar_month, by_calendar.repeat_rate)
        },
        "cohorts": f"{grid.index[0]:%Y-%m} to {grid.index[-1]:%Y-%m}",
        "mean_month_1": round(float(grid[1].mean()), 4),
        "min_cell": round(float(np.nanmin(vals)), 4),
        "max_cell": round(float(np.nanmax(vals)), 4),
        "mean_all": round(float(np.nanmean(vals)), 4),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    data = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eef4fc", ACCENT])
    cmap.set_bad(SURFACE)
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=0.05)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not data.mask[i, j]:
                ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=8.5,
                        color="white" if data[i, j] > 0.035 else INK)
    ax.set_xticks(range(12), range(1, 13))
    ax.set_yticks(range(len(grid)), [f"{c:%b %Y}  (n={sizes[c]:,})" for c in grid.index])
    ax.set_xlabel("Months since first order")
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    then, now = by_calendar.repeat_rate.iloc[0], by_calendar.repeat_rate.iloc[-1]
    title(ax, f"Repeat buying is rising store-wide: {then:.1%} to {now:.1%} a month in a year",
          "Share of each first-order cohort ordering that month; each diagonal is one calendar month")
    save(fig, "06_cohort_retention.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
    out = {"as_of_date": AS_OF}
    north_star(con, out)
    revenue_drivers(con, out)
    funnel(con, out)
    channel_test(con, out)
    time_to_second_order(con, out)
    cohort_heatmap(con, out)
    (ROOT / "reports" / "stage3_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
