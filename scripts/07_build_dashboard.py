"""
Stage 8: builds the project dashboard as one static web page.

Pulls the monthly KPIs and weekly revenue from the warehouse and every other
number from the reports/stage*_numbers.json files, so the page always matches
the analysis. Writes docs/index.html, which GitHub Pages serves. It's static
HTML with interactive Plotly charts (loaded from Plotly's CDN), so anyone
can open it without installing anything.

Run the stage scripts and `dbt build` first.

usage: python scripts/07_build_dashboard.py
"""
import html
import json
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "index.html"
REPO = "https://github.com/apushp/ecommerce-product-analytics"

# same palette as the matplotlib charts
ACCENT = "#2a78d6"
ACCENT_SOFT = "rgba(42,120,214,0.30)"
ACCENT_FAINT = "rgba(42,120,214,0.14)"
GREY = "#b4b2ab"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def numbers(stage):
    return json.loads((ROOT / "reports" / f"stage{stage}_numbers.json").read_text())


def style(fig, height=320, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK_2),
        hoverlabel=dict(font=dict(family=FONT, size=13)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GREY, ticks="", automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GREY, automargin=True)
    return fig


def chart_html(fig, first=False):
    # plotly.js is pulled from the CDN once, by the first chart on the page
    # width 100% so each chart fills its card instead of plotly's fixed 700px
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                       default_width="100%", default_height=f"{fig.layout.height}px",
                       config={"displayModeBar": False, "responsive": True})


def buyers_chart(con):
    df = con.sql("""
        select order_month, new_buyers, returning_buyers, active_buyers
        from mart_monthly_kpis
        where is_complete_month and order_month >= '2023-01-01'
        order by order_month
    """).df()
    fig = go.Figure()
    fig.add_bar(x=df.order_month, y=df.new_buyers, name="New buyers", marker_color=GREY,
                hovertemplate="%{x|%b %Y}<br>New: %{y:,}<extra></extra>")
    fig.add_bar(x=df.order_month, y=df.returning_buyers, name="Returning buyers",
                marker_color=ACCENT,
                hovertemplate="%{x|%b %Y}<br>Returning: %{y:,}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.15)
    fig.update_yaxes(tickformat=",")
    return style(fig, legend=True)


def forecast_chart(con, s7):
    daily = con.sql("""
        select date_trunc('day', ordered_at)::date as day, sum(net_revenue) as revenue
        from fct_orders group by 1
    """).df()
    s = daily.set_index(pd.to_datetime(daily.day)).revenue.asfreq("D", fill_value=0)
    weekly = s.resample("W-SUN").sum()
    weekly = weekly[weekly.index <= pd.Timestamp(s7["last_full_week"])].iloc[-60:]
    fc = s7["forecast"]
    dates = pd.to_datetime(list(fc["weekly_median"]))
    med = list(fc["weekly_median"].values())
    lo80, hi80 = zip(*fc["weekly_80"])
    lo95, hi95 = zip(*fc["weekly_95"])
    # the fan starts from the last actual week so the two lines join up
    x = [weekly.index[-1], *dates]
    start = weekly.iloc[-1]
    fig = go.Figure()
    for lo, hi, color, name in ((lo95, hi95, ACCENT_FAINT, "95% range"),
                                (lo80, hi80, ACCENT_SOFT, "80% range")):
        fig.add_scatter(x=x + x[::-1], y=[start, *hi] + [start, *lo][::-1], fill="toself",
                        fillcolor=color, line=dict(width=0), name=name, hoverinfo="skip")
    fig.add_scatter(x=weekly.index, y=weekly, name="Actual", line=dict(color=INK_2, width=2),
                    hovertemplate="Week ending %{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=x, y=[start, *med], name="Forecast (median)",
                    line=dict(color=ACCENT, width=2.5),
                    hovertemplate="Week ending %{x|%b %d, %Y}<br>$%{y:,.0f} forecast<extra></extra>")
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
    return style(fig, legend=True)


def funnel_chart(s3):
    f = s3["funnel"]
    steps = ["Sessions", "Viewed a product", "Added to cart", "Purchased"]
    vals = [f[k] for k in steps]
    colors = [GREY, GREY, GREY, ACCENT]
    text = [f"{v:,}" if i == 0 else f"{v:,}  ({v / vals[i - 1]:.0%} carry on)"
            for i, v in enumerate(vals)]
    fig = go.Figure(go.Bar(x=vals, y=steps, orientation="h", marker_color=colors, text=text,
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(visible=False, range=[0, max(vals) * 1.7])
    return style(fig, height=260)


def segment_chart(s4):
    seg = s4["rfm"]["segments"]
    names = list(seg)
    fig = go.Figure()
    fig.add_bar(y=names, x=[seg[n]["customer_share"] for n in names], orientation="h",
                name="Share of customers", marker_color=GREY,
                hovertemplate="%{y}: %{x:.0%} of customers<extra></extra>")
    fig.add_bar(y=names, x=[seg[n]["revenue_share"] for n in names], orientation="h",
                name="Share of revenue", marker_color=ACCENT,
                hovertemplate="%{y}: %{x:.0%} of revenue<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID)
    return style(fig, height=340, legend=True)


def lift_chart(s5):
    rows = [("Visits", "visit"), ("Conversions", "conversion")]
    fig = go.Figure()
    # numeric rows with a small offset, so the two estimates for one metric
    # sit just above and below each other instead of on top of each other
    for src, offset, color, name in ((s5["lift"], 0.18, GREY, "Plain comparison"),
                                     (s5["cupac"], -0.18, ACCENT, "Adjusted (CUPAC)")):
        x = [src[m]["rel_lift"] for _, m in rows]
        lo = [src[m]["rel_lift"] - src[m]["rel_ci"][0] for _, m in rows]
        hi = [src[m]["rel_ci"][1] - src[m]["rel_lift"] for _, m in rows]
        fig.add_scatter(x=x, y=[i + offset for i in range(len(rows))], mode="markers", name=name,
                        marker=dict(color=color, size=11),
                        error_x=dict(type="data", array=hi, arrayminus=lo, color=color,
                                     thickness=2.5, width=0),
                        customdata=[r[0] for r in rows],
                        hovertemplate="%{customdata}: %{x:+.1%}<extra>" + name + "</extra>")
    fig.update_yaxes(showgrid=False, tickvals=list(range(len(rows))),
                     ticktext=[r[0] for r in rows], range=[1.6, -0.6])
    fig.update_xaxes(tickformat=".0%", range=[0, 0.8], showgrid=True, gridcolor=GRID,
                     title=dict(text="Lift from ads, with 95% CI", font=dict(size=12)))
    return style(fig, height=230, legend=True)


def qini_chart(s6):
    pol = s6["policy"]
    ks = [0, 10, 20, 30, 50, 100]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Random targeting",
                    line=dict(color=INK_2, width=1, dash="dash"), hoverinfo="skip")
    for model, color, label in (("Response model", GREY, "Target likely visitors"),
                                (s6["best_model"], ACCENT, f"Uplift ({s6['best_model']})")):
        y = [0] + [pol[model][f"share_of_incremental_at_{k}"] for k in ks[1:-1]] + [1]
        fig.add_scatter(x=[k / 100 for k in ks], y=y, name=label, mode="lines+markers",
                        line=dict(color=color, width=2.5), marker=dict(size=7),
                        hovertemplate="Top %{x:.0%} of users: %{y:.0%} of extra visits"
                                      "<extra>" + label + "</extra>")
    fig.update_xaxes(tickformat=".0%", title=dict(text="Share of users shown ads",
                                                  font=dict(size=12)))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05],
                     title=dict(text="Share of extra visits", font=dict(size=12)))
    return style(fig, height=320, legend=True)


def tile(label, value, note):
    return (f'<div class="tile"><div class="tile-label">{html.escape(label)}</div>'
            f'<div class="tile-value">{html.escape(value)}</div>'
            f'<div class="tile-note">{html.escape(note)}</div></div>')


def segment_table(s4):
    seg, clv = s4["rfm"]["segments"], s4["clv"]["by_segment"]
    rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{seg[n]['customers']:,}</td>"
        f"<td>{seg[n]['revenue_share']:.0%}</td><td>${clv[n]['avg_clv']:,.0f}</td></tr>"
        for n in seg)
    return ("<table><thead><tr><th>Segment</th><th>Customers</th><th>Revenue share</th>"
            "<th>12-month value</th></tr></thead><tbody>" + rows + "</tbody></table>")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E-commerce Product Analytics</title>
<meta name="description" content="Funnel, retention, segments, A/B testing, uplift and forecasting for an online clothing store and a real ad experiment.">
<style>
:root {{
  color-scheme: light;
  --accent: {accent}; --ink: {ink}; --ink-2: {ink2}; --grey: {grey};
  --line: {grid}; --surface: #fcfcfb; --card: #ffffff;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--surface); color: var(--ink); font-family: {font};
  font-size: 15px; line-height: 1.5; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px 16px 64px; }}
header h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: -0.01em; }}
header p {{ margin: 0; color: var(--ink-2); max-width: 760px; }}
header a, footer a {{ color: var(--accent); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px;
  margin: 24px 0 8px; }}
.tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }}
.tile-label {{ color: var(--ink-2); font-size: 13px; }}
.tile-value {{ font-size: 26px; font-weight: 700; margin: 2px 0; font-variant-numeric: tabular-nums; }}
.tile-note {{ color: var(--ink-2); font-size: 12.5px; }}
section {{ margin-top: 36px; }}
section > h2 {{ font-size: 19px; margin: 0 0 4px; }}
section > p.lead {{ margin: 0 0 14px; color: var(--ink-2); max-width: 780px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 16px; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; min-width: 0; overflow: hidden; }}
.card h3 {{ font-size: 15px; margin: 0 0 2px; }}
.card p {{ margin: 0 0 8px; color: var(--ink-2); font-size: 13.5px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--line); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--ink-2); font-weight: 600; font-size: 13px; }}
.note {{ font-size: 13px; color: var(--ink-2); margin-top: 8px; }}
footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--ink-2); font-size: 13px; }}
</style>
</head>
<body>
<main>
<header>
  <h1>E-commerce Product Analytics</h1>
  <p>An online clothing store (Google's thelook dataset, data to {as_of}) and a real
  14M-user ad experiment (Criteo). Built with BigQuery, dbt, DuckDB and Python.
  <a href="{repo}">Code and write-up on GitHub</a>.</p>
</header>

<div class="tiles">{tiles}</div>

<section>
  <h2>Store health</h2>
  <p class="lead">Growth comes from more buyers, not bigger orders. Returning buyers went from
  {ret_first:.0%} to {ret_last:.0%} of monthly buyers.</p>
  <div class="grid">
    <div class="card"><h3>Active buyers per month</h3><p>Customers with at least one order that wasn't cancelled</p>{buyers}</div>
    <div class="card"><h3>Weekly net revenue and next-quarter forecast</h3><p>{fc_note}</p>{forecast}</div>
  </div>
</section>

<section>
  <h2>Where customers drop off</h2>
  <p class="lead">The cart is the biggest leak: {cart:.0%} of carts become orders. Traffic channels
  don't differ (chi-square p = {chi_p:.2f}).</p>
  <div class="grid">
    <div class="card"><h3>Session funnel</h3><p>All web sessions</p>{funnel}</div>
    <div class="card"><h3>How long customers take to come back</h3>
      <p>Kaplan-Meier, which counts recent customers only for as long as they've been watched</p>
      <table><thead><tr><th>Ordered again within</th><th>Share of customers</th></tr></thead><tbody>{km}</tbody></table>
      <div class="note">The simple "ordered twice" rate says {naive:.0%}, but it mixes long-time customers with brand-new ones.</div>
    </div>
  </div>
</section>

<section>
  <h2>Who the customers are</h2>
  <p class="lead">Champions and Loyal customers are {top_cust:.0%} of buyers and {top_rev:.0%} of revenue.
  12-month value comes from a BG/NBD + Gamma-Gamma model that ran {clv_low:.0%} low on a holdout, so read it as a ranking.</p>
  <div class="grid">
    <div class="card"><h3>Share of customers vs share of revenue</h3><p>RFM segments</p>{segments}</div>
    <div class="card"><h3>Segments at a glance</h3><p>Lifetime net revenue share and predicted value per customer</p>{seg_table}</div>
  </div>
</section>

<section>
  <h2>Do ads work, and for whom?</h2>
  <p class="lead">Criteo's experiment: ads lifted visits about {lift:.0%} once treatment and control were made
  comparable, not the {plain:.0%} a plain comparison shows. The effect is concentrated: with budget for 10% of
  users, uplift targeting captures {up10:.0%} of the extra visits against {resp10:.0%} for targeting likely visitors.</p>
  <div class="grid">
    <div class="card"><h3>Lift from ads</h3><p>The adjustment uses only features measured before the test</p>{lift_chart}</div>
    <div class="card"><h3>Who to show ads to</h3><p>Share of all extra visits captured when showing ads to the top-scored users first</p>{qini}</div>
  </div>
</section>

<footer>
  Synthetic store data has generator quirks (every channel converts alike, customers stop at 4 orders),
  and they're listed in the <a href="{repo}#readme">README</a>. Every number on this page comes from
  <code>reports/stage*_numbers.json</code> in the repo. Built by Amal Pushp.
</footer>
</main>
<script>
// charts can measure their card before the grid settles; resize once it has
window.addEventListener("load", () => {{
  document.querySelectorAll(".js-plotly-plot").forEach((el) => Plotly.Plots.resize(el));
}});
</script>
</body>
</html>
"""


def main():
    con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
    s3, s4, s5, s6, s7 = (numbers(i) for i in (3, 4, 5, 6, 7))
    ns, fc, km = s3["north_star"], s7["forecast"], s3["time_to_second_order"]
    seg = s4["rfm"]["segments"]

    tiles = "".join([
        tile("Active buyers, " + pd.Timestamp(ns["latest_month"]).strftime("%b %Y"),
             f"{ns['active_buyers']:,}", f"{ns['yoy_growth']:+.0%} on a year earlier"),
        tile("Cart to purchase", f"{s3['funnel']['cart_to_purchase']:.0%}", "the biggest drop in the funnel"),
        tile("Ordered again within a year", f"{km['km_returned_by']['365_days']:.0%}", "Kaplan-Meier estimate"),
        tile("Next-quarter net revenue", f"${fc['quarter_median'] / 1e6:.2f}M",
             f"80% range ${fc['quarter_80'][0] / 1e6:.2f}M to ${fc['quarter_80'][1] / 1e6:.2f}M"),
        tile("Ad lift on visits", f"+{s5['cupac']['visit']['rel_lift']:.0%}",
             f"not +{s5['lift']['visit']['rel_lift']:.0%}: groups weren't comparable"),
    ])
    km_rows = "".join(
        f"<tr><td>{k.replace('_days', ' days')}</td><td>{v:.0%}</td></tr>"
        for k, v in km["km_returned_by"].items())
    page = PAGE.format(
        accent=ACCENT, ink=INK, ink2=INK_2, grey=GREY, grid=GRID, font=FONT, repo=REPO,
        as_of=s3["as_of_date"], tiles=tiles,
        ret_first=ns["returning_share_first"], ret_last=ns["returning_share_latest"],
        buyers=chart_html(buyers_chart(con), first=True),
        fc_note=(f"{fc['model']} model on log revenue, the most accurate of four in a rolling backtest; "
                 f"weeks ending {fc['first_week']} to {fc['last_week']}"),
        forecast=chart_html(forecast_chart(con, s7)),
        cart=s3["funnel"]["cart_to_purchase"], chi_p=s3["channel_test"]["p_value"],
        funnel=chart_html(funnel_chart(s3)), km=km_rows, naive=km["naive_repeat_rate"],
        top_cust=seg["Champions"]["customer_share"] + seg["Loyal"]["customer_share"],
        top_rev=seg["Champions"]["revenue_share"] + seg["Loyal"]["revenue_share"],
        clv_low=1 - 1 / s4["clv_validation"]["actual_over_predicted"],
        segments=chart_html(segment_chart(s4)), seg_table=segment_table(s4),
        lift=s5["cupac"]["visit"]["rel_lift"], plain=s5["lift"]["visit"]["rel_lift"],
        up10=s6["policy"][s6["best_model"]]["share_of_incremental_at_10"],
        resp10=s6["policy"]["Response model"]["share_of_incremental_at_10"],
        lift_chart=chart_html(lift_chart(s5)), qini=chart_html(qini_chart(s6)),
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
