"""
Stage 7: forecasting weekly net revenue for the next quarter, with
prediction intervals.

Four models compete: naive (next week = this week), drift (keep growing at
last year's average weekly rate), damped-trend exponential smoothing (ETS) and
ARIMA. All work on log revenue, because the store grows by a percentage each
week rather than a fixed amount. They're compared on a rolling-origin
backtest: pretend it's an earlier week, forecast 13 weeks ahead, score it
against what happened, and repeat from 8 different starting weeks. Besides
point accuracy, the backtest checks whether the 80% and 95% intervals really
contain the actual revenue 80% and 95% of the time.

Reads fct_orders from data/warehouse.duckdb (run `dbt build` first). Saves
charts to reports/figures/ and every quoted number to
reports/stage7_numbers.json.

usage: python scripts/06_forecast_revenue.py
"""
import json
import warnings
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

import chart_style
from chart_style import ACCENT, GREY, INK, INK_2

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
AS_OF = pd.Timestamp("2026-10-22")  # matches as_of_date in dbt/dbt_project.yml
HORIZON = 13
ORIGINS = 8
ORIGIN_STEP = 4
HISTORY_WEEKS = 104  # fit on the last two years: earlier weeks are tiny and behave differently
SIMS = 4000
LEVELS = (0.8, 0.95)


def weekly_revenue(con):
    daily = con.sql("""
        select date_trunc('day', ordered_at)::date as day, sum(net_revenue) as revenue
        from fct_orders group by 1
    """).df()
    s = daily.set_index(pd.to_datetime(daily.day)).revenue.asfreq("D", fill_value=0)
    weekly = s.resample("W-SUN").sum()
    # the week holding the cutoff is only partly observed, so it's dropped
    return weekly[weekly.index < AS_OF]


# Each model returns simulated future paths of log revenue (SIMS x HORIZON).
# Paths rather than formulas so every model gets intervals the same way, and
# so a quarter total can be read off by summing each path.

def sim_random_walk(log_y, rng, drift):
    steps = np.diff(log_y[-52:])
    mu = steps.mean() if drift else 0.0
    sigma = steps.std(ddof=1)
    # the drift itself is estimated from 51 steps, so its own uncertainty is
    # added to every path instead of pretending we know it exactly
    mu_draw = mu + (rng.normal(0, sigma / np.sqrt(len(steps)), (SIMS, 1)) if drift else 0)
    shocks = rng.normal(0, sigma, (SIMS, HORIZON))
    return log_y[-1] + np.cumsum(mu_draw + shocks, axis=1)


def sim_ets(log_y, rng):
    fit = ETSModel(pd.Series(log_y), error="add", trend="add", damped_trend=True).fit(disp=False)
    paths = fit.simulate(HORIZON, repetitions=SIMS, anchor="end", rng=rng)
    return np.asarray(paths).T


def sim_arima(log_y, rng):
    # (1,1,1) on log revenue with a drift term: weekly growth that wanders a
    # little around an average rate
    fit = ARIMA(log_y, order=(1, 1, 1), trend="t").fit()
    paths = fit.simulate(HORIZON, repetitions=SIMS, anchor="end", rng=rng)
    return np.asarray(paths).reshape(HORIZON, SIMS).T


MODELS = {
    "Naive": lambda y, rng: sim_random_walk(y, rng, drift=False),
    "Drift": lambda y, rng: sim_random_walk(y, rng, drift=True),
    "ETS (damped trend)": sim_ets,
    "ARIMA(1,1,1) + drift": sim_arima,
}


def summarize(paths):
    # median of the simulated revenue as the point forecast: exp() of a
    # log-scale mean would land below the average, the median doesn't care
    rev = np.exp(paths)
    out = {"median": np.median(rev, axis=0)}
    for level in LEVELS:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        out[level] = (np.quantile(rev, lo, axis=0), np.quantile(rev, hi, axis=0))
    out["quarter"] = rev.sum(axis=1)
    return out


def backtest(weekly, out):
    log_all = np.log(weekly.to_numpy())
    rows = []
    rng = np.random.default_rng(7)
    last_origin = len(weekly) - HORIZON
    origins = [last_origin - ORIGIN_STEP * i for i in range(ORIGINS)][::-1]
    for origin in origins:
        train = log_all[origin - HISTORY_WEEKS:origin]
        actual = weekly.to_numpy()[origin:origin + HORIZON]
        for name, sim in MODELS.items():
            f = summarize(sim(train, rng))
            ape = np.abs(f["median"] - actual) / actual
            rows.append({
                "origin": weekly.index[origin - 1], "model": name,
                "mape": ape.mean(),
                # positive = forecast too high
                "bias": ((f["median"] - actual) / actual).mean(),
                "cover_80": ((actual >= f[0.8][0]) & (actual <= f[0.8][1])).mean(),
                "cover_95": ((actual >= f[0.95][0]) & (actual <= f[0.95][1])).mean(),
            })
    bt = pd.DataFrame(rows)
    summary = bt.groupby("model")[["mape", "bias", "cover_80", "cover_95"]].mean().reindex(MODELS)
    out["backtest"] = {
        "origins": [str(d.date()) for d in bt.origin.unique()],
        "horizon_weeks": HORIZON,
        "models": {m: {k: round(float(v), 4) for k, v in r.items()} for m, r in summary.iterrows()},
    }
    return bt, summary


def forecast(weekly, best, out):
    rng = np.random.default_rng(13)
    log_y = np.log(weekly.to_numpy()[-HISTORY_WEEKS:])
    f = summarize(MODELS[best](log_y, rng))
    dates = pd.date_range(weekly.index[-1] + pd.Timedelta(weeks=1), periods=HORIZON, freq="W-SUN")
    last_13 = weekly.iloc[-HORIZON:].sum()
    q = f["quarter"]
    out["forecast"] = {
        "model": best,
        "first_week": str(dates[0].date()),
        "last_week": str(dates[-1].date()),
        "weekly_median": {str(d.date()): round(float(v), 0) for d, v in zip(dates, f["median"])},
        "quarter_median": round(float(np.median(q)), 0),
        "quarter_80": [round(float(v), 0) for v in np.quantile(q, [0.1, 0.9])],
        "quarter_95": [round(float(v), 0) for v in np.quantile(q, [0.025, 0.975])],
        "previous_13_weeks": round(float(last_13), 0),
        "growth_vs_previous_13_weeks": round(float(np.median(q) / last_13 - 1), 3),
    }
    return dates, f


def forecast_chart(weekly, dates, f, out):
    fc = out["forecast"]
    hist = weekly.iloc[-60:]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(hist.index, hist, color=INK_2, lw=1.6)
    # join the forecast to the last actual week so the fan starts from it
    x = [hist.index[-1], *dates]
    last = hist.iloc[-1]
    ax.fill_between(x, [last, *f[0.95][0]], [last, *f[0.95][1]], color=ACCENT, alpha=0.15, lw=0)
    ax.fill_between(x, [last, *f[0.8][0]], [last, *f[0.8][1]], color=ACCENT, alpha=0.3, lw=0)
    ax.plot(x, [last, *f["median"]], color=ACCENT, lw=2)
    ax.text(dates[-1], f[0.95][1][-1], " 95%", color=ACCENT, va="center", fontsize=9)
    ax.text(dates[-1], f[0.8][1][-1], " 80%", color=ACCENT, va="center", fontsize=9)
    ax.text(hist.index[5], hist.max() * 0.95, "Actual weekly net revenue", color=INK_2, fontsize=9.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"\\${v / 1000:,.0f}k"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(0, None)
    ax.set_xlim(hist.index[0], dates[-1] + pd.Timedelta(weeks=3))
    lo, hi = fc["quarter_80"]
    # dollar signs escaped: a pair of them would switch matplotlib to math mode
    chart_style.title(ax, f"Next quarter: about \\${fc['quarter_median'] / 1e6:.2f}M in net revenue, "
                          f"{fc['growth_vs_previous_13_weeks']:+.0%} on the last 13 weeks",
                      f"{fc['model']} on log revenue; 80% range \\${lo / 1e6:.2f}M to "
                      f"\\${hi / 1e6:.2f}M, weeks ending {fc['first_week']} to {fc['last_week']}")
    chart_style.save(fig, FIG_DIR / "17_revenue_forecast.png")


def backtest_chart(summary, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    names = list(summary.index)
    y = np.arange(len(names))
    best = out["forecast"]["model"]
    colors = [ACCENT if n == best else GREY for n in names]
    ax1.barh(y, summary.mape, color=colors, height=0.6)
    for i, v in enumerate(summary.mape):
        ax1.text(v + 0.003, i, f"{v:.1%}", va="center", color=INK, fontsize=9.5)
    ax1.set_yticks(y, names)
    ax1.invert_yaxis()
    ax1.tick_params(axis="y", length=0)
    ax1.set_xticks([0, 0.05, 0.1, 0.15])
    ax1.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax1.set_xlim(0, summary.mape.max() * 1.3)
    ax1.grid(True, axis="x")
    ax1.grid(False, axis="y")
    ax1.set_title("Average error, 13 weeks ahead", fontsize=10.5, color=INK_2,
                  fontweight="normal", loc="left")

    w = 0.36
    ax2.barh(y - w / 2, summary.cover_80, height=w, color=colors, alpha=0.55)
    ax2.barh(y + w / 2, summary.cover_95, height=w, color=colors)
    for level, offset in ((0.8, -w / 2), (0.95, w / 2)):
        for i, v in enumerate(summary[f"cover_{int(level * 100)}"]):
            ax2.text(v + 0.01, i + offset, f"{v:.0%} of {level:.0%}", va="center",
                     color=INK_2, fontsize=8.5)
    for level in LEVELS:
        ax2.axvline(level, color=INK_2, lw=0.8, ls="--")
    ax2.set_yticks(y, [""] * len(names))
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(0, 1.3)
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax2.grid(True, axis="x")
    ax2.grid(False, axis="y")
    ax2.set_title("How often the intervals held the actual", fontsize=10.5, color=INK_2,
                  fontweight="normal", loc="left")
    b = summary.loc[best]
    worst = summary.cover_80.idxmin()
    fig.suptitle(f"{best} was most accurate ({b.mape:.1%} error); {worst.split()[0]}'s 80% intervals "
                 f"held only {summary.loc[worst, 'cover_80']:.0%}",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.9, f"Rolling backtest from {ORIGINS} starting weeks, each forecasting "
             f"{HORIZON} weeks; dashed lines are the promised 80% and 95%", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "18_forecast_backtest.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
    weekly = weekly_revenue(con)
    out = {"as_of_date": str(AS_OF.date()), "last_full_week": str(weekly.index[-1].date()),
           "history_weeks_used": HISTORY_WEEKS}
    with warnings.catch_warnings():
        # statsmodels warns about convergence on a few backtest windows; the
        # fits still finish and get scored like the rest
        warnings.simplefilter("ignore")
        bt, summary = backtest(weekly, out)
        best = summary.mape.idxmin()
        dates, f = forecast(weekly, best, out)
    forecast_chart(weekly, dates, f, out)
    backtest_chart(summary, out)
    (ROOT / "reports" / "stage7_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
