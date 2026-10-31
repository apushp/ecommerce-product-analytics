"""
Stage 5: reading Criteo's ad experiment the way an experimentation team would.

Criteo randomly held back 15% of users from an ad campaign and recorded
whether each user visited the site and converted. Before trusting the lift,
the script checks the split (sample ratio mismatch) and the balance of the
pre-treatment features, then works out what the test could detect (power),
and finally estimates the lift with confidence intervals.

Reads data/raw/criteo-uplift-v2.1.csv.gz (converted to parquet on first run),
saves charts to reports/figures/ and every quoted number to
reports/stage5_numbers.json.

usage: python scripts/03_ab_test_criteo.py
"""
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats

import chart_style
from chart_style import ACCENT, GREY, INK, INK_2, title

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
RAW = ROOT / "data" / "raw"
PARQUET = RAW / "criteo.parquet"
FEATURES = [f"f{i}" for i in range(12)]

# Criteo's paper says 85% of users were assigned to see ads. The SRM check
# compares against this designed split, not against whatever we observed
DESIGNED_TREATMENT_SHARE = 0.85
ALPHA = 0.05
POWER = 0.80


def save(fig, name):
    chart_style.save(fig, FIG_DIR / name)


def load():
    if not PARQUET.exists():
        # the csv takes ~30s to parse every time, parquet loads in a couple
        duckdb.sql(f"""
            copy (select * from read_csv_auto('{RAW / "criteo-uplift-v2.1.csv.gz"}'))
            to '{PARQUET}' (format parquet)
        """)
    return duckdb.sql(f"select * from '{PARQUET}'")


def srm_check(data, out):
    n_t, n_c = duckdb.sql(
        "select sum(treatment), count(*) - sum(treatment) from data"
    ).fetchall()[0]
    n = n_t + n_c
    expected = [n * DESIGNED_TREATMENT_SHARE, n * (1 - DESIGNED_TREATMENT_SHARE)]
    chi2, p = stats.chisquare([n_t, n_c], f_exp=expected)
    out["srm"] = {
        "users": int(n),
        "treatment": int(n_t),
        "control": int(n_c),
        "observed_treatment_share": round(n_t / n, 5),
        "designed_treatment_share": DESIGNED_TREATMENT_SHARE,
        "chi2": round(chi2, 3),
        "p_value": round(p, 4),
        # a common rule is to stop and investigate below p = 0.001, since a
        # broken split means the groups may differ in ways we can't see
        "passed": bool(p > 0.001),
    }


def balance_check(data, out):
    # standardized mean difference: gap between group means in units of
    # pooled standard deviation. with 14M users every gap would be
    # "significant", so a size threshold (0.1) is more useful than a p-value
    cols = ", ".join(
        f"(avg({f}) filter (where treatment = 1) - avg({f}) filter (where treatment = 0))"
        f" / sqrt((var_samp({f}) filter (where treatment = 1)"
        f" + var_samp({f}) filter (where treatment = 0)) / 2) as {f}"
        for f in FEATURES
    )
    smd = duckdb.sql(f"select {cols} from data").df().iloc[0]
    # under pure random assignment an SMD is roughly normal with this spread,
    # so it tells us how big a gap chance alone would produce
    noise = np.sqrt(1 / out["srm"]["treatment"] + 1 / out["srm"]["control"])

    # the file isn't shuffled: splitting it into 14 equal blocks by row order
    # shows treatment share and visit rate jumping between blocks, which looks
    # like several tests with different splits stacked together
    blocks = duckdb.sql(f"""
        with numbered as (
            select treatment, visit, ntile(14) over (order by file_row_number) as block
            from read_parquet('{PARQUET}', file_row_number = true)
        )
        select block, avg(treatment) as share_t, avg(visit) as visit_rate
        from numbered group by 1 order by 1
    """).df()
    out["balance"] = {
        "smd": {f: round(float(v), 4) for f, v in smd.items()},
        "max_abs_smd": round(float(smd.abs().max()), 4),
        "all_below_0.1": bool((smd.abs() < 0.1).all()),
        "smd_chance_sd": round(float(noise), 5),
        "max_smd_in_chance_sds": round(float(smd.abs().max() / noise), 1),
        "row_block_treatment_share": [round(float(s), 3) for s in blocks.share_t],
        "row_block_visit_rate": [round(float(v), 4) for v in blocks.visit_rate],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    order = smd.abs().sort_values().index
    colors = [ACCENT if abs(smd[f]) / noise > 3 else GREY for f in order]
    ax1.scatter(smd[order], range(len(order)), color=colors, s=36, zorder=3)
    ax1.axvspan(-1.96 * noise, 1.96 * noise, color=GREY, alpha=0.35, lw=0)
    for x in (-0.1, 0.1):
        ax1.axvline(x, color=INK_2, lw=0.8, ls="--")
    ax1.text(0.1, len(order) - 0.4, " 0.1 rule", color=INK_2, fontsize=9, va="top")
    ax1.set_yticks(range(len(order)), order)
    ax1.set_xlim(-0.12, 0.12)
    ax1.set_xlabel("Standardized mean difference")
    ax1.grid(True, axis="x")
    ax1.grid(False, axis="y")
    ax1.set_title("Feature gaps, treatment vs control", fontsize=10.5, color=INK_2,
                  fontweight="normal", loc="left")

    ax2.bar(blocks.block, blocks.share_t, color=GREY, width=0.7)
    ax2.axhline(DESIGNED_TREATMENT_SHARE, color=ACCENT, lw=2)
    ax2.text(14.6, DESIGNED_TREATMENT_SHARE, "85%\noverall", color=ACCENT, va="center",
             fontweight="bold", fontsize=9)
    ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax2.set_xticks(blocks.block)
    ax2.set_xlabel("Block of ~1M rows, in file order")
    ax2.set_title("Treatment share by block of the file", fontsize=10.5, color=INK_2,
                  fontweight="normal", loc="left")
    fig.suptitle(f"Features pass the 0.1 rule, but the biggest gap is "
                 f"{smd.abs().max() / noise:.0f}x what chance would give",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.9, "Grey band: 95% range under random assignment. The file stacks blocks "
             "with very different splits, which would explain it", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "08_balance.png", dpi=200)
    plt.close(fig)


def mde(p, n_total, share_t):
    # smallest absolute lift a two-sided z-test would catch with 80% power
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    return z * np.sqrt(p * (1 - p) * (1 / (n_total * share_t) + 1 / (n_total * (1 - share_t))))


def power_analysis(data, out):
    base = duckdb.sql(
        "select avg(visit), avg(conversion), count(*) from data where treatment = 0"
    ).fetchall()[0]
    visit_c, conv_c = base[0], base[1]
    n = out["srm"]["users"]
    share = out["srm"]["observed_treatment_share"]

    # an 85/15 split needs more users than 50/50 for the same precision,
    # because the small control group sets the noise. the variance factor
    # is 1/0.85 + 1/0.15 against 1/0.5 + 1/0.5
    split_cost = (1 / share + 1 / (1 - share)) / 4
    out["power"] = {
        "alpha": ALPHA,
        "power": POWER,
        "visit_rate_control": round(visit_c, 5),
        "conversion_rate_control": round(conv_c, 5),
        "visit_mde_relative": round(mde(visit_c, n, share) / visit_c, 4),
        "conversion_mde_relative": round(mde(conv_c, n, share) / conv_c, 4),
        "visit_mde_relative_if_50_50": round(mde(visit_c, n, 0.5) / visit_c, 4),
        "users_needed_vs_50_50": round(split_cost, 2),
    }

    sizes = np.logspace(4.5, np.log10(n * 1.6), 200)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for s, color, label in [(share, ACCENT, "85/15 split (as run)"), (0.5, GREY, "50/50 split")]:
        rel = mde(visit_c, sizes, s) / visit_c
        ax.plot(sizes, rel, color=color, linewidth=2)
        ax.text(sizes[-1] * 1.03, rel[-1], label, color=color if s != 0.5 else INK_2,
                va="center", fontweight="bold" if s != 0.5 else "normal")
    actual = mde(visit_c, n, share) / visit_c
    ax.scatter([n], [actual], color=ACCENT, s=40, zorder=3)
    ax.annotate(f"{n / 1e6:.1f}M users: can detect a {actual:.1%} lift",
                (n, actual), xytext=(-12, 28), textcoords="offset points", ha="right",
                color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"))
    ax.set_xlabel("Users in the test (log scale)")
    ax.set_ylabel("Smallest detectable lift in visits")
    ax.grid(True, which="major", axis="both")
    title(ax, f"The 85/15 split needs {split_cost:.1f}x the users of a 50/50 test for the same precision",
          f"Minimum detectable relative lift in visit rate ({visit_c:.2%} baseline), "
          f"80% power, 5% two-sided")
    fig.subplots_adjust(right=0.8)
    save(fig, "07_power_mde.png")


def diff_in_means(y_t, y_c, base):
    # difference in means with an unpooled standard error. relative lift uses
    # the delta method: lift = mean_t / mean_c - 1, and its variance comes
    # from both means being noisy
    m_t, m_c = y_t.mean(), y_c.mean()
    v_t, v_c = y_t.var(ddof=1) / len(y_t), y_c.var(ddof=1) / len(y_c)
    diff = m_t - m_c
    se = np.sqrt(v_t + v_c)
    z = stats.norm.ppf(1 - ALPHA / 2)
    rel_se = np.sqrt(v_t / base ** 2 + (m_t ** 2) * v_c / base ** 4)
    return {
        "treatment_mean": round(float(m_t), 6),
        "control_mean": round(float(m_c), 6),
        "abs_lift": round(float(diff), 6),
        "abs_ci": [round(float(diff - z * se), 6), round(float(diff + z * se), 6)],
        "rel_lift": round(float(diff / base), 4),
        "rel_ci": [round(float(diff / base - z * rel_se), 4), round(float(diff / base + z * rel_se), 4)],
        "z": round(float(diff / se), 2),
        "p_value": float(f"{2 * stats.norm.sf(abs(diff / se)):.3g}"),
        "se": float(se),
    }


def lift(arrays, out):
    t = arrays["treatment"] == 1
    out["lift"] = {}
    for metric in ["visit", "conversion"]:
        y = arrays[metric]
        out["lift"][metric] = diff_in_means(y[t], y[~t], y[~t].mean())


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    out = {}
    srm_check(data, out)
    balance_check(data, out)
    power_analysis(data, out)

    cols = duckdb.sql("select treatment, visit, conversion, exposure from data").fetchnumpy()
    arrays = {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}
    lift(arrays, out)

    for metric in out["lift"].values():
        metric.pop("se")
    (ROOT / "reports" / "stage5_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
