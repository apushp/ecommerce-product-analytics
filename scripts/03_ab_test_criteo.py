"""
Stage 5: reading Criteo's ad experiment the way an experimentation team would.

Criteo randomly held back 15% of users from an ad campaign and recorded
whether each user visited the site and converted. Before trusting the lift,
the script checks the split (sample ratio mismatch) and the balance of the
pre-treatment features, then works out what the test could detect (power),
and estimates the lift with confidence intervals, both plain and with CUPAC
variance reduction. A/A simulations show what peeking does to false
positives, and the last part separates the effect of being assigned to see
ads from the effect of actually seeing one.

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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
        # past z of about 38 the p-value is smaller than a float can hold
        "p_value": float(f"{p:.3g}") if (p := 2 * stats.norm.sf(abs(diff / se))) > 0 else "<1e-300",
        "se": float(se),
    }


def lift(arrays, out):
    t = arrays["treatment"] == 1
    out["lift"] = {}
    for metric in ["visit", "conversion"]:
        y = arrays[metric]
        out["lift"][metric] = diff_in_means(y[t], y[~t], y[~t].mean())


def cupac(features, arrays, out):
    """
    CUPED needs a covariate that the treatment can't move. Criteo has no
    pre-period metric, so instead a model predicts each user's outcome from
    the 12 pre-treatment features (CUPAC), and that prediction is the
    covariate. Cross-fitting: the model that scores a user never trained on
    them, so the prediction can't leak the user's own outcome.
    """
    t = arrays["treatment"] == 1
    rng = np.random.default_rng(42)
    fold = rng.random(len(t)) < 0.5
    out["cupac"] = {}
    for metric in ["visit", "conversion"]:
        y = arrays[metric]
        pred = np.empty(len(y))
        for train_fold in (True, False):
            train = np.flatnonzero(fold == train_fold)
            # 1.5M rows is plenty for 12 features and keeps each fit under a minute
            train = rng.choice(train, size=min(1_500_000, len(train)), replace=False)
            model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=63, random_state=0)
            model.fit(features[train], y[train])
            score = fold != train_fold
            pred[score] = model.predict_proba(features[score])[:, 1]

        theta = np.cov(y, pred)[0, 1] / pred.var(ddof=1)
        y_adj = y - theta * (pred - pred.mean())
        plain = out["lift"][metric]
        adj = diff_in_means(y_adj[t], y_adj[~t], y[~t].mean())
        out["cupac"][metric] = {
            "corr_prediction_outcome": round(float(np.corrcoef(y, pred)[0, 1]), 4),
            "theta": round(float(theta), 4),
            "variance_reduction": round(1 - (adj["se"] / plain["se"]) ** 2, 4),
            # same precision without adjustment would need this many times the users
            "equivalent_sample_multiplier": round((plain["se"] / adj["se"]) ** 2, 3),
            # a big move here means the groups differed on the features, not just noise
            "estimate_shift_in_plain_ses": round((adj["abs_lift"] - plain["abs_lift"]) / plain["se"], 1),
            **adj,
        }


def propensity_check(features, arrays, out):
    """
    A second way to make the groups comparable, from the other side: instead
    of modelling the outcome, model who got assigned to treatment, then
    reweight each group to look like the whole population (inverse propensity
    weighting). If assignment were truly random, the model would score an
    AUC of 0.5 and the weights would do nothing.
    """
    t = arrays["treatment"] == 1
    rng = np.random.default_rng(1)
    fold = rng.random(len(t)) < 0.5
    ps = np.empty(len(t))
    for train_fold in (True, False):
        train = rng.choice(np.flatnonzero(fold == train_fold), size=1_500_000, replace=False)
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=63, random_state=0)
        model.fit(features[train], t[train])
        score = fold != train_fold
        ps[score] = model.predict_proba(features[score])[:, 1]

    # clipping stops a handful of extreme scores from dominating the weights
    ps = np.clip(ps, 0.01, 0.99)
    w_t, w_c = 1 / ps[t], 1 / (1 - ps[~t])
    out["propensity"] = {
        "auc": round(float(roc_auc_score(t, ps)), 4),
        "score_range_1_99": [round(float(q), 3) for q in np.quantile(ps, [0.01, 0.99])],
    }
    for metric in ["visit", "conversion"]:
        y = arrays[metric]
        m_t = (y[t] * w_t).sum() / w_t.sum()
        m_c = (y[~t] * w_c).sum() / w_c.sum()
        out["propensity"][metric] = {
            "treatment_mean": round(float(m_t), 6),
            "control_mean": round(float(m_c), 6),
            "rel_lift": round(float(m_t / m_c - 1), 4),
        }


def lift_chart(out):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    rows = [("Visits", "visit"), ("Conversions", "conversion")]
    for i, (label, metric) in enumerate(rows):
        base = i * 3
        for offset, src, color, name in [(0, out["cupac"], ACCENT, "CUPAC"),
                                         (0.8, out["propensity"], GREY, "propensity weights"),
                                         (1.6, out["lift"], GREY, "plain difference")]:
            r = src[metric]
            y_pos = base + offset
            if "rel_ci" in r:
                ax.plot(r["rel_ci"], [y_pos] * 2, color=color, lw=2.5, solid_capstyle="round")
                right = r["rel_ci"][1]
                note = f"  ({r['rel_ci'][0]:.1%} to {r['rel_ci'][1]:.1%})"
            else:
                right, note = r["rel_lift"], ""
            ax.scatter([r["rel_lift"]], [y_pos], color=color, s=40, zorder=3,
                       marker="o" if note else "D")
            ax.text(right + 0.012, y_pos, f"{name}  {r['rel_lift']:+.1%}{note}", va="center",
                    color=INK if offset == 0 else INK_2, fontsize=9.5,
                    fontweight="bold" if offset == 0 else "normal")
    ax.set_yticks([0.8, 3.8], [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 5.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Relative lift, treatment vs control (lines are 95% CIs)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    v, plain = out["cupac"]["visit"], out["lift"]["visit"]
    title(ax, f"Ads lifted visits about {v['rel_lift']:.0%}, not the {plain['rel_lift']:.0%} "
              f"a plain comparison shows",
          "Both adjustments use only pre-treatment features; CUPAC also cut visit variance by "
          f"{v['variance_reduction']:.0%}")
    save(fig, "09_lift.png")


def aa_and_peeking(arrays, out, sims=1000, looks=20):
    """
    A/A tests: split the control group in half at random, so there is no real
    effect, and test it. About 5% should come out "significant". Then the same
    A/A tests are checked 20 times as data comes in, stopping at the first
    p < 0.05, which is what happens when someone watches a dashboard.
    """
    y = arrays["visit"][arrays["treatment"] == 0]
    n = len(y)
    rng = np.random.default_rng(7)
    starts = np.linspace(0, n, looks + 1).astype(int)[:-1]
    n_seen = np.diff(np.append(starts, n)).cumsum()
    visits_seen = np.add.reduceat(y, starts).cumsum()
    p_values = np.empty((sims, looks))
    for s in range(sims):
        a = rng.random(n) < 0.5
        n_a = np.add.reduceat(a.astype(np.int64), starts).cumsum()
        v_a = np.add.reduceat(np.where(a, y, 0.0), starts).cumsum()
        n_b, v_b = n_seen - n_a, visits_seen - v_a
        pooled = visits_seen / n_seen
        se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
        p_values[s] = 2 * stats.norm.sf(np.abs((v_a / n_a - v_b / n_b) / se))

    final = p_values[:, -1]
    by_looks = {}
    for k in [1, 2, 4, 5, 10, 20]:
        idx = np.arange(1, k + 1) * (looks // k) - 1
        by_looks[k] = float((p_values[:, idx] < ALPHA).any(axis=1).mean())
    out["aa"] = {
        "sims": sims,
        "users_per_sim": int(n),
        "false_positive_rate": round(float((final < ALPHA).mean()), 3),
        # p-values from a working test with no effect should be uniform
        "ks_uniform_p": round(float(stats.kstest(final, "uniform").pvalue), 3),
        "false_positive_rate_by_looks": {str(k): round(v, 3) for k, v in by_looks.items()},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.hist(final, bins=20, range=(0, 1), color=GREY, edgecolor="white", linewidth=1.5)
    ax1.axhline(sims / 20, color=INK_2, lw=1, ls="--")
    ax1.set_ylim(0, sims / 20 * 1.6)
    ax1.text(0, sims / 20 * 1.5, "dashed line: what a flat (uniform) spread looks like",
             color=INK_2, fontsize=9)
    ax1.set_xlabel("p-value of A/A test, checked once")
    ax1.set_ylabel("Simulations")
    ax1.set_title(f"Checked once: {out['aa']['false_positive_rate']:.1%} false wins",
                  fontsize=10.5, color=INK_2, fontweight="normal", loc="left")
    ks = list(by_looks)
    ax2.plot(ks, [by_looks[k] for k in ks], color=ACCENT, lw=2, marker="o", ms=6)
    for k in ks:
        ax2.text(k, by_looks[k] + 0.012, f"{by_looks[k]:.0%}", ha="center", color=INK, fontsize=9)
    ax2.axhline(ALPHA, color=INK_2, lw=1, ls="--")
    ax2.text(20, ALPHA - 0.012, "5% promised", ha="right", va="top", color=INK_2, fontsize=9)
    ax2.set_xticks(ks)
    ax2.set_ylim(0, max(by_looks.values()) + 0.06)
    ax2.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax2.set_xlabel("Times the test is checked before stopping")
    ax2.set_ylabel("Share with a false win")
    ax2.set_title("Peeking and stopping at the first p < 0.05", fontsize=10.5, color=INK_2,
                  fontweight="normal", loc="left")
    fig.suptitle(f"Peeking 20 times turns a 5% false-win rate into {by_looks[20]:.0%}",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.9, f"{sims:,} A/A tests on the control group's {n / 1e6:.1f}M users, "
             "split at random so there is no real effect", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "10_aa_peeking.png", dpi=200)
    plt.close(fig)


def exposure_effect(arrays, out):
    """
    Only 3.6% of the treatment group actually saw an ad. The A/B difference
    (intent to treat) averages over everyone assigned, including the 96% who
    never saw one. Dividing by the exposure rate gives the effect on the users
    who were exposed (complier average causal effect), using random
    assignment as an instrument. Comparing exposed to unexposed users directly
    is the tempting shortcut, and it's biased because the people who get
    shown ads were already more likely to visit.
    """
    t = arrays["treatment"] == 1
    e = arrays["exposure"] == 1
    exposed_share = float(e[t].mean() - e[~t].mean())
    z = stats.norm.ppf(1 - ALPHA / 2)
    out["exposure"] = {"exposed_share_treatment": round(float(e[t].mean()), 4),
                       "exposed_share_control": round(float(e[~t].mean()), 4)}
    for metric in ["visit", "conversion"]:
        y = arrays[metric]
        # the CUPAC estimate, since the plain difference is inflated by the
        # groups not being comparable (see the balance check)
        itt = out["cupac"][metric]
        # the exposure rate is measured on 11.9M users, so its own noise is
        # tiny next to the ITT's and the SE just scales by it
        cace = itt["abs_lift"] / exposed_share
        se = itt["se"] / exposed_share
        exposed_rate = float(y[t & e].mean())
        out["exposure"][metric] = {
            "itt_abs": itt["abs_lift"],
            "cace_abs": round(cace, 5),
            "cace_ci": [round(cace - z * se, 5), round(cace + z * se, 5)],
            "naive_exposed_vs_unexposed": round(exposed_rate - float(y[t & ~e].mean()), 5),
            "naive_exposed_vs_control": round(exposed_rate - float(y[~t].mean()), 5),
            "exposed_rate": round(exposed_rate, 5),
            # what exposed users would have done without the ad
            "complier_baseline": round(exposed_rate - cace, 5),
        }

    r = out["exposure"]["visit"]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    bars = [("Exposed vs unexposed\n(treatment group only)", r["naive_exposed_vs_unexposed"], GREY),
            ("Exposed vs control group", r["naive_exposed_vs_control"], GREY),
            ("Causal effect of seeing the ad\n(assignment as instrument)", r["cace_abs"], ACCENT)]
    for i, (label, v, color) in enumerate(bars):
        ax.barh(i, v, color=color, height=0.6)
        # the causal bar has an error bar, so its label starts past the CI
        x = r["cace_ci"][1] if color == ACCENT else v
        ax.text(x + 0.006, i, f"+{v * 100:.1f} pts", va="center", color=INK,
                fontweight="bold" if color == ACCENT else "normal")
    ax.errorbar(r["cace_abs"], 2, xerr=[[r["cace_abs"] - r["cace_ci"][0]], [r["cace_ci"][1] - r["cace_abs"]]],
                color=INK, capsize=4, lw=1)
    ax.set_yticks(range(3), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(b[1] for b in bars) * 1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f} pts"))
    ax.set_xlabel("Gain in visit rate, percentage points")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ratio = r["naive_exposed_vs_unexposed"] / r["cace_abs"]
    fig.suptitle(f"Comparing exposed to unexposed users shows {ratio:.1f}x the ad's real effect",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.86, f"Exposed users would have visited {r['complier_baseline']:.1%} of the "
             f"time without the ad; the average control user visits "
             f"{out['lift']['visit']['control_mean']:.1%}", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIG_DIR / "11_exposure_effect.png", dpi=200)
    plt.close(fig)


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
    features = np.column_stack([
        np.asarray(v, dtype=np.float32)
        for v in duckdb.sql(f"select {', '.join(FEATURES)} from data").fetchnumpy().values()
    ])
    cupac(features, arrays, out)
    propensity_check(features, arrays, out)
    lift_chart(out)
    aa_and_peeking(arrays, out)
    exposure_effect(arrays, out)

    for section in ["lift", "cupac"]:
        for metric in out[section].values():
            metric.pop("se")
    (ROOT / "reports" / "stage5_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
