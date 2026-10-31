"""
Stage 6: who should see an ad? Uplift modeling on Criteo's experiment.

Stage 5 showed ads lift visits on average. Ad budgets get spent one user at a
time though, and users differ: some only visit because of the ad, some would
visit anyway, some ignore ads entirely. An uplift model predicts, for each
user, how much the ad changes their chance of visiting, so the budget can go
where it actually makes a difference.

The script trains on half the users and scores the other half, compares two
uplift models (S-learner and T-learner) with the usual shortcut of targeting
likely visitors, and evaluates them with Qini curves. Evaluation is weighted
by inverse propensity, because Stage 5 found treatment and control aren't
perfectly comparable in this file.

Needs data/raw/criteo.parquet (run scripts/03_ab_test_criteo.py first).
Saves charts to reports/figures/ and every quoted number to
reports/stage6_numbers.json.

usage: python scripts/04_uplift_criteo.py
"""
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklift.metrics import qini_auc_score

import chart_style
from chart_style import ACCENT, GREY, INK, INK_2

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
PARQUET = ROOT / "data" / "raw" / "criteo.parquet"
FEATURES = [f"f{i}" for i in range(12)]
TRAIN_ROWS = 4_000_000
GRID = np.linspace(0, 1, 101)


def model():
    # uplift is a small difference between two noisy predictions, so leaves
    # need plenty of users each or the model chases noise
    return HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=63,
                                          min_samples_leaf=2000, random_state=0)


def load():
    cols = duckdb.sql(
        f"select {', '.join(FEATURES)}, treatment, visit from '{PARQUET}'"
    ).fetchnumpy()
    x = np.column_stack([np.asarray(cols[f], dtype=np.float32) for f in FEATURES])
    t = np.asarray(cols["treatment"]) == 1
    y = np.asarray(cols["visit"], dtype=np.float64)
    return x, t, y


def propensity(x, t, rng):
    # same cross-fitted assignment model as stage 5. the weights make the
    # treated and control users in any slice look like the same population
    fold = rng.random(len(t)) < 0.5
    ps = np.empty(len(t))
    for train_fold in (True, False):
        train = rng.choice(np.flatnonzero(fold == train_fold), size=1_500_000, replace=False)
        m = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=63, random_state=0)
        m.fit(x[train], t[train])
        ps[fold != train_fold] = m.predict_proba(x[fold != train_fold])[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    return np.where(t, 1 / ps, 1 / (1 - ps))


def fit_models(x, t, y, train):
    """
    S-learner: one model, with treatment as an extra input. Uplift = its
    prediction with treatment switched on minus switched off.
    T-learner: two models, one trained only on treated users and one only on
    control. Uplift = the difference between their predictions.
    The treated-only model on its own is also the "response model": it
    predicts who is likely to visit after seeing ads, which is how campaigns
    often get targeted when nobody builds an uplift model.
    """
    xs = np.column_stack([x[train], t[train].astype(np.float32)])
    s_model = model().fit(xs, y[train])
    tr_t, tr_c = train[t[train]], train[~t[train]]
    m_t = model().fit(x[tr_t], y[tr_t])
    m_c = model().fit(x[tr_c], y[tr_c])
    return s_model, m_t, m_c


def score(x, s_model, m_t, m_c):
    on = np.column_stack([x, np.ones(len(x), dtype=np.float32)])
    off = np.column_stack([x, np.zeros(len(x), dtype=np.float32)])
    p_t, p_c = m_t.predict_proba(x)[:, 1], m_c.predict_proba(x)[:, 1]
    return {
        "S-learner": s_model.predict_proba(on)[:, 1] - s_model.predict_proba(off)[:, 1],
        "T-learner": p_t - p_c,
        "Response model": p_t,
    }


def qini_curve(order, t, y, w):
    """
    Sort users from highest score to lowest. At each cut-off, incremental
    visits = visits among treated users so far, minus what the control users
    so far say those treated users would have done without ads (control
    visit rate x number of treated). Weighted sums stand in for counts.
    Returns incremental visits at each 1% of users targeted. `order` is the
    users sorted by score, passed in so the bootstrap only sorts once.
    """
    t, y, w = t[order], y[order], w[order]
    n_t = np.cumsum(w * t)
    v_t = np.cumsum(w * t * y)
    n_c = np.cumsum(w * ~t)
    v_c = np.cumsum(w * ~t * y)
    idx = np.clip((GRID * len(y)).astype(int) - 1, 0, len(y) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = v_t[idx] - np.where(n_c[idx] > 0, v_c[idx] / n_c[idx], 0) * n_t[idx]
    inc[0] = 0
    return inc


def bootstrap(orders, t, y, w, out, reps=100):
    """
    How sure are we that one targeting rule beats another? Each round gives
    every user a random Poisson(1) count, which is the same as resampling
    users with replacement, and recomputes every curve. Using the same draw
    for all models keeps the comparison paired, so their shared noise cancels.
    """
    rng = np.random.default_rng(11)
    at_10 = {name: [] for name in orders}
    for _ in range(reps):
        wb = w * rng.poisson(1, len(w))
        for name, order in orders.items():
            inc = qini_curve(order, t, y, wb)
            at_10[name].append(inc[10] / inc[-1])
    at_10 = {k: np.array(v) for k, v in at_10.items()}
    diff = at_10[out["best_model"]] - at_10["Response model"]
    out["bootstrap"] = {
        "reps": reps,
        "share_at_10_ci": {k: [round(float(q), 3) for q in np.quantile(v, [0.025, 0.975])]
                           for k, v in at_10.items()},
        "best_minus_response_at_10": round(float(diff.mean()), 3),
        "best_minus_response_at_10_ci": [round(float(q), 3) for q in np.quantile(diff, [0.025, 0.975])],
    }


def decile_uplift(score, t, y, w):
    # observed uplift within each tenth of users, ranked by predicted uplift.
    # if the model works, the top tenth shows the biggest real gap
    cuts = np.quantile(score, np.linspace(0, 1, 11))
    bucket = np.clip(np.searchsorted(cuts, score, side="right") - 1, 0, 9)
    rows = []
    for d in range(9, -1, -1):
        m = bucket == d
        mt, mc = m & t, m & ~t
        r_t = np.average(y[mt], weights=w[mt])
        r_c = np.average(y[mc], weights=w[mc])
        # unweighted SE is close enough here: the weights only run from about
        # 1.1 to 1.2 for treated users and 6 to 10 for control
        se = np.sqrt(r_t * (1 - r_t) / mt.sum() + r_c * (1 - r_c) / mc.sum())
        rows.append({"decile": 10 - d, "users": int(m.sum()), "treated_rate": round(r_t, 5),
                     "control_rate": round(r_c, 5), "uplift": round(r_t - r_c, 5),
                     "ci": [round(r_t - r_c - 1.96 * se, 5), round(r_t - r_c + 1.96 * se, 5)]})
    return rows


def qini_chart(curves, total, out):
    best = out["best_model"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(GRID, GRID, color=INK_2, lw=1, ls="--", label="Random targeting")
    for name in ["S-learner", "T-learner", "Response model"]:
        if name == best:
            color, lw = ACCENT, 2.4
        elif name == "Response model":
            color, lw = GREY, 2
        else:
            color, lw = "#7aa9e3", 1.6
        label = "Response model (likely visitors)" if name == "Response model" else name
        ax.plot(GRID, curves[name] / total, color=color, lw=lw, label=label)
    # curves can pass 100%: the last users added had a slightly negative
    # measured effect, so including them takes some extra visits back off
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of users shown ads, best-scored first")
    ax.set_ylabel("Share of all extra visits the ads caused")
    ax.axvline(0.1, color=GREY, lw=0.8, ls=":")
    k = out["policy"]
    chart_style.title(ax, f"With budget for 10% of users, uplift targeting captures "
                          f"{k[best]['share_of_incremental_at_10']:.0%} of the extra visits",
                      f"Targeting likely visitors captures {k['Response model']['share_of_incremental_at_10']:.0%}; "
                      f"by 30% both are near {k[best]['share_of_incremental_at_30']:.0%}. "
                      f"{out['test_users'] / 1e6:.1f}M held-out users")
    chart_style.save(fig, FIG_DIR / "12_qini.png")


def decile_chart(rows, name):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = [r["decile"] for r in rows]
    up = [r["uplift"] for r in rows]
    colors = [ACCENT if r["decile"] == 1 else GREY for r in rows]
    ax.bar(x, up, color=colors, width=0.7)
    ax.errorbar(x, up, yerr=[[u - r["ci"][0] for u, r in zip(up, rows)],
                             [r["ci"][1] - u for u, r in zip(up, rows)]],
                fmt="none", ecolor=INK_2, capsize=3, lw=1)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, [f"{d}" for d in x])
    ax.set_xlabel(f"Tenth of users, ranked by {name} predicted uplift (1 = highest)")
    ax.set_ylabel("Observed gain in visit rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    top, rest = rows[0]["uplift"], np.mean([r["uplift"] for r in rows[5:]])
    chart_style.title(ax, f"The top tenth gains {top * 100:.1f} points from ads; "
                          f"the bottom half gains {rest * 100:.2f}",
                      "Treated minus control visit rate in each tenth of held-out users, "
                      "propensity-weighted, with 95% CIs")
    chart_style.save(fig, FIG_DIR / "13_uplift_deciles.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6)
    x, t, y = load()
    w = propensity(x, t, rng)

    is_train = rng.random(len(y)) < 0.5
    train = rng.choice(np.flatnonzero(is_train), size=TRAIN_ROWS, replace=False)
    test = np.flatnonzero(~is_train)
    models = fit_models(x, t, y, train)
    scores = score(x[test], *models)
    t_te, y_te, w_te = t[test], y[test], w[test]

    orders = {name: np.argsort(-s, kind="stable") for name, s in scores.items()}
    curves = {name: qini_curve(o, t_te, y_te, w_te) for name, o in orders.items()}
    total = curves["T-learner"][-1]
    out = {"train_users": TRAIN_ROWS, "test_users": int(len(test)),
           "total_incremental_visits_test": round(float(total), 1), "policy": {}}
    for name, inc in curves.items():
        # area between the curve and random targeting, as a share of all the
        # extra visits: 0 = no better than random
        out["policy"][name] = {
            "qini_coefficient": round(float(np.mean(inc - GRID * total) / total), 4),
            **{f"share_of_incremental_at_{k}": round(float(inc[k] / total), 3)
               for k in (10, 20, 30, 50)},
            # cross-check against scikit-uplift, which can't take weights
            "sklift_qini_auc_unweighted": round(float(qini_auc_score(y_te, scores[name], t_te)), 4),
        }
    uplift_models = ["S-learner", "T-learner"]
    best = max(uplift_models, key=lambda m: out["policy"][m]["qini_coefficient"])
    out["best_model"] = best
    out["deciles"] = decile_uplift(scores[best], t_te, y_te, w_te)
    out["share_negative_predicted_uplift"] = round(float((scores[best] < 0).mean()), 4)
    bootstrap(orders, t_te, y_te, w_te, out)

    qini_chart(curves, total, out)
    decile_chart(out["deciles"], best)
    (ROOT / "reports" / "stage6_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
