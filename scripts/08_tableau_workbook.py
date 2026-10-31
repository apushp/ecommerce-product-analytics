"""
Builds a Tableau version of the dashboard: tableau/ecommerce_analytics.twb plus
one .hyper extract per chart.

Exports the data behind each chart to tableau/*.csv (from the warehouse and the
reports/stage*_numbers.json files), turns each CSV into a Tableau extract, and
writes the workbook XML. Open the .twb in Tableau Public, then
File -> Save to Tableau Public.

usage: python scripts/08_tableau_workbook.py
"""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import duckdb
import pandas as pd
from tableauhyperapi import (Connection, CreateMode, HyperProcess, SqlType, TableDefinition,
                             TableName, Telemetry, escape_string_literal)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tableau"
OUT = OUT_DIR / "ecommerce_analytics.twb"

ACCENT, ACCENT_LIGHT, GREY, INK_2 = "#2a78d6", "#9cc0eb", "#b4b2ab", "#52514e"

HYPER_TYPES = {"string": SqlType.text(), "real": SqlType.double(),
               "integer": SqlType.big_int(), "date": SqlType.date()}
EXTRACT_TABLE = TableName("Extract", "Extract")


def numbers(stage):
    return json.loads((ROOT / "reports" / f"stage{stage}_numbers.json").read_text())


# ---- data: one small table per chart ---------------------------------------------

def export_data():
    con = duckdb.connect(str(ROOT / "data" / "warehouse.duckdb"), read_only=True)
    s3, s4, s5, s6, s7 = (numbers(i) for i in (3, 4, 5, 6, 7))
    tables = {}

    kpis = con.sql("""
        select order_month, new_buyers, returning_buyers from mart_monthly_kpis
        where is_complete_month and order_month >= '2023-01-01' order by order_month
    """).df()
    tables["buyers"] = [
        {"order_month": r.order_month.date(), "buyer_type": kind, "buyers": int(v)}
        for r in kpis.itertuples() for kind, v in (("New", r.new_buyers), ("Returning", r.returning_buyers))]

    steps = ["Sessions", "Viewed a product", "Added to cart", "Purchased"]
    f = s3["funnel"]
    tables["funnel"] = [
        {"step": s, "step_order": i, "sessions": f[s],
         "carry_on": "" if i == 0 else f"{f[s] / f[steps[i - 1]]:.0%} carry on",
         "highlight": "Biggest drop" if s == "Purchased" else "Other"}
        for i, s in enumerate(steps)]

    seg, clv = s4["rfm"]["segments"], s4["clv"]["by_segment"]
    tables["segments"] = [
        {"segment": name, "segment_order": i, "measure": label, "share": seg[name][key],
         "customers": seg[name]["customers"], "value_12m": clv[name]["avg_clv"]}
        for i, name in enumerate(seg)
        for label, key in (("Share of customers", "customer_share"), ("Share of revenue", "revenue_share"))]

    daily = con.sql("""
        select date_trunc('day', ordered_at)::date as day, sum(net_revenue) as revenue
        from fct_orders group by 1
    """).df()
    s = daily.set_index(pd.to_datetime(daily.day)).revenue.asfreq("D", fill_value=0)
    weekly = s.resample("W-SUN").sum()
    weekly = weekly[weekly.index <= pd.Timestamp(s7["last_full_week"])].iloc[-60:]
    fc = s7["forecast"]
    rows = [{"week": d.date(), "series": "Actual", "revenue": round(float(v), 2)} for d, v in weekly.items()]
    last = weekly.index[-1].date(), round(float(weekly.iloc[-1]), 2)
    # every forecast line starts at the last actual week so the lines join up
    for series in ("Forecast", "80% range low", "80% range high"):
        rows.append({"week": last[0], "series": series, "revenue": last[1]})
    for (d, med), (lo, hi) in zip(fc["weekly_median"].items(), fc["weekly_80"]):
        rows += [{"week": pd.Timestamp(d).date(), "series": "Forecast", "revenue": med},
                 {"week": pd.Timestamp(d).date(), "series": "80% range low", "revenue": lo},
                 {"week": pd.Timestamp(d).date(), "series": "80% range high", "revenue": hi}]
    tables["forecast"] = rows

    tables["lift"] = [
        {"metric": label, "method": method, "lift": src[key]["rel_lift"],
         "ci_low": src[key]["rel_ci"][0], "ci_high": src[key]["rel_ci"][1]}
        for label, key in (("Visits", "visit"), ("Conversions", "conversion"))
        for method, src in (("Simple comparison", s5["lift"]), ("Adjusted (CUPAC)", s5["cupac"]))]

    pol, best = s6["policy"], s6["best_model"]
    ks = [0, 10, 20, 30, 50, 100]
    rows = [{"share_targeted": k / 100, "model": "Random targeting", "share_captured": k / 100} for k in ks]
    for model, label in (("Response model", "Target likely visitors"), (best, f"Uplift ({best})")):
        for k in ks:
            v = 0 if k == 0 else 1 if k == 100 else pol[model][f"share_of_incremental_at_{k}"]
            rows.append({"share_targeted": k / 100, "model": label, "share_captured": v})
    tables["targeting"] = rows

    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in tables.items():
        with open(OUT_DIR / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return s3, s4, s5, s6, s7


# column name -> tableau type, per data source
SOURCES = {
    "buyers": ("Active buyers", [("order_month", "date"), ("buyer_type", "string"), ("buyers", "integer")]),
    "funnel": ("Session funnel", [("step", "string"), ("step_order", "integer"), ("sessions", "integer"),
                                  ("carry_on", "string"), ("highlight", "string")]),
    "segments": ("RFM segments", [("segment", "string"), ("segment_order", "integer"), ("measure", "string"),
                                  ("share", "real"), ("customers", "integer"), ("value_12m", "real")]),
    "forecast": ("Revenue forecast", [("week", "date"), ("series", "string"), ("revenue", "real")]),
    "lift": ("Ad lift", [("metric", "string"), ("method", "string"), ("lift", "real"),
                         ("ci_low", "real"), ("ci_high", "real")]),
    "targeting": ("Uplift targeting", [("share_targeted", "real"), ("model", "string"),
                                       ("share_captured", "real")]),
}
CAPTIONS = {
    "order_month": "Order date", "buyer_type": "Buyer type", "buyers": "Buyers", "step": "Step",
    "sessions": "Sessions", "segment": "Segment", "measure": "Measure", "share": "Share",
    "customers": "Customers", "value_12m": "12-month value ($)", "week": "Week ending", "series": "Series",
    "revenue": "Net revenue ($)", "metric": "Metric", "method": "Method", "lift": "Lift",
    "ci_low": "95% CI low", "ci_high": "95% CI high", "share_targeted": "Share of users shown ads",
    "model": "Targeting", "share_captured": "Share of extra visits",
}
PERCENT = {"share", "lift", "ci_low", "ci_high", "share_targeted", "share_captured"}
DOLLARS = {"revenue", "value_12m"}


def build_extracts():
    """Tableau Public only opens workbooks whose data is an extract (.hyper)."""
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        for name, (_, columns) in SOURCES.items():
            table = TableDefinition(EXTRACT_TABLE, [TableDefinition.Column(n, HYPER_TYPES[t]) for n, t in columns])
            with Connection(hyper.endpoint, OUT_DIR / f"{name}.hyper", CreateMode.CREATE_AND_REPLACE) as conn:
                conn.catalog.create_schema("Extract")
                conn.catalog.create_table(table)
                conn.execute_command(f"COPY {EXTRACT_TABLE} FROM {escape_string_literal(str(OUT_DIR / f'{name}.csv'))} "
                                     "WITH (format csv, header, NULL '')")


# ---- workbook XML -----------------------------------------------------------------

def ds_name(name):
    return f"federated.{name}"


def column_def(name, dtype):
    role, kind = ("dimension", "nominal") if dtype == "string" else (
        ("dimension", "ordinal") if dtype == "date" else ("measure", "quantitative"))
    caption = f" caption={quoteattr(CAPTIONS[name])}" if name in CAPTIONS else ""
    fmt = " default-format='p0%'" if name in PERCENT else " default-format='c\"$\"#,##0'" if name in DOLLARS else ""
    return f"<column{caption} datatype='{dtype}'{fmt} name='[{name}]' role='{role}' type='{kind}' />"


def datasource(name, colors):
    caption, columns = SOURCES[name]
    relation_cols = "".join(f"<column datatype='{t}' name='{n}' ordinal='{i}' />" for i, (n, t) in enumerate(columns))
    return f"""
  <datasource caption={quoteattr(caption)} inline='true' name='{ds_name(name)}' version='18.1'>
    <connection class='federated'>
      <named-connections>
        <named-connection caption='{name}' name='textscan.{name}'>
          <connection class='textscan' directory={quoteattr(str(OUT_DIR))} filename='{name}.csv' password='' server='' />
        </named-connection>
      </named-connections>
      <relation connection='textscan.{name}' name='{name}.csv' table='[{name}#csv]' type='table'>
        <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>{relation_cols}</columns>
      </relation>
    </connection>
    <aliases enabled='yes' />
    {"".join(column_def(n, t) for n, t in columns)}
    {"".join(instance(ref.split(":")[1], "None")[0] for ref, _ in colors.get(name, []))}
    <extract count='-1' enabled='true' units='records'>
      <connection access_mode='readonly' authentication='auth-none' author-locale='en_US' class='hyper'
                  dbname={quoteattr(str(OUT_DIR / f"{name}.hyper"))} default-settings='hyper' schema='Extract'
                  sslmode='' tablename='Extract' username='tableau_internal_user'>
        <relation name='Extract' table='[Extract].[Extract]' type='table' />
        <refresh increment-key='' incremental-updates='false' />
      </connection>
    </extract>
    <style>{"".join(color_rule(ref, c) for ref, c in colors.get(name, []))}</style>
  </datasource>"""


def instance(field, derivation):
    """(column-instance xml, reference) for a field with an aggregation or date level."""
    prefix, kind, suffix = {
        "None": ("none", "nominal", "nk"), "Sum": ("sum", "quantitative", "qk"),
        "Month-Trunc": ("tmn", "quantitative", "qk"), "Day-Trunc": ("tdy", "quantitative", "qk"),
        "Continuous": ("none", "quantitative", "qk"),  # a number used as a continuous dimension
    }[derivation]
    derivation = "None" if derivation == "Continuous" else derivation
    name = f"[{prefix}:{field}:{suffix}]"
    return (f"<column-instance column='[{field}]' derivation='{derivation}' name='{name}' "
            f"pivot='key' type='{kind}' />"), name


def color_rule(ref, colors):
    """Fixed colors per category, so the accent always means the same thing. Tableau keeps
    assigned colors on the data source, with the field written without the source name."""
    # each category is written as a quoted string, escaped for XML: &quot;New&quot;
    maps = "".join(f"<map to='{c}'><bucket>{escape(chr(34) + v + chr(34), {chr(34): '&quot;'})}</bucket></map>"
                   for v, c in colors.items())
    return (f"<style-rule element='mark'><encoding attr='color' field='{ref}' type='palette'>"
            f"{maps}</encoding></style-rule>")


COLORS = {}  # source -> [(field reference, {category: color})], filled as sheets are built


def worksheet(name, title, source, deps, rows, cols, mark, encodings, extra_view="", colors=None, labels=False,
              mark_size=None):
    if colors:
        COLORS.setdefault(source, []).append(colors)
    ds = ds_name(source)
    columns = SOURCES[source][1]
    used = "".join(x for x, _ in deps.values())
    col_defs = "".join(column_def(n, t) for n, t in columns if f"column='[{n}]'" in used)
    enc = "".join(f"<{kind} column='[{ds}].{deps[key][1]}' />" for kind, key in encodings)
    # a fixed mark size, for charts where Tableau's automatic sizing makes dots overlap
    sizing = "<mark-sizing mark-sizing-setting='marks-scaling-off' />" if mark_size else ""
    pane_style = (f"<style><style-rule element='mark'><format attr='size' value='{mark_size}' />"
                  "</style-rule></style>") if mark_size else ""
    label_style = ("<style-rule element='mark'><format attr='mark-labels-show' value='true' />"
                   "<format attr='mark-labels-cull' value='true' /></style-rule>") if labels else ""
    return f"""
    <worksheet name={quoteattr(name)}>
      <layout-options><title><formatted-text><run bold='true' fontsize='12'>{escape(title)}</run></formatted-text></title></layout-options>
      <table>
        <view>
          <datasources><datasource name='{ds}' /></datasources>
          <datasource-dependencies datasource='{ds}'>{col_defs}{used}</datasource-dependencies>
          {extra_view}
          <aggregation value='true' />
        </view>
        <style>{label_style}</style>
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view><breakdown value='auto' /></view>
            <mark class='{mark}' />
            {sizing}
            <encodings>{enc}</encodings>
            {pane_style}
          </pane>
        </panes>
        <rows>{rows}</rows>
        <cols>{cols}</cols>
      </table>
    </worksheet>"""


def sort_by(source, dim_ref, measure_ref):
    ds = ds_name(source)
    # order a dimension by a number instead of alphabetically
    return f"<sort class='computed' column='[{ds}].{dim_ref}' direction='ASC' using='[{ds}].{measure_ref}' />"


def build_sheets(s3, s4, s5, s6, s7):
    sheets = []

    d = {"month": instance("order_month", "Month-Trunc"), "type": instance("buyer_type", "None"),
         "buyers": instance("buyers", "Sum")}
    ds = ds_name("buyers")
    sheets.append(worksheet(
        "Active Buyers", "Active buyers per month (blue = returning, grey = new)", "buyers", d,
        f"[{ds}].{d['buyers'][1]}", f"[{ds}].{d['month'][1]}", "Bar",
        [("color", "type"), ("tooltip", "buyers")],
        colors=(d["type"][1], {"Returning": ACCENT, "New": GREY})))

    d = {"week": instance("week", "Day-Trunc"), "series": instance("series", "None"),
         "revenue": instance("revenue", "Sum")}
    ds = ds_name("forecast")
    fc = s7["forecast"]
    sheets.append(worksheet(
        "Revenue Forecast",
        f"Weekly net revenue and next-quarter forecast: about ${fc['quarter_median'] / 1e6:.2f}M (light lines = 80% range)",
        "forecast", d, f"[{ds}].{d['revenue'][1]}", f"[{ds}].{d['week'][1]}", "Line",
        [("color", "series"), ("tooltip", "revenue")],
        colors=(d["series"][1], {"Actual": INK_2, "Forecast": ACCENT,
                                              "80% range low": ACCENT_LIGHT, "80% range high": ACCENT_LIGHT})))

    d = {"step": instance("step", "None"), "order": instance("step_order", "Sum"),
         "sessions": instance("sessions", "Sum"), "carry": instance("carry_on", "None"),
         "hl": instance("highlight", "None")}
    ds = ds_name("funnel")
    sheets.append(worksheet(
        "Session Funnel", f"Session funnel: only {s3['funnel']['cart_to_purchase']:.0%} of carts become orders",
        "funnel", d, f"[{ds}].{d['step'][1]}", f"[{ds}].{d['sessions'][1]}", "Bar",
        [("color", "hl"), ("text", "carry"), ("tooltip", "sessions")],
        extra_view=sort_by("funnel", d["step"][1], d["order"][1]),
        colors=(d["hl"][1], {"Biggest drop": ACCENT, "Other": GREY}), labels=True))

    d = {"segment": instance("segment", "None"), "order": instance("segment_order", "Sum"),
         "measure": instance("measure", "None"), "share": instance("share", "Sum"),
         "customers": instance("customers", "Sum"), "value": instance("value_12m", "Sum")}
    ds = ds_name("segments")
    seg = s4["rfm"]["segments"]
    top_c = seg["Champions"]["customer_share"] + seg["Loyal"]["customer_share"]
    top_r = seg["Champions"]["revenue_share"] + seg["Loyal"]["revenue_share"]
    sheets.append(worksheet(
        "Customer Segments",
        f"Champions + Loyal: {top_c:.0%} of buyers, {top_r:.0%} of revenue (blue dot = revenue, grey dot = customers)",
        # segments run across and share goes up, one column with two dots per segment.
        # stacked as rows, seven segment names didn't fit the short dashboard tile
        "segments", d, f"[{ds}].{d['share'][1]}", f"[{ds}].{d['segment'][1]}", "Circle",
        [("color", "measure"), ("tooltip", "customers"), ("tooltip", "value")],
        extra_view=sort_by("segments", d["segment"][1], d["order"][1]),
        colors=(d["measure"][1], {"Share of revenue": ACCENT, "Share of customers": GREY}), mark_size=0.5))

    d = {"metric": instance("metric", "None"), "method": instance("method", "None"),
         "lift": instance("lift", "Sum"), "lo": instance("ci_low", "Sum"), "hi": instance("ci_high", "Sum")}
    ds = ds_name("lift")
    sheets.append(worksheet(
        "Ad Lift",
        f"Ads lifted visits {s5['cupac']['visit']['rel_lift']:.0%}, not {s5['lift']['visit']['rel_lift']:.0%} "
        "(blue = adjusted, grey = simple comparison)",
        "lift", d, f"[{ds}].{d['metric'][1]}", f"[{ds}].{d['lift'][1]}", "Circle",
        [("color", "method"), ("tooltip", "lo"), ("tooltip", "hi")],
        colors=(d["method"][1], {"Adjusted (CUPAC)": ACCENT, "Simple comparison": GREY})))

    best = s6["best_model"]
    d = {"x": instance("share_targeted", "Continuous"), "model": instance("model", "None"),
         "y": instance("share_captured", "Sum")}
    ds = ds_name("targeting")
    sheets.append(worksheet(
        "Uplift Targeting",
        f"Top 10% by uplift: {s6['policy'][best]['share_of_incremental_at_10']:.0%} of the extra visits "
        f"(blue = uplift, grey = likely visitors)",
        "targeting", d, f"[{ds}].{d['y'][1]}", f"[{ds}].{d['x'][1]}", "Line",
        [("color", "model")],
        colors=(d["model"][1], {f"Uplift ({best})": ACCENT, "Target likely visitors": GREY,
                                             "Random targeting": "#d9d8d3"})))
    return sheets


def build_dashboard(s3, s5, s7):
    findings = ("Buyers doubled in a year. Checkout is the biggest leak "
                f"({s3['funnel']['cart_to_purchase']:.0%} of carts become orders). Ads lifted visits "
                f"{s5['cupac']['visit']['rel_lift']:.0%}, not {s5['lift']['visit']['rel_lift']:.0%}.")
    tiles = [("Active Buyers", 0, 11000), ("Revenue Forecast", 50000, 11000),
             ("Session Funnel", 0, 40000), ("Customer Segments", 50000, 40000),
             ("Ad Lift", 0, 69000), ("Uplift Targeting", 50000, 69000)]
    zones = "".join(f"<zone h='29000' id='{10 + i}' name={quoteattr(n)} w='50000' x='{x}' y='{y}' />"
                    for i, (n, x, y) in enumerate(tiles))
    return f"""
    <dashboard name='E-commerce Product Analytics'>
      <style />
      <size sizing-mode='automatic' />
      <zones>
        <zone h='100000' id='1' type-v2='layout-basic' w='100000' x='0' y='0'>
          <zone h='5000' id='2' type-v2='text' w='100000' x='0' y='0'>
            <formatted-text><run bold='true' fontsize='20'>E-commerce Product Analytics</run></formatted-text>
          </zone>
          <zone h='6000' id='3' type-v2='text' w='100000' x='0' y='5000'>
            <formatted-text><run fontsize='11'>{escape(findings)}</run></formatted-text>
          </zone>
          {zones}
          <zone h='2000' id='4' type-v2='text' w='100000' x='0' y='98000'>
            <formatted-text><run fontsize='9'>Data: Google's thelook_ecommerce (to 2026-10-22) and Criteo's uplift experiment (14M users). Code: github.com/apushp/ecommerce-product-analytics</run></formatted-text>
          </zone>
        </zone>
      </zones>
    </dashboard>"""


def build():
    s3, s4, s5, s6, s7 = export_data()
    build_extracts()
    sheets = build_sheets(s3, s4, s5, s6, s7)
    names = ["Active Buyers", "Revenue Forecast", "Session Funnel", "Customer Segments", "Ad Lift",
             "Uplift Targeting"]
    # Tableau needs a window per sheet and an active (maximized) one to open on
    cards = ("<cards><edge name='left'><strip size='160'><card type='pages' /><card type='filters' />"
             "<card type='marks' /></strip></edge><edge name='top'><strip size='2147483647'>"
             "<card type='columns' /></strip><strip size='2147483647'><card type='rows' /></strip>"
             "<strip size='31'><card type='title' /></strip></edge></cards>")
    windows = "".join(f"\n    <window class='worksheet' name={quoteattr(n)}>{cards}</window>" for n in names)
    windows += ("\n    <window class='dashboard' maximized='true' name='E-commerce Product Analytics'>"
                # entire-view: each chart shrinks to fit its box instead of scrolling
                "<viewpoints>" + "".join(f"<viewpoint name={quoteattr(n)}><zoom type='entire-view' /></viewpoint>"
                                         for n in names) +
                "</viewpoints><active id='-1' /><device-preview /></window>")
    workbook = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook original-version='18.1' source-build='2025.2.0 (20252.25.0603.1006)' source-platform='mac' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <datasources>{"".join(datasource(n, COLORS) for n in SOURCES)}
  </datasources>
  <worksheets>{"".join(sheets)}
  </worksheets>
  <dashboards>{build_dashboard(s3, s5, s7)}
  </dashboards>
  <windows source-height='30'>{windows}
  </windows>
</workbook>
"""
    OUT.write_text(workbook)
    print(f"wrote {OUT.relative_to(ROOT)} and {len(SOURCES)} extracts")


if __name__ == "__main__":
    build()
