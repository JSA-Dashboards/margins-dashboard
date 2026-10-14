"""
Weekly EIA view of the Ethanol page: the JSA-style Weekly EIA Report (ethanol + petroleum weekly
status). Ported from the EIA Energy Dashboard's "Weekly EIA Report" section, with two changes:
it reads straight from the EIA API (the dashboard reads its ethanol routes through a Snowflake
cache that this app has no access to) and it only pulls history back to HISTORY_START, which is
what the 6-marketing-year charts and the 5-year averages need.

It differs from the dashboard in one more place on purpose: propane stocks read process SAXP, because the
SAE series the dashboard uses stopped in April 2020 and would show a 2020 figure as this week's.
Don't re-sync that line from the dashboard.

Data: EIA Open Data API v2, weekly petroleum status report routes (see eia_common.eia_get).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eia_common import (COL_ETHANOL, DM_BORDER, DM_MUTED, DM_SURFACE, NEG, POS, eia_get, eia_get_many,
                        inject_css, section_header, snap_anchor, snap_toolbar, style_fig)

HISTORY_START = "2016-01-01"

PADD_CODES = ["R10", "R20", "R30", "R40", "R50"]
PADD_LABELS = {"R10": "East Coast", "R20": "Midwest", "R30": "Gulf Coast",
               "R40": "Rocky Mountains", "R50": "West Coast"}
PADD_MOVE_CODES = {"R10": "R10-Z00", "R20": "R20-Z00", "R30": "R30-Z00",
                   "R40": "R40-Z00", "R50": "R50-Z00"}


# ── small helpers (pure) ──────────────────────────────────────────────────────
def wow(df):
    """(this_period, this_val, last_val, delta) for the two most recent weeks."""
    if df.empty:
        return None, None, None, None
    s = df.sort_values("period").dropna(subset=["value"])
    if len(s) < 2:
        return None, None, None, None
    this_row, last_row = s.iloc[-1], s.iloc[-2]
    return this_row["period"], this_row["value"], last_row["value"], this_row["value"] - last_row["value"]


def padd_wow(df, padd_col="duoarea"):
    """{area: (this_val, last_val, delta)} for the two most recent weeks per area."""
    out = {}
    if df.empty:
        return out
    for area, sub in df.groupby(padd_col):
        _, tv, lv, d = wow(sub)
        if tv is not None:
            out[area] = (tv, lv, d)
    return out


def mover_phrase(deltas: dict, label_map: dict, unit="k bpd", decimals=0):
    """Turn {area_code: delta} into a natural-language driver sentence fragment."""
    items = [(label_map.get(k, k), v) for k, v in deltas.items() if v is not None and abs(v) >= 0.5]
    if not items:
        return "with little change across regions"
    ups = sorted([it for it in items if it[1] > 0], key=lambda x: -x[1])
    downs = sorted([it for it in items if it[1] < 0], key=lambda x: x[1])

    def fmt_group(grp):
        return ", ".join(f"{name} ({v:+,.{decimals}f}{unit})" for name, v in grp)

    if ups and downs:
        lead, other = (ups, downs) if sum(v for _, v in ups) >= -sum(v for _, v in downs) else (downs, ups)
        verb = "gains" if lead is ups else "declines"
        other_verb = "declines" if lead is ups else "gains"
        return f"as {other_verb} in {fmt_group(other)} were outweighed by {verb} in {fmt_group(lead)}"
    elif ups:
        return f"led by gains in {fmt_group(ups)}"
    return f"led by declines in {fmt_group(downs)}"


def marketing_year_week(ts: pd.Timestamp):
    my_start_year = ts.year if ts.month >= 9 else ts.year - 1
    my_start = pd.Timestamp(year=my_start_year, month=9, day=1)
    week = ((ts - my_start).days // 7) + 1
    return f"{my_start_year}-{str(my_start_year + 1)[-2:]}", week


def seasonal_status(df, current_val, current_period):
    """Is current_val a seasonal (marketing-year-to-date) high or low?"""
    if df.empty or current_val is None:
        return ""
    my_label, _ = marketing_year_week(current_period)
    s = df.copy()
    s["my_label"] = s["period"].apply(lambda p: marketing_year_week(p)[0])
    ytd = s[(s["my_label"] == my_label) & (s["period"] <= current_period)]
    if ytd.empty:
        return ""
    if current_val >= ytd["value"].max():
        return " to a seasonal high"
    if current_val <= ytd["value"].min():
        return " to a seasonal low"
    return ""


def pct_vs_5yr_avg(df, current_val, current_period):
    if df.empty or current_val is None:
        return None
    s = df.copy()
    s["year"] = s["period"].dt.year
    s["doy"] = s["period"].dt.dayofyear
    band_years = [current_period.year - n for n in range(1, 6)]
    hist = s[s["year"].isin(band_years)]
    near = hist.iloc[(hist["doy"] - current_period.dayofyear).abs().argsort()[:5]] if not hist.empty else hist
    if near.empty:
        return None
    avg = near["value"].mean()
    return (current_val / avg - 1) * 100 if avg else None


def build_padd_block(title, padd_data, unit, decimals=0, snap_id=""):
    rows_html = ""
    nat = padd_data.get("NUS")
    for code in PADD_CODES:
        if code not in padd_data:
            continue
        tv, lv, d = padd_data[code]
        rows_html += (f"<tr><td style='padding:4px 10px'>{PADD_LABELS[code]}</td>"
                      f"<td style='padding:4px 10px;text-align:right'>{tv:,.{decimals}f}</td>"
                      f"<td style='padding:4px 10px;text-align:right'>{lv:,.{decimals}f}</td>"
                      f"<td style='padding:4px 10px;text-align:right;color:{POS if d >= 0 else NEG}'>{d:+,.{decimals}f}</td></tr>")
    total_html = ""
    if nat:
        tv, lv, d = nat
        total_html = (f"<tr style='border-top:1px solid {DM_BORDER};font-weight:700'>"
                      f"<td style='padding:4px 10px'>U.S. Total</td>"
                      f"<td style='padding:4px 10px;text-align:right'>{tv:,.{decimals}f}</td>"
                      f"<td style='padding:4px 10px;text-align:right'>{lv:,.{decimals}f}</td>"
                      f"<td style='padding:4px 10px;text-align:right;color:{POS if d >= 0 else NEG}'>{d:+,.{decimals}f}</td></tr>")
    header_row = (f"<tr style='color:{DM_MUTED};font-size:0.72rem;text-transform:uppercase'>"
                  f"<td style='padding:4px 10px'>{title}</td>"
                  f"<td style='padding:4px 10px;text-align:right'>This Week</td>"
                  f"<td style='padding:4px 10px;text-align:right'>Last Week</td>"
                  f"<td style='padding:4px 10px;text-align:right'>Change</td></tr>")
    return (f"<table id='{snap_id}' style='width:100%;border-collapse:collapse;background:{DM_SURFACE};"
            f"border:1px solid {DM_BORDER};border-radius:6px;margin-bottom:6px'>"
            f"{header_row}{rows_html}{total_html}</table>"
            f"<div class='note-text' style='margin-bottom:16px'>{unit}</div>")


def seasonal_chart(df, title, yaxis_title, snap_id):
    if df.empty:
        st.info(f"No data for {title}.")
        return
    s = df.copy()
    s[["my_label", "week"]] = s["period"].apply(lambda p: pd.Series(marketing_year_week(p)))
    years = sorted(s["my_label"].unique())[-6:]
    current_my = years[-1] if years else None
    fig = go.Figure()
    palette = ["#8a7233", "#3fa9f5", "#5ec48c", "#8db8e0", "#e0716f", "#f2b134"]
    for i, my in enumerate(years):
        sub = s[s["my_label"] == my].sort_values("week")
        is_current = my == current_my
        fig.add_trace(go.Scatter(
            x=sub["week"], y=sub["value"], mode="lines+markers" if is_current else "lines",
            name=my, line=dict(width=3 if is_current else 1.5,
                               color=COL_ETHANOL if is_current else palette[i % len(palette)]),
            marker=dict(size=5) if is_current else None,
        ))
    style_fig(fig, yaxis_title=yaxis_title)
    fig.update_xaxes(title="Marketing-Year Week")
    snap_anchor(snap_id)
    st.plotly_chart(fig, width="stretch")
    st.caption(title)
    snap_toolbar(snap_id, title)


# ── the view ──────────────────────────────────────────────────────────────────
def render_weekly_eia():
    inject_css()

    head, btn = st.columns([6, 1])
    with btn:
        if st.button("🔄 Refresh", key="weia_refresh", width="stretch",
                     help="Pull fresh EIA data now (otherwise cached for an hour)."):
            eia_get.clear()
            st.rerun()

    S = HISTORY_START
    sndw_padd = {"duoarea": PADD_CODES + ["NUS"], "product": "EPOOXE"}
    move_padd = {"duoarea": list(PADD_MOVE_CODES.values()) + ["NUS-Z00"], "product": "EPOOXE"}
    with st.spinner("Loading EIA weekly data… the first load of each hour can take up to a minute."):
        d = eia_get_many({
            "eth_prod_padd": ("petroleum/sum/sndw", {**sndw_padd, "process": "YOP"}, S),
            "eth_stock_padd": ("petroleum/sum/sndw", {**sndw_padd, "process": "SAE"}, S),
            "eth_blend_padd": ("petroleum/sum/sndw", {**sndw_padd, "process": "YIR"}, S),
            "eth_imports_padd": ("petroleum/move/wkly", {**move_padd, "process": "IM0"}, S),
            "eth_exports_nat": ("petroleum/move/wkly",
                                {"duoarea": "NUS-Z00", "product": "EPOOXE", "process": "EEX"}, S),
            "gas_demand": ("petroleum/cons/wpsup", {"duoarea": "NUS", "product": "EPM0F"}, S),

            "crude_stock": ("petroleum/sum/sndw", {"duoarea": "NUS", "product": "EPC0", "process": "SAX"}, S),
            "cushing_stock": ("petroleum/sum/sndw", {"duoarea": "YCUOK", "product": "EPC0", "process": "SAX"}, S),
            "gasoline_stock": ("petroleum/sum/sndw", {"duoarea": "NUS", "product": "EPM0", "process": "SAE"}, S),
            "distillate_stock": ("petroleum/sum/sndw", {"duoarea": "NUS", "product": "EPD0", "process": "SAE"}, S),
            # SAXP, not SAE: EIA re-coded the weekly propane stocks (excluding propylene at terminals) in
            # April 2020 and the old SAE series ends 2020-04-03. The EIA Energy Dashboard still reads that one.
            "propane_stock": ("petroleum/sum/sndw", {"duoarea": "NUS", "product": "EPLLPZ", "process": "SAXP"}, S),

            "crude_input": ("petroleum/pnp/wiup", {"duoarea": "NUS", "product": "EPC0", "process": "YIY"}, S),
            "refinery_capacity": ("petroleum/pnp/wiup", {"duoarea": "NUS", "process": "YRL"}, S),
            "gasoline_prod": ("petroleum/pnp/wprodrb", {"duoarea": "NUS", "product": "EPM0F", "process": "YPR"}, S),
            "distillate_prod": ("petroleum/pnp/wprodrb", {"duoarea": "NUS", "product": "EPD0", "process": "YPR"}, S),
        })
    eth_prod_padd, eth_stock_padd, eth_blend_padd = d["eth_prod_padd"], d["eth_stock_padd"], d["eth_blend_padd"]
    eth_imports_padd, eth_exports_nat, gas_demand = d["eth_imports_padd"], d["eth_exports_nat"], d["gas_demand"]
    crude_stock, cushing_stock, gasoline_stock = d["crude_stock"], d["cushing_stock"], d["gasoline_stock"]
    distillate_stock, propane_stock = d["distillate_stock"], d["propane_stock"]
    crude_input, refinery_capacity = d["crude_input"], d["refinery_capacity"]
    gasoline_prod, distillate_prod = d["gasoline_prod"], d["distillate_prod"]

    if eth_prod_padd.empty:
        st.info("No weekly ethanol data returned from EIA.")
        return

    eth_prod_nat = eth_prod_padd[eth_prod_padd["duoarea"] == "NUS"]
    eth_stock_nat = eth_stock_padd[eth_stock_padd["duoarea"] == "NUS"] if not eth_stock_padd.empty else pd.DataFrame()
    eth_blend_nat = eth_blend_padd[eth_blend_padd["duoarea"] == "NUS"] if not eth_blend_padd.empty else pd.DataFrame()

    prod_period, prod_this, prod_last, prod_delta = wow(eth_prod_nat)
    stock_period, stock_this, stock_last, stock_delta = wow(eth_stock_nat)
    blend_period, blend_this, blend_last, blend_delta = wow(eth_blend_nat)
    _, exp_this, exp_last, exp_delta = wow(eth_exports_nat)
    _, gas_this, gas_last, gas_delta = wow(gas_demand)

    dos_this = (stock_this / blend_this) if (stock_this and blend_this) else None
    dos_last = (stock_last / blend_last) if (stock_last and blend_last) else None
    dos_delta = (dos_this - dos_last) if (dos_this is not None and dos_last is not None) else None

    prod_deltas_padd = {k: v[2] for k, v in padd_wow(eth_prod_padd).items() if k in PADD_LABELS}
    stock_deltas_padd = {k: v[2] for k, v in padd_wow(eth_stock_padd).items() if k in PADD_LABELS}

    report_date = f"{prod_period.strftime('%B')} {prod_period.day}" if prod_period is not None else "this week"
    prod_seasonal = seasonal_status(eth_prod_nat, prod_this, prod_period) if prod_period is not None else ""
    stock_seasonal = seasonal_status(eth_stock_nat, stock_this, stock_period) if stock_period is not None else ""

    st.subheader("Weekly EIA Report")
    if prod_period is not None:
        st.caption(f"Week ending {prod_period.strftime('%B')} {prod_period.day}, {prod_period.year} "
                   f"— John Stewart & Associates")

    section_header("Summary")
    if stock_delta is not None:
        st.markdown(
            f"**EIA Fuel Ethanol Stocks:** {'Increased' if stock_delta >= 0 else 'Decreased'} "
            f"{abs(stock_delta):,.0f}k to {stock_this / 1000:,.3f} million barrels"
        )
    if prod_delta is not None:
        st.markdown(
            f"**EIA Fuel Ethanol Production:** {'Increased' if prod_delta >= 0 else 'Decreased'} "
            f"{abs(prod_delta):,.0f}k to {prod_this:,.0f}k barrels/day"
        )

    if prod_delta is not None and stock_delta is not None:
        narrative = (
            f"Ethanol production {'increased' if prod_delta >= 0 else 'decreased'} "
            f"{abs(prod_delta):,.0f}k bpd{prod_seasonal} of {prod_this:,.0f}k bpd for the week ending "
            f"{report_date}, {mover_phrase(prod_deltas_padd, PADD_LABELS)}. "
            f"Ethanol stocks {'also increased' if stock_delta >= 0 else 'decreased'} "
            f"{abs(stock_delta):,.0f}k barrels on the week{stock_seasonal} of "
            f"{stock_this / 1000:,.3f} million barrels, "
            f"{mover_phrase(stock_deltas_padd, PADD_LABELS, unit='k bbl')}. "
        )
        if gas_delta is not None:
            narrative += (
                f"Gasoline demand {'bounced back, increasing' if gas_delta >= 0 else 'eased,  decreasing'} "
                f"{abs(gas_delta):,.0f}k bpd on the week to {gas_this:,.0f}k bpd. "
            )
        if exp_delta is not None:
            narrative += (
                f"Ethanol exports {'increased' if exp_delta >= 0 else 'decreased'} by {abs(exp_delta):,.0f}k bpd "
                f"to {exp_this:,.0f}k bpd. "
            )
        if dos_delta is not None:
            narrative += (
                f"Days of ethanol supply {'increased' if dos_delta >= 0 else 'decreased'} by {abs(dos_delta):.2f} "
                f"days this week to {dos_this:.2f} days."
            )
        st.markdown(narrative)
        st.markdown("<div class='note-text'>Days of supply = national ending stocks ÷ national blender net "
                    "input — a JSA-style estimate, not an EIA-published series.</div>", unsafe_allow_html=True)

    # ── PADD breakdown tables ────────────────────────────────────────────────
    section_header("EIA Weekly Fuel Ethanol — PADD Breakdown")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(build_padd_block("Ethanol Production", padd_wow(eth_prod_padd), "Thousand Barrels / Day",
                                     snap_id="snap_padd_prod"), unsafe_allow_html=True)
        snap_toolbar("snap_padd_prod", "Ethanol Production by PADD")
        st.markdown(build_padd_block("Ethanol Blender Demand", padd_wow(eth_blend_padd), "Thousand Barrels / Day",
                                     snap_id="snap_padd_blend"), unsafe_allow_html=True)
        snap_toolbar("snap_padd_blend", "Ethanol Blender Demand by PADD")
    with col_b:
        imports_data = {k: v for area, v in padd_wow(eth_imports_padd).items()
                        for k, mv in PADD_MOVE_CODES.items() if area == mv}
        imports_data["NUS"] = padd_wow(eth_imports_padd).get("NUS-Z00")
        st.markdown(build_padd_block("Ethanol Imports", imports_data, "Thousand Barrels / Day",
                                     snap_id="snap_padd_imports"), unsafe_allow_html=True)
        snap_toolbar("snap_padd_imports", "Ethanol Imports by PADD")
        stock_padd_data = {k: (v[0] / 1000, v[1] / 1000, v[2] / 1000)
                           for k, v in padd_wow(eth_stock_padd).items()}
        st.markdown(build_padd_block("Fuel Ethanol Stocks", stock_padd_data, "Million Barrels", decimals=1,
                                     snap_id="snap_padd_stocks"), unsafe_allow_html=True)
        snap_toolbar("snap_padd_stocks", "Fuel Ethanol Stocks by PADD")

    # ── Seasonal charts ──────────────────────────────────────────────────────
    section_header("Seasonal Charts (Marketing Year, Sept–Aug)")
    col_a, col_b = st.columns(2)
    with col_a:
        seasonal_chart(eth_prod_nat, "Weekly Ethanol Plant Production", "Thousand bbl/day", "snap_seasonal_prod")
    with col_b:
        seasonal_chart(eth_stock_nat, "Weekly U.S. Ending Stocks of Fuel Ethanol", "Thousand bbl",
                       "snap_seasonal_stock")

    # ── Refinery narrative + petroleum stocks ────────────────────────────────
    section_header("Refinery Operations")
    _, ci_this, ci_last, ci_delta = wow(crude_input)
    _, cap_this, cap_last, _ = wow(refinery_capacity)
    ut_this = (ci_this / cap_this * 100) if (ci_this and cap_this) else None
    ut_last = (ci_last / cap_last * 100) if (ci_last and cap_last) else None
    ut_delta = (ut_this - ut_last) if (ut_this is not None and ut_last is not None) else None
    _, gp_this, gp_last, gp_delta = wow(gasoline_prod)
    _, dp_this, dp_last, dp_delta = wow(distillate_prod)
    if ci_this is not None:
        refinery_narrative = (
            f"U.S. crude oil refinery inputs averaged {ci_this / 1000:,.1f} million barrels per day for the week "
            f"ending {report_date}, {abs(ci_delta):,.0f}k bpd {'more' if ci_delta >= 0 else 'less'} than the "
            f"previous week. "
        )
        if ut_this is not None:
            refinery_narrative += (
                f"Refineries operated at {ut_this:.1f}% of operable capacity"
                f"{f' ({ut_delta:+.1f} pts WoW)' if ut_delta is not None else ''}. "
            )
        if gp_this is not None:
            refinery_narrative += (
                f"Gasoline production {'increased' if gp_delta >= 0 else 'decreased'} to "
                f"{gp_this / 1000:,.1f} million bpd. "
            )
        if dp_this is not None:
            refinery_narrative += (
                f"Distillate fuel production {'increased' if dp_delta >= 0 else 'decreased'} to "
                f"{dp_this / 1000:,.1f} million bpd."
            )
        st.markdown(refinery_narrative)

    section_header("EIA Petroleum Stocks")
    df_map = {"Comm. Crude Oil": crude_stock, "Cushing, OK": cushing_stock,
              "Total Motor Gasoline": gasoline_stock, "Distillate Fuel Oil": distillate_stock,
              "Propane/Propylene": propane_stock}
    rows_html = (f"<tr style='color:{DM_MUTED};font-size:0.72rem;text-transform:uppercase'>"
                 f"<td style='padding:4px 10px'>Million Barrels</td>"
                 f"<td style='padding:4px 10px;text-align:right'>This Week</td>"
                 f"<td style='padding:4px 10px;text-align:right'>Last Week</td>"
                 f"<td style='padding:4px 10px;text-align:right'>Change</td>"
                 f"<td style='padding:4px 10px;text-align:right'>vs 5-Yr Avg</td></tr>")
    # a series EIA has stopped publishing must not pass for this week's number: skip it and say so
    stocks_week = max((p for p in (wow(df)[0] for df in df_map.values()) if p is not None), default=None)
    stale = []
    for label, df in df_map.items():
        per, tv, lv, d = wow(df)
        if tv is None:
            continue
        if per < stocks_week:
            stale.append(label)
            continue
        pct5 = pct_vs_5yr_avg(df, tv, per)
        pct5_html = f"{pct5:+.0f}%" if pct5 is not None else "—"
        rows_html += (f"<tr><td style='padding:4px 10px'>{label}</td>"
                      f"<td style='padding:4px 10px;text-align:right'>{tv / 1000:,.1f}</td>"
                      f"<td style='padding:4px 10px;text-align:right'>{lv / 1000:,.1f}</td>"
                      f"<td style='padding:4px 10px;text-align:right;color:{POS if d >= 0 else NEG}'>{d / 1000:+,.1f}</td>"
                      f"<td style='padding:4px 10px;text-align:right'>{pct5_html}</td></tr>")
    st.markdown(
        f"<table id='snap_petro_stocks' style='width:100%;border-collapse:collapse;background:{DM_SURFACE};"
        f"border:1px solid {DM_BORDER};border-radius:6px'>{rows_html}</table>",
        unsafe_allow_html=True,
    )
    snap_toolbar("snap_petro_stocks", "EIA Petroleum Stocks")
    if stale:
        st.caption(f"Not shown: {', '.join(stale)} — EIA's newest figure for it is older than this week's report.")

    petro_narrative_parts = []
    for label, df in [("crude oil", crude_stock), ("total motor gasoline", gasoline_stock),
                      ("distillate fuel", distillate_stock), ("propane/propylene", propane_stock)]:
        per, tv, lv, d = wow(df)
        if tv is None or per < stocks_week:
            continue
        pct5 = pct_vs_5yr_avg(df, tv, per)
        clause = (f"{label.capitalize()} inventories {'increased' if d >= 0 else 'decreased'} by "
                  f"{abs(d) / 1000:,.1f} million barrels last week")
        if pct5 is not None:
            clause += f" and are {abs(pct5):.0f}% {'above' if pct5 >= 0 else 'below'} the five-year average"
        petro_narrative_parts.append(clause + ".")
    if petro_narrative_parts:
        st.markdown(" ".join(petro_narrative_parts))

    st.markdown("---")
