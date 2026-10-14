"""
Ethanol Exports view of the Ethanol page: U.S. fuel ethanol exports by destination.

Source: EIA Open Data API v2.
  * petroleum/move/expc  (product EPOOXE, "Fuel Ethanol"): monthly exports by destination country.
    EIA builds it from U.S. Census Bureau export statistics (ethyl alcohol, HS 2207, non-beverage);
    checked 2026-10-14 against Census HS-10 data: the national total agrees within 0.1% every month
    and destinations match to the decimal. It runs about a month behind Census.
  * petroleum/move/wkly  (EEX, NUS-Z00): the weekly national total, a preliminary estimate.

Two traps handled here (both found while building this):
  * every series comes twice, in thousand barrels (MBBL, the month's total) and as a daily rate
    (MBBL/D); adding both overstates each month by about 3%, so only MBBL is used;
  * the destinations add up to EIA's published monthly total (NUS-Z00) to within ~1.5% in the worst
    month, so the headline is the published total and any gap is shown as "All other".
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eia_common import (COL_BIODSL, COL_ETHANOL, DM_BORDER, DM_MUTED, DM_SURFACE, DM_TEXT, JSA_BLUE, NEG, POS,
                        eia_get, eia_get_many, fmt_num, inject_css, latest_and_delta, section_header,
                        snap_anchor, snap_toolbar, style_fig)

KBBL_TO_MGAL = 42 / 1000       # thousand barrels -> million gallons (42 gallons per barrel)
DEST_COLORS = ["#f2b134", "#3fa9f5", "#5ec48c", "#e0716f", "#8db8e0", "#c792ea", "#1f6fb2", "#c9a24b"]
DEST_OTHER_COLOR = "#b8c2cc"
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKLY_FACETS = {"duoarea": "NUS-Z00", "product": "EPOOXE", "process": "EEX"}


# ── data prep (pure) ──────────────────────────────────────────────────────────
def ethanol_export_frames(raw: pd.DataFrame):
    """Split EIA's expc fuel-ethanol rows into (official monthly total, month x destination table),
    both in thousand barrels. See the module docstring for why only MBBL rows are kept."""
    d = raw[raw["units"] == "MBBL"].copy()
    total = d[d["duoarea"] == "NUS-Z00"].groupby("period")["value"].sum().sort_index()
    dest = d[d["duoarea"] != "NUS-Z00"].copy()
    name = dest["series-description"].str.extract(r"Exports to (.+?) of Fuel Ethanol")[0]
    dest["country"] = name.fillna(dest["area-name"])
    wide = dest.pivot_table(index="period", columns="country", values="value", aggfunc="sum").sort_index()
    return total, wide


def weekly_vs_monthly_gaps(total: pd.Series, wk: pd.DataFrame, n_months: int = 6) -> list:
    """% by which the weekly estimates' monthly average differed from EIA's monthly figure (as a daily
    rate), for the last n months that have both and at least 3 weeks."""
    if wk.empty or total.empty:
        return []
    kbd = total / pd.Series(total.index.days_in_month, index=total.index)
    w = wk[["period", "value"]].dropna()
    w = w.assign(m=w["period"].dt.to_period("M").dt.to_timestamp()).groupby("m")["value"].agg(["mean", "count"])
    return [(w.loc[m, "mean"] / kbd[m] - 1) * 100
            for m in kbd.index[-n_months:] if m in w.index and w.loc[m, "count"] >= 3 and kbd[m]]


def _chg(a, b):
    """% change a vs b; 'new' when b is zero and a is not; None when it can't be said."""
    if a is None or b is None:
        return None
    if b == 0:
        return "new" if a > 0 else None
    return (a / b - 1) * 100


def build_export_compare_table(total, wide, countries, snap_id, factor, decimals, unit_note):
    """House-style comparison table: last 4 months, same month last year, M/M and Y/Y %, trailing 12
    months with each destination's share, and YTD vs YTD last year. `total` is the official monthly
    total and `wide` the month x destination table, both in thousand barrels; `factor` converts to the
    display unit. Whatever the listed destinations don't cover (smaller markets, any rounding gap) is
    'All other'. The id sits on a wrapper so the PNG button captures the whole table."""
    latest = total.index.max()
    months = [latest - pd.DateOffset(months=n) for n in (3, 2, 1, 0)]
    ly_month = latest - pd.DateOffset(months=12)
    t12_idx = pd.date_range(end=latest, periods=12, freq="MS")
    ytd_idx = pd.date_range(start=pd.Timestamp(latest.year, 1, 1), end=latest, freq="MS")
    ytd_ly_idx = ytd_idx - pd.DateOffset(years=1)

    listed = wide.reindex(total.index)[countries].fillna(0)
    rows = {c: listed[c] for c in countries}
    rows["All other"] = (total - listed.sum(axis=1)).clip(lower=0)
    rows["Total"] = total

    def at(s, p):
        v = s.get(p)
        return None if v is None or pd.isna(v) else float(v) * factor

    def over(s, idx):
        return float(s.reindex(idx).fillna(0).sum()) * factor

    def fmt_val(v):
        return "—" if v is None or v == 0 else f"{v:,.{decimals}f}"

    def fmt_chg(c):
        if c is None:
            return f"<span style='color:{DM_MUTED}'>—</span>"
        if c == "new":
            return f"<span style='color:{JSA_BLUE}'>new</span>"
        return f"<span style='color:{POS if c >= 0 else NEG}'>{c:+.1f}%</span>"

    t12_total = over(total, t12_idx)
    yy = str(latest.year)[-2:]
    headers = (["Destination"] + [m.strftime("%b '%y") for m in months] +
               [f"{ly_month.strftime('%b')} '{str(ly_month.year)[-2:]} (LY)", "M/M %", "Y/Y %", "12-Mo", "Share",
                f"YTD '{yy}", f"YTD '{str(latest.year - 1)[-2:]}", "YTD %"])
    header_html = "".join(
        f"<th style='padding:4px 8px;text-align:{'left' if i == 0 else 'right'};color:{DM_MUTED};"
        f"font-size:0.68rem;text-transform:uppercase;white-space:nowrap'>{h}</th>"
        for i, h in enumerate(headers))

    body = ""
    for label, s in rows.items():
        vals = [at(s, m) for m in months]
        ly_val = at(s, ly_month)
        t12 = over(s, t12_idx)
        ytd, ytd_ly = over(s, ytd_idx), over(s, ytd_ly_idx)
        bold = f"font-weight:700;border-top:1px solid {DM_BORDER};" if label == "Total" else ""
        td = f"padding:4px 8px;text-align:right;{bold}"
        cells = [f"<td style='padding:4px 8px;{bold}'>{label}</td>"]
        cells += [f"<td style='{td}'>{fmt_val(v)}</td>" for v in vals]
        cells += [f"<td style='{td}'>{fmt_val(ly_val)}</td>",
                  f"<td style='{td}'>{fmt_chg(_chg(vals[-1], vals[-2]))}</td>",
                  f"<td style='{td}'>{fmt_chg(_chg(vals[-1], ly_val))}</td>",
                  f"<td style='{td}'>{fmt_val(t12)}</td>",
                  f"<td style='{td}'>{(f'{t12 / t12_total * 100:.0f}%' if t12_total and t12 else '—')}</td>",
                  f"<td style='{td}'>{fmt_val(ytd)}</td>",
                  f"<td style='{td}'>{fmt_val(ytd_ly)}</td>",
                  f"<td style='{td}'>{fmt_chg(_chg(ytd, ytd_ly))}</td>"]
        body += f"<tr>{''.join(cells)}</tr>"

    caption = (f"{unit_note}. 12-Mo = {t12_idx[0].strftime('%b %Y')}–{latest.strftime('%b %Y')}; "
               f"YTD = Jan–{latest.strftime('%b %Y')} vs Jan–{ly_month.strftime('%b %Y')}; "
               f"Share = share of the 12-month total.")
    return (f"<div id='{snap_id}'><div style='overflow-x:auto'>"
            f"<table id='{snap_id}_tbl' style='width:100%;border-collapse:collapse;background:{DM_SURFACE};"
            f"border:1px solid {DM_BORDER};border-radius:6px;font-size:0.82rem'>"
            f"<tr>{header_html}</tr>{body}</table></div>"
            f"<div class='note-text' style='margin:4px 0 12px 0'>{caption}</div></div>")


# ── the view ──────────────────────────────────────────────────────────────────
def render_ethanol_exports():
    inject_css()
    head, btn = st.columns([6, 1])
    with btn:
        if st.button("🔄 Refresh", key="exp_refresh", width="stretch",
                     help="Pull fresh EIA data now (otherwise cached for an hour)."):
            eia_get.clear()
            st.rerun()

    with st.spinner("Loading EIA export data… the first load of each hour can take up to a minute."):
        d = eia_get_many({"raw": ("petroleum/move/expc", {"product": "EPOOXE"}, None),
                          "wk": ("petroleum/move/wkly", WEEKLY_FACETS, "2022-01-01")})
    raw, wk = d["raw"], d["wk"]
    if raw.empty or "units" not in raw.columns:
        st.info("No ethanol export data returned from EIA.")
        return
    total, wide = ethanol_export_frames(raw)
    if total.empty:
        st.info("No monthly ethanol export totals returned from EIA.")
        return

    latest = total.index.max()
    today = pd.Timestamp.today()
    lag = (today.year - latest.year) * 12 + (today.month - latest.month)
    with head:
        st.subheader("U.S. Fuel Ethanol Exports")
    st.info(
        f"📅 EIA's exports-by-destination data is monthly and published with a lag — the latest month is "
        f"**{latest.strftime('%B %Y')}**, {lag} month{'s' if lag != 1 else ''} behind today "
        f"({today.strftime('%B %Y')}). The weekly total further down is about a week behind but is a "
        f"preliminary estimate."
    )

    c1, c2 = st.columns(2)
    with c1:
        unit = st.segmented_control("Units", ["Million gallons", "Thousand barrels"],
                                    default="Million gallons", required=True, key="exp_unit")
    with c2:
        window = st.segmented_control("Chart window", ["2 years", "3 years", "5 years", "All"],
                                      default="3 years", required=True, key="exp_window")
    mgal = unit == "Million gallons"
    f = KBBL_TO_MGAL if mgal else 1.0
    dec = 1 if mgal else 0
    months_back = {"2 years": 24, "3 years": 36, "5 years": 60, "All": None}[window]
    periods = (total.index if months_back is None
               else total.index[total.index > latest - pd.DateOffset(months=months_back)])

    # ── headline numbers ─────────────────────────────────────────────────────
    ly_month = latest - pd.DateOffset(months=12)
    t12_idx = pd.date_range(end=latest, periods=12, freq="MS")
    t12_prev_idx = pd.date_range(end=ly_month, periods=12, freq="MS")
    ytd_idx = pd.date_range(start=pd.Timestamp(latest.year, 1, 1), end=latest, freq="MS")
    ytd_ly_idx = ytd_idx - pd.DateOffset(years=1)

    def tot(idx):
        return float(total.reindex(idx).fillna(0).sum())

    def pct(a, b):
        return (a / b - 1) * 100 if b else None

    cur = float(total[latest])
    ly_val = float(total[ly_month]) if ly_month in total.index else None
    t12, t12_prev = tot(t12_idx), tot(t12_prev_idx)
    ytd, ytd_ly = tot(ytd_idx), tot(ytd_ly_idx)

    def big(v):                       # 12-month / YTD totals read better in billions of gallons
        return f"{v * f / 1000:,.2f} bn gal" if mgal else f"{v:,.0f} kbbl"

    wk_df = wk[["period", "value"]].dropna().sort_values("period") if not wk.empty else pd.DataFrame()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        p = pct(cur, ly_val)
        st.metric(f"Exports, {latest.strftime('%b %Y')}",
                  f"{fmt_num(cur * f, dec)} {'Mgal' if mgal else 'kbbl'}",
                  f"{p:+.1f}% vs {ly_month.strftime('%b %Y')}" if p is not None else None)
    with m2:
        p = pct(t12, t12_prev)
        st.metric(f"12 months to {latest.strftime('%b %Y')}", big(t12),
                  f"{p:+.1f}% vs prior 12 months" if p is not None else None)
    with m3:
        p = pct(ytd, ytd_ly)
        st.metric(f"YTD (Jan–{latest.strftime('%b')})", big(ytd),
                  f"{p:+.1f}% vs last year" if p is not None else None)
    with m4:
        v, d, _ = latest_and_delta(wk_df) if not wk_df.empty else (None, None, None)
        wk_end = wk_df["period"].max().strftime("%b %d") if not wk_df.empty else ""
        st.metric("Latest week (kb/d, prelim.)", fmt_num(v, 0),
                  f"{d:+,.0f} kb/d WoW" if d is not None else None,
                  help=f"Week ending {wk_end}. EIA's weekly estimate; the monthly series above is the reference.")

    # ── monthly by destination ───────────────────────────────────────────────
    t12_rank = wide.reindex(t12_idx).fillna(0).sum().sort_values(ascending=False)
    t12_rank = t12_rank[t12_rank > 0]
    top = list(t12_rank.index[:8])
    listed = wide.reindex(total.index)[top].fillna(0)
    other = (total - listed.sum(axis=1)).clip(lower=0)

    section_header("Monthly Exports by Destination")
    fig = go.Figure()
    for i, c in enumerate(top):
        fig.add_trace(go.Bar(x=periods, y=listed.loc[periods, c] * f, name=c,
                             marker_color=DEST_COLORS[i % len(DEST_COLORS)],
                             hovertemplate=f"{c}: %{{y:,.{dec}f}}<extra></extra>"))
    fig.add_trace(go.Bar(x=periods, y=other.loc[periods] * f, name="All other", marker_color=DEST_OTHER_COLOR,
                         hovertemplate=f"All other: %{{y:,.{dec}f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=periods, y=total.loc[periods] * f, name="Total", mode="lines",
                             line=dict(color=DM_TEXT, width=1.5),
                             hovertemplate=f"Total: %{{y:,.{dec}f}}<extra></extra>"))
    fig.update_layout(barmode="stack")
    style_fig(fig, yaxis_title=f"{unit} / month")
    snap_anchor("snap_exp_monthly")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Top 8 destinations by the last 12 months; everything else is 'All other'. "
               f"The line is EIA's published monthly total.")
    snap_toolbar("snap_exp_monthly", "U.S. Fuel Ethanol Exports by Destination")

    # ── seasonal (calendar years) ────────────────────────────────────────────
    section_header("Seasonal — Total Monthly Exports by Year")
    t = total.to_frame("v")
    t["year"], t["m"] = t.index.year, t.index.month
    years = sorted(t["year"].unique())[-6:]
    palette = ["#8a7233", "#3fa9f5", "#5ec48c", "#8db8e0", "#e0716f"]
    fig = go.Figure()
    for i, y in enumerate(years):
        sub = t[t["year"] == y]
        is_cur = y == years[-1]
        fig.add_trace(go.Scatter(
            x=[MONTH_ABBR[m - 1] for m in sub["m"]], y=sub["v"] * f, name=str(y),
            mode="lines+markers" if is_cur else "lines",
            line=dict(width=3 if is_cur else 1.5, color=COL_ETHANOL if is_cur else palette[i % len(palette)]),
            marker=dict(size=5) if is_cur else None,
            hovertemplate=f"{y}: %{{y:,.{dec}f}}<extra></extra>"))
    style_fig(fig, yaxis_title=f"{unit} / month")
    fig.update_xaxes(categoryorder="array", categoryarray=MONTH_ABBR)
    snap_anchor("snap_exp_seasonal")
    st.plotly_chart(fig, width="stretch")
    snap_toolbar("snap_exp_seasonal", "U.S. Fuel Ethanol Exports - Seasonal")

    # ── who is buying: trailing 12 months, and the year-to-date shifts ───────
    section_header("Destinations — Last 12 Months and Year-to-Date Change")
    left, right = st.columns(2)
    with left:
        d12 = t12_rank.head(12)[::-1]
        fig = go.Figure(go.Bar(
            x=d12.values * f, y=list(d12.index), orientation="h", marker_color=COL_ETHANOL,
            text=[f"{v * f:,.{dec}f}" for v in d12.values], textposition="outside", cliponaxis=False,
            hovertemplate=f"%{{y}}: %{{x:,.{dec}f}}<extra></extra>"))
        style_fig(fig, legend=False)
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=50, t=30, b=10))
        fig.update_xaxes(title=f"{unit}, {t12_idx[0].strftime('%b %Y')}–{latest.strftime('%b %Y')}")
        snap_anchor("snap_exp_t12")
        st.plotly_chart(fig, width="stretch")
        st.caption("Top 12 destinations, last 12 months")
        snap_toolbar("snap_exp_t12", "U.S. Fuel Ethanol Exports - Top Destinations, Last 12 Months")
    with right:
        chg = (wide.reindex(ytd_idx).fillna(0).sum()
               .sub(wide.reindex(ytd_ly_idx).fillna(0).sum(), fill_value=0)) * f
        chg = chg[chg.abs() > 0]
        pick = pd.concat([chg.sort_values(ascending=False).head(6), chg.sort_values().head(6)])
        pick = pick[~pick.index.duplicated()].sort_values()
        fig = go.Figure(go.Bar(
            x=pick.values, y=list(pick.index), orientation="h",
            marker_color=[POS if v >= 0 else NEG for v in pick.values],
            text=[f"{v:+,.{dec}f}" for v in pick.values], textposition="outside", cliponaxis=False,
            hovertemplate=f"%{{y}}: %{{x:+,.{dec}f}}<extra></extra>"))
        style_fig(fig, legend=False)
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=50, t=30, b=10))
        fig.update_xaxes(title=f"{unit}, Jan–{latest.strftime('%b')} {latest.year} vs {latest.year - 1}",
                         zeroline=True, zerolinecolor=DM_BORDER)
        snap_anchor("snap_exp_ytdchg")
        st.plotly_chart(fig, width="stretch")
        st.caption("Biggest gainers and decliners, year to date vs the same months last year")
        snap_toolbar("snap_exp_ytdchg", "U.S. Fuel Ethanol Exports - YTD Change by Destination")

    # ── comparison table ─────────────────────────────────────────────────────
    section_header("Monthly Exports by Destination — Comparison")
    st.markdown(build_export_compare_table(total, wide, list(t12_rank.index[:12]), "snap_exp_table", f, dec,
                                           unit_note=f"{unit} per month"), unsafe_allow_html=True)
    snap_toolbar("snap_exp_table", "U.S. Fuel Ethanol Exports by Destination - Comparison")

    # ── weekly total ─────────────────────────────────────────────────────────
    section_header("Weekly Total (Early Estimate)")
    if len(wk_df) > 60:
        last_wk = wk_df["period"].max()
        cur_wk = wk_df[wk_df["period"] > last_wk - pd.Timedelta(weeks=52)]
        prev_wk = wk_df[(wk_df["period"] > last_wk - pd.Timedelta(weeks=104))
                        & (wk_df["period"] <= last_wk - pd.Timedelta(weeks=52))].copy()
        prev_wk["period"] = prev_wk["period"] + pd.Timedelta(weeks=52)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prev_wk["period"], y=prev_wk["value"], name="Year ago", mode="lines",
                                 line=dict(color="#84939f", width=1.5, dash="dot"),
                                 hovertemplate="Year ago: %{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=cur_wk["period"], y=cur_wk["value"], name="Last 52 weeks", mode="lines",
                                 line=dict(color=COL_BIODSL, width=2),
                                 hovertemplate="%{x|%b %d, %Y}: %{y:,.0f}<extra></extra>"))
        style_fig(fig, yaxis_title="Thousand bbl/day")
        snap_anchor("snap_exp_weekly")
        st.plotly_chart(fig, width="stretch")
        gaps = weekly_vs_monthly_gaps(total, wk_df)
        if gaps:
            st.markdown(
                f"<div class='note-text'>Weekly figures are preliminary. Over the last {len(gaps)} months with both, "
                f"the weekly average ran {min(gaps):+.0f}% to {max(gaps):+.0f}% against EIA's monthly figure "
                f"(average {sum(gaps) / len(gaps):+.0f}%), so use the monthly series above for the level and the "
                f"weekly line for direction.</div>", unsafe_allow_html=True)
        snap_toolbar("snap_exp_weekly", "U.S. Fuel Ethanol Exports - Weekly")
    else:
        st.info("No weekly ethanol export data returned.")

    st.markdown(
        "<div class='note-text' style='margin-top:10px'>EIA's Fuel Ethanol exports are built from U.S. Census "
        "Bureau export statistics (ethyl alcohol, HS 2207, non-beverage). Destination is the country the "
        "shipment was reported to be going to; the Netherlands (Rotterdam) is a transshipment hub for the rest "
        "of Europe. Destinations add up to EIA's published monthly total; any small gap is shown as "
        "All other.</div>", unsafe_allow_html=True)
