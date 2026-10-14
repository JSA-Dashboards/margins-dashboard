"""Ethanol — JSA
Three views, one at a time (selector under the title):
  Ethanol Margins  — three tabs:
    1. Report Style  — three-location gross margin matching the JSA Ethanol Margins report
    2. Sensitivity Matrix — AMS cash-price matrix (ethanol × corn grid)
    3. CU Curve & RINs — Chicago ethanol forward curve, corn/gas strips, RINs
  Ethanol Exports  — U.S. fuel ethanol exports by destination (EIA; see ethanol_exports.py)
  Weekly EIA       — the JSA-style Weekly EIA Report (see weekly_eia.py)
The two EIA views need EIA_API_KEY in the app's secrets; the margin tabs never did.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import calc_ethanol
import ethanol_grind
from common import (
    GOLD, GREEN, JPSI_BLUE, ORANGE,
    add_time_range_controls, col_style, ddg_ratio_anchor, friendly_ticker, get_api_key,
    load_ams_plant_corn, load_ams_weekly, load_curve, load_hh, load_hist,
    load_zc_dec_archive, oil_ratio_anchor, plotly_base, plotly_cfg,
)
from ethanol_exports import render_ethanol_exports
from weekly_eia import render_weekly_eia

today = date.today()

LOCATIONS = calc_ethanol.LOCATIONS
GAL_PER_BU = calc_ethanol.GAL_PER_BU
DDG_LB_PER_BU = calc_ethanol.DDG_LB_PER_BU
OIL_LB_PER_BU = calc_ethanol.OIL_LB_PER_BU
GAS_MMBTU_PER_GAL = calc_ethanol.GAS_MMBTU_PER_GAL


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ─  REPORT-STYLE THREE-LOCATION MARGIN
# ─────────────────────────────────────────────────────────────────────────────
def render_report_tab(api_key: str, weekly: pd.DataFrame):
    st.markdown("##### Three-location gross margin — December corn futures + basis methodology")

    # ── Live futures ──────────────────────────────────────────────────────────
    zc_curve = load_curve("ZC", api_key, today.isoformat(), 8)
    cu_curve = load_curve("CU", api_key, today.isoformat(), 4)
    ng_curve = load_curve("NG", api_key, today.isoformat(), 8)
    sbo_curve = load_curve("ZL", api_key, today.isoformat(), 3)

    zc_dec = zc_curve[zc_curve["ticker"].str.contains(r"ZCZ\d", regex=True)] if not zc_curve.empty else pd.DataFrame()
    corn_zc_raw = float(zc_dec.iloc[0]["price"]) / 100.0 if not zc_dec.empty else None

    cu_front = float(cu_curve.iloc[0]["price"]) if not cu_curve.empty else None

    ng_oct = ng_curve[ng_curve["ticker"].str.contains("V")] if not ng_curve.empty else pd.DataFrame()
    if ng_oct.empty and not ng_curve.empty:
        ng_oct = ng_curve
    ng_price_raw = float(ng_oct.iloc[0]["price"]) if not ng_oct.empty else None

    sbo_front = float(sbo_curve.iloc[0]["price"]) if not sbo_curve.empty else None
    oil_anchor = oil_ratio_anchor(weekly, sbo_curve, api_key, today.isoformat())
    ddg_anchor = ddg_ratio_anchor(weekly, zc_curve, api_key, today.isoformat())

    # ── Assumptions & derived inputs ──────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**Assumptions & derived inputs**")
        a1, a2, a3, a4, a5 = st.columns(5)
        with a1:
            gal = st.number_input("Ethanol gal/bu", 0.0, 4.0, GAL_PER_BU, 0.01, key="rpt_gal")
        with a2:
            ddg_lb = st.number_input("DDGS lb/bu", 0.0, 20.0, DDG_LB_PER_BU, 0.5, key="rpt_ddglb")
        with a3:
            oil_lb = st.number_input("Corn oil lb/bu", 0.0, 3.0, OIL_LB_PER_BU, 0.05, key="rpt_oillb")
        with a4:
            gas_use = st.number_input("Gas MMBtu/gal", 0.0, 0.20, GAS_MMBTU_PER_GAL,
                                      0.001, format="%.3f", key="rpt_gasuse")
        with a5:
            gas_basis = st.number_input("Gas basis $/MMBtu", -2.0, 2.0, -0.35, 0.01,
                                        key="rpt_gasbasis",
                                        help="'35-V' in the report = −$0.35 under October NG")

        basis_df = pd.DataFrame({
            "Location": list(LOCATIONS.keys()),
            "Region": [cfg["region"] for cfg in LOCATIONS.values()],
            "Corn basis ¢": [round(cfg["corn_basis"] * 100) for cfg in LOCATIONS.values()],
            "Eth adj ¢": [round(cfg["eth_adj"] * 100) for cfg in LOCATIONS.values()],
        })
        edited = st.data_editor(
            basis_df, hide_index=True, width="stretch", key="rpt_basis_editor",
            disabled=["Location", "Region"],
            column_config={
                "Corn basis ¢": st.column_config.NumberColumn(format="%+d¢", step=1,
                                                               help="vs December ZC"),
                "Eth adj ¢": st.column_config.NumberColumn(format="%+d¢", step=1,
                                                           help="vs Chicago (CU)"),
            },
        )
        b_col = dict(zip(edited["Location"], edited["Corn basis ¢"] / 100))
        eth_adj = dict(zip(edited["Location"], edited["Eth adj ¢"] / 100))

        oil_spot_cents = (sbo_front * oil_anchor["ratio"]) if (sbo_front is not None and oil_anchor) else None
        ddgs_pct = ddg_anchor["ratio"] if ddg_anchor else None
        if ng_price_raw is None:
            hh = load_hh()
            ng_price_raw = float(hh.iloc[-1]) if len(hh) else None
        gas_price = (ng_price_raw + gas_basis) if ng_price_raw else None

        if oil_anchor and ddg_anchor:
            gas_note = f" · gas = Oct NG ${ng_price_raw:.3f}/MMBtu {gas_basis:+.2f}" if ng_price_raw else ""
            st.caption(
                f"Derived — corn oil = ZL ${sbo_front:.2f}¢ × {oil_anchor['ratio']*100:.1f}% "
                f"(AMS {oil_anchor['ams_date']:%b %d}: {oil_anchor['ams_price']:.1f}¢ ÷ "
                f"ZL {oil_anchor['ref_price']:.1f}¢) · DDGS = Dec ZC × {ddgs_pct*100:.1f}% "
                f"(AMS {ddg_anchor['ams_date']:%b %d}: ${ddg_anchor['ams_price']:.0f}/ton ÷ "
                f"ZC ${ddg_anchor['ref_price']:.0f}/ton){gas_note}"
            )
        else:
            st.caption("Derived corn-oil/DDGS ratios unavailable — need an AMS report plus ZL/ZC futures.")

    unit_choice = st.segmented_control("Margin unit", ["¢/gal", "$/gal", "$/bu"],
                                       default="¢/gal", key="rpt_unit") or "¢/gal"

    # ── Output ────────────────────────────────────────────────────────────────
    has_futures = corn_zc_raw and cu_front and gas_price and oil_spot_cents and ddgs_pct

    if not has_futures:
        if not api_key:
            st.info("Futures unavailable — add `MASSIVE_API_KEY` to secrets or .env.")
        else:
            missing = [x for x, v in [("ZC Dec", corn_zc_raw), ("CU front", cu_front),
                                       ("gas price", gas_price), ("corn oil ratio", oil_spot_cents),
                                       ("DDG ratio", ddgs_pct)] if not v]
            st.warning(f"Some futures unavailable: {', '.join(missing)}. Showing available data only.")
    else:
        ddgs_label = f"DDGS rev ({unit_choice})"
        margin_label = f"Gross margin ({unit_choice})"
        rows = {}
        for loc, cfg in LOCATIONS.items():
            corn_loc = corn_zc_raw + b_col[loc]
            eth_loc = cu_front + eth_adj[loc]
            m = calc_ethanol.report_margin(eth_loc, corn_loc, ddgs_pct, gas_price, gal, ddg_lb,
                                           oil_lb, oil_spot_cents, gas_use, unit_choice)
            rows[loc] = {
                "Ethanol $/gal": eth_loc,
                "Corn $/bu (ZC+basis)": corn_loc,
                "Gas $/MMBtu": gas_price,
                "Corn oil ¢/lb": oil_spot_cents,
                ddgs_label: m["ddg_rev"],
                margin_label: m["margin"],
            }
        out_df = pd.DataFrame(rows)

        margin_prec = 1 if unit_choice == "¢/gal" else 2
        row_fmt = {
            "Ethanol $/gal": "${:.3f}",
            "Corn $/bu (ZC+basis)": "${:.3f}",
            "Gas $/MMBtu": "${:.3f}",
            "Corn oil ¢/lb": "{:.2f}¢",
            ddgs_label: f"{{:+.{margin_prec}f}}",
            margin_label: f"{{:+.{margin_prec}f}}",
        }
        def _margin_bg(val):
            if pd.isna(val):
                return ""
            return "background-color:#e8f8ef;color:#1e7e42;" if val >= 0 else "background-color:#fdecea;color:#b3261e;"

        styled = out_df.style
        for row_label, fmt_str in row_fmt.items():
            styled = styled.format(fmt_str, subset=pd.IndexSlice[row_label, :])
        styled = (styled
            .apply(lambda row: [_margin_bg(v) for v in row] if row.name == margin_label
                   else [""] * len(row), axis=1)
            .set_properties(**{"font-weight": "700"}, subset=pd.IndexSlice[margin_label, :])
            .set_properties(**{"font-size": "0.85rem", "text-align": "center", "padding": "5px 12px",
                               "border": "1px solid #dee2e6"})
        )
        html = styled.to_html()
        st.markdown(f'<div style="overflow-x:auto;border-radius:6px;border:1px solid #dee2e6;">{html}</div>',
                    unsafe_allow_html=True)
        st.caption(
            " · ".join(f"{loc} — {cfg['region']}" for loc, cfg in LOCATIONS.items())
            + f"  ·  as of {today:%b %d, %Y}"
        )

    # ── History chart (AMS-based proxy) ──────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Historical margin — AMS weekly ethanol price + ZC Dec archive + location basis")

    zc_archive = load_zc_dec_archive()
    hist = calc_ethanol.historical_margins(
        weekly, zc_archive, b_col=b_col, eth_adj=eth_adj,
        ddg_lb=ddg_lb, oil_lb=oil_lb, gas_basis=gas_basis, gas_use=gas_use,
        gal=gal, unit=unit_choice,
    )
    if hist:
        fig = plotly_base(380)
        loc_colors = [JPSI_BLUE, GREEN, ORANGE]
        for (loc, margin_s), clr in zip(hist.items(), loc_colors):
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(d) for d in margin_s.index],
                y=margin_s.values,
                name=loc, mode="lines",
                line=dict(color=clr, width=2),
                hovertemplate=f"{loc}: %{{y:+.2f}}{unit_choice}<extra></extra>",
            ))
        fig.add_hline(y=0, line_color="#dee2e6", line_width=1)
        fig.update_layout(yaxis_title=f"Gross margin ({unit_choice})")
        add_time_range_controls(fig)
        st.plotly_chart(fig, width="stretch", config=plotly_cfg())
        st.caption(
            "Historical ethanol: AMS weekly cash (Illinois/Iowa). "
            "Historical corn: ZC December archive (cents/bu). "
            "Historical DDG and corn oil: real AMS weekly cash prints (no projection needed — "
            "every point already lands on an AMS report date). "
            "Historical gas: $3.00/MMBtu proxy (live data only available for current session)."
        )
    else:
        st.info("Historical chart requires the AMS snapshot and the ZC December archive. "
                "Run `python ethanol_grind.py` to extend the AMS snapshot.")

    return {
        "b_col": b_col, "eth_adj": eth_adj, "ddgs_pct": ddgs_pct, "gas_basis": gas_basis,
        "gal": gal, "ddg_lb": ddg_lb, "oil_lb": oil_lb, "gas_use": gas_use,
        "oil_anchor": oil_anchor, "ddg_anchor": ddg_anchor,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ─  SENSITIVITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def _latest(frame: pd.DataFrame, commodity: str, state: str,
            variety: str | None = None) -> float | None:
    s = ethanol_grind.series(frame, commodity, state, variety)
    return float(s.iloc[-1]) if len(s) else None


def render_matrix_tab(weekly: pd.DataFrame, plant_corn: pd.DataFrame):
    st.markdown("##### Sensitivity matrix — AMS cash prices (ethanol × corn grid)")
    st.caption(
        "Co-product revenue from AMS Market News weekly report. "
        "Matrix shows margin across a range of ethanol and corn prices."
    )

    state_list = ["All states"] + ethanol_grind.states(weekly)
    hh = load_hh()
    hh_latest = float(hh.iloc[-1]) if len(hh) else 3.00

    with st.container(border=True):
        st.markdown("**Assumptions**")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            state = st.selectbox("State", state_list, index=min(1, len(state_list)-1), key="mx_state")
        with r1c2:
            ddg_variety = st.selectbox("Distillers grain", ethanol_grind.DDG_VARIETIES, key="mx_ddgvar")
        with r1c3:
            unit = st.segmented_control("Margin unit", ["$/bu", "$/gal"],
                                        default="$/bu", key="mx_unit") or "$/bu"

        r2c1, r2c2 = st.columns([1, 2])
        with r2c1:
            gas_src = st.segmented_control("Gas price", ["Henry Hub", "Fixed"],
                                           default="Henry Hub", key="mx_gassrc") or "Henry Hub"
        with r2c2:
            if gas_src == "Fixed" or not len(hh):
                gas_price = st.number_input("Gas $/MMBtu", 0.0, 25.0,
                                            round(hh_latest, 2), 0.05, key="mx_gasfix")
            else:
                gas_price = hh_latest
                st.caption(f"Henry Hub: ${hh_latest:.2f}/MMBtu ({hh.index[-1]:%b %d})")

        r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
        with r3c1:
            gal = st.number_input("Ethanol gal/bu", 0.0, 4.0, GAL_PER_BU, 0.01, key="mx_gal")
        with r3c2:
            ddg_lb = st.number_input("DDGS lb/bu", 0.0, 20.0, DDG_LB_PER_BU, 0.5, key="mx_ddglb")
        with r3c3:
            oil_lb = st.number_input("Corn oil lb/bu", 0.0, 3.0, OIL_LB_PER_BU, 0.05, key="mx_oillb")
        with r3c4:
            gas_use = st.number_input("Gas MMBtu/gal", 0.0, 0.2, GAS_MMBTU_PER_GAL,
                                      0.001, format="%.3f", key="mx_gasuse")
        with r3c5:
            opex = st.number_input("Other costs $/bu", 0.0, 3.0, 0.0, 0.05, key="mx_opex",
                                   help="Power, enzymes, labour — excluding corn and gas")

    eth_spot = _latest(weekly, "Ethanol", state)
    ddg_spot = _latest(weekly, "Distillers Grain", state, ddg_variety)
    oil_spot = _latest(weekly, "Distillers Corn Oil", state)
    corn_spot = _latest(plant_corn, "Corn", state)

    ddg_rev = (ddg_spot * ddg_lb / 2000) if ddg_spot else 0.0
    oil_rev = (oil_spot / 100 * oil_lb) if oil_spot else 0.0
    gas_cost = gas_price * gas_use * gal
    latest_week = weekly["date"].max() if len(weekly) else today

    spot_margin = None
    if eth_spot and corn_spot:
        spot_margin = eth_spot * gal + ddg_rev + oil_rev - corn_spot - gas_cost - opex

    spot_rows = {
        "Ethanol spot": f"${eth_spot:.3f}/gal  ·  ${eth_spot*gal:.2f}/bu" if eth_spot else "—",
        "Distillers grain": f"${ddg_spot:.0f}/ton  ·  ${ddg_rev:.2f}/bu rev" if ddg_spot else "—",
        "Corn oil": f"{oil_spot:.1f}¢/lb  ·  ${oil_rev:.2f}/bu rev" if oil_spot else "—",
        "Plant corn bid": f"${corn_spot:.2f}/bu" if corn_spot else "—",
        "Natural gas": f"${gas_price:.2f}/MMBtu  ·  −${gas_cost:.2f}/bu cost",
        "Spot margin": (f"${spot_margin:+.2f}/bu  ·  ${spot_margin/gal:+.3f}/gal"
                        if spot_margin is not None else "—"),
    }
    spot_df = pd.DataFrame({"Value": spot_rows})

    def _spot_margin_bg(row):
        if row.name != "Spot margin" or spot_margin is None:
            return [""] * len(row)
        clr = "#e8f8ef;color:#1e7e42" if spot_margin >= 0 else "#fdecea;color:#b3261e"
        return [f"background-color:{clr};font-weight:700;"] * len(row)

    styled_spot = (spot_df.style
        .apply(_spot_margin_bg, axis=1)
        .set_properties(**{"font-size": "0.85rem", "padding": "6px 14px", "border": "1px solid #dee2e6"})
        .set_properties(subset=pd.IndexSlice[:, "Value"], **{"text-align": "right"}))
    st.markdown(
        f'<div style="overflow-x:auto;border-radius:6px;border:1px solid #dee2e6;max-width:440px;">'
        f'{styled_spot.to_html()}</div>', unsafe_allow_html=True)
    st.caption(f"Week of {latest_week:%b %d, %Y} · {state} · {ddg_variety} DDG")

    st.markdown("---")
    st.markdown("**Price range**")
    mc1, mc2 = st.columns(2)
    with mc1:
        eth_lo = st.number_input("Ethanol low $/gal", 0.50, 3.50,
                                 round(max(1.20, (eth_spot or 1.80) - 0.50), 2), 0.05, key="mx_ethlo")
        eth_hi = st.number_input("Ethanol high $/gal", 0.50, 4.00,
                                 round(min(3.50, (eth_spot or 1.80) + 0.50), 2), 0.05, key="mx_ethi")
    with mc2:
        corn_lo = st.number_input("Corn low $/bu", 1.00, 8.00,
                                  round(max(2.50, (corn_spot or 4.50) - 1.50), 2), 0.25, key="mx_cornlo")
        corn_hi = st.number_input("Corn high $/bu", 1.00, 12.00,
                                  round(min(9.00, (corn_spot or 4.50) + 1.50), 2), 0.25, key="mx_cornhi")

    n_eth = max(3, round((eth_hi - eth_lo) / 0.05) + 1)
    n_corn = max(3, round((corn_hi - corn_lo) / 0.25) + 1)
    eth_prices = np.linspace(eth_lo, eth_hi, n_eth)
    corn_prices = np.linspace(corn_lo, corn_hi, n_corn)

    matrix = calc_ethanol.build_matrix(eth_prices[::-1], corn_prices, ddg_rev, oil_rev,
                                       gas_cost, opex, gal, unit)

    fmt = "{:+.2f}" if unit == "$/bu" else "{:+.3f}"
    styled = (matrix.style.map(col_style).format(fmt)
              .set_properties(**{"font-size":"0.78rem","text-align":"center","padding":"3px 8px",
                                 "border":"1px solid #dee2e6"}))

    if eth_spot:
        nr = f"{eth_prices[::-1][np.argmin(np.abs(eth_prices[::-1] - eth_spot))]:.2f}"
        styled = styled.set_properties(**{"border":"2.5px solid #0693e3","font-weight":"700"},
                                       subset=pd.IndexSlice[nr, :])
    if corn_spot:
        nc = f"{corn_prices[np.argmin(np.abs(corn_prices - corn_spot))]:.2f}"
        styled = styled.set_properties(**{"border":"2.5px solid #e8833a","font-weight":"700"},
                                       subset=pd.IndexSlice[:, nc])

    st.markdown(f"**Margin {unit} · {ddg_variety} DDG · gas ${gas_price:.2f}/MMBtu · {gal:.2f} gal/bu**")
    html = styled.to_html()
    st.markdown(f'<div style="overflow-x:auto;border-radius:6px;border:1px solid #dee2e6;">{html}</div>',
                unsafe_allow_html=True)
    st.caption(
        "Blue border = nearest ethanol spot row. Orange border = nearest plant corn bid column. "
        f"Co-products: DDG ${ddg_rev:.2f}/bu + oil ${oil_rev:.2f}/bu = ${ddg_rev+oil_rev:.2f}/bu. "
        "Sources: USDA AMS Market News, EIA."
    )

    with st.expander("Breakeven — ethanol price needed at each corn price"):
        co = ddg_rev + oil_rev
        be = (corn_prices + gas_cost + opex - co) / gal if gal else corn_prices * 0
        fig = plotly_base(320)
        fig.add_trace(go.Scatter(x=corn_prices, y=be, mode="lines",
                                 line=dict(color=JPSI_BLUE, width=2.5), name="Breakeven ethanol",
                                 hovertemplate="Corn $%{x:.2f} → break-even $%{y:.3f}/gal<extra></extra>"))
        if eth_spot:
            fig.add_hline(y=eth_spot, line=dict(color=JPSI_BLUE, width=1.5, dash="dot"),
                          annotation_text=f"Spot ${eth_spot:.3f}/gal")
        if corn_spot:
            fig.add_vline(x=corn_spot, line=dict(color=ORANGE, width=1.5, dash="dot"),
                          annotation_text=f"Plant bid ${corn_spot:.2f}/bu")
        fig.update_layout(xaxis_title="Corn $/bu", yaxis_title="Break-even ethanol $/gal")
        st.plotly_chart(fig, width='stretch', config=plotly_cfg())


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ─  CU CURVE & RINS
# ─────────────────────────────────────────────────────────────────────────────
def _spread_label(i: int, tickers: list[str], code: str) -> str:
    if i + 1 >= len(tickers):
        return ""
    m1 = tickers[i][len(code)][0]
    m2 = tickers[i+1][len(code)][0]
    return f"{m1}/{m2}"


def render_curve_tab(api_key: str, weekly: pd.DataFrame, report_ctx: dict | None):
    st.markdown("##### Chicago Ethanol forward curve, corn & nat gas strips, and RINs")

    cu = load_curve("CU", api_key, today.isoformat(), 10)
    zc = load_curve("ZC", api_key, today.isoformat(), 6)
    ng = load_curve("NG", api_key, today.isoformat(), 6)
    zl = load_curve("ZL", api_key, today.isoformat(), 10)

    no_api = cu.empty and zc.empty and ng.empty
    if no_api:
        st.info("Add `MASSIVE_API_KEY` to your Streamlit secrets or .env to load live futures data.")

    left, right = st.columns([1.3, 1.0])

    with left:
        st.markdown("###### Chicago Ethanol Curve w/ Spreads")
        if not cu.empty:
            hist = load_hist(tuple(cu["ticker"]), api_key, today.isoformat())
            rows = []
            prev_price = None
            for i, r in enumerate(cu.itertuples(index=False)):
                series = hist.get(r.ticker)
                chg = float(r.price - series.iloc[-2]) if series is not None and len(series) >= 2 else None
                spd_label = _spread_label(i, list(cu["ticker"]), "CU")
                if prev_price is not None and spd_label:
                    spd_val = round(float(r.price) - prev_price, 4)
                else:
                    spd_val = None
                rows.append({
                    "Contract": pd.Timestamp(r.expiration).strftime("%b-%y"),
                    "Settle": round(float(r.price), 4),
                    "+/-": round(chg, 4) if chg is not None else None,
                    "Vol": None,
                    "OI": None,
                    "CU Spread": spd_label,
                    "Spread": round(spd_val, 4) if spd_val is not None else None,
                })
                prev_price = float(r.price)

            df_cu = pd.DataFrame(rows)

            def _cu_style(col):
                if col.name in ("+/-", "Spread"):
                    return [f"color:{'#27ae60' if (v or 0)>=0 else '#c0392b'};" if v is not None else "" for v in col]
                return [""] * len(col)

            styled_cu = (df_cu.style
                         .apply(_cu_style)
                         .format({"Settle": "{:.4f}", "+/-": "{:+.4f}", "Spread": "{:+.4f}"},
                                 na_rep="—")
                         .set_properties(**{"font-size": "0.80rem", "text-align": "center"}))
            st.dataframe(styled_cu, hide_index=True, width='stretch',
                         height=min(36*(len(df_cu)+1)+3, 420))
        else:
            st.caption("Live CU data unavailable.")

    with right:
        if not cu.empty:
            st.markdown("###### CU Forward Curve")
            fig = plotly_base(200)
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(d) for d in cu["expiration"]],
                y=cu["price"].tolist(), mode="lines+markers",
                line=dict(color=JPSI_BLUE, width=2),
                marker=dict(size=6),
                hovertemplate="%{x|%b %y}: $%{y:.4f}/gal<extra></extra>",
            ))
            fig.update_layout(
                margin=dict(l=40, r=10, t=20, b=40),
                yaxis_title="$/gal", height=200,
            )
            st.plotly_chart(fig, width='stretch', config=plotly_cfg())

    st.divider()

    cc1, cc2, cc3 = st.columns([1.1, 1.1, 0.8])

    with cc1:
        st.markdown("###### Corn Futures")
        if not zc.empty:
            hist_zc = load_hist(tuple(zc["ticker"]), api_key, today.isoformat())
            rows_zc = []
            for r in zc.itertuples(index=False):
                s = hist_zc.get(r.ticker)
                chg = float(r.price - s.iloc[-2]) if s is not None and len(s) >= 2 else None
                rows_zc.append({
                    "Contract": friendly_ticker(r.ticker, "ZC"),
                    "Last": round(r.price, 2),
                    "+/-": round(chg, 2) if chg is not None else None,
                })
            df_zc = pd.DataFrame(rows_zc)

            def _zc_style(col):
                if col.name == "+/-":
                    return [f"color:{'#27ae60' if (v or 0)>=0 else '#c0392b'};" if v is not None else "" for v in col]
                return [""] * len(col)

            st.dataframe(
                df_zc.style.apply(_zc_style)
                     .format({"Last": "{:.2f}", "+/-": "{:+.2f}"}, na_rep="—")
                     .set_properties(**{"font-size": "0.80rem", "text-align": "center"}),
                hide_index=True, width='stretch',
                height=min(36*(len(df_zc)+1)+3, 280),
            )
        else:
            st.caption("Live ZC data unavailable.")

    with cc2:
        st.markdown("###### Nat Gas Futures")
        if not ng.empty:
            hist_ng = load_hist(tuple(ng["ticker"]), api_key, today.isoformat())
            rows_ng = []
            for r in ng.itertuples(index=False):
                s = hist_ng.get(r.ticker)
                chg = float(r.price - s.iloc[-2]) if s is not None and len(s) >= 2 else None
                rows_ng.append({
                    "Contract": friendly_ticker(r.ticker, "NG"),
                    "Last": round(r.price, 3),
                    "+/-": round(chg, 3) if chg is not None else None,
                })
            df_ng = pd.DataFrame(rows_ng)

            def _ng_style(col):
                if col.name == "+/-":
                    return [f"color:{'#27ae60' if (v or 0)>=0 else '#c0392b'};" if v is not None else "" for v in col]
                return [""] * len(col)

            st.dataframe(
                df_ng.style.apply(_ng_style)
                     .format({"Last": "{:.3f}", "+/-": "{:+.3f}"}, na_rep="—")
                     .set_properties(**{"font-size": "0.80rem", "text-align": "center"}),
                hide_index=True, width='stretch',
                height=min(36*(len(df_ng)+1)+3, 280),
            )
        else:
            st.caption("Live NG data unavailable.")

    with cc3:
        st.markdown("###### RINs")
        st.caption("OPIS D6 Ethanol & D4 Biodiesel")
        rin_year = today.year
        d6 = st.number_input(f"D6 Ethanol {rin_year}", 0.0, 5.0, 2.12, 0.005,
                             format="%.4f", key="rin_d6")
        d6_chg = st.number_input("D6 +/-", -1.0, 1.0, -0.005, 0.001,
                                  format="%.4f", key="rin_d6chg")
        d4 = st.number_input(f"D4 Biodiesel {rin_year}", 0.0, 5.0, 2.175, 0.005,
                             format="%.4f", key="rin_d4")
        d4_chg = st.number_input("D4 +/-", -1.0, 1.0, -0.005, 0.001,
                                  format="%.4f", key="rin_d4chg")
        st.caption("Enter from OPIS RIN report")

    st.divider()

    st.markdown("###### Forward margin curve — projected gross margin by ethanol contract month")
    oil_anchor = report_ctx.get("oil_anchor") if report_ctx else None
    ddg_pct = report_ctx.get("ddgs_pct") if report_ctx else None
    if report_ctx and oil_anchor and ddg_pct and not cu.empty:
        ratio = oil_anchor["ratio"]
        curves = calc_ethanol.forward_margin_curve(
            cu, zc, ng, zl,
            b_col=report_ctx["b_col"], eth_adj=report_ctx["eth_adj"],
            ddgs_pct=report_ctx["ddgs_pct"], ddg_lb=report_ctx["ddg_lb"],
            oil_lb=report_ctx["oil_lb"], gas_use=report_ctx["gas_use"],
            gas_basis=report_ctx["gas_basis"], oil_ratio=ratio, gal=report_ctx["gal"],
            unit="¢/gal",
        )
        if curves and any(len(df) for df in curves.values()):
            fig = plotly_base(340)
            loc_colors = [JPSI_BLUE, GREEN, ORANGE]
            for (loc, df), clr in zip(curves.items(), loc_colors):
                if df.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=[pd.Timestamp(d) for d in df["expiration"]],
                    y=df["margin"].values,
                    name=loc, mode="lines+markers",
                    line=dict(color=clr, width=2), marker=dict(size=5),
                    hovertemplate=f"{loc}: %{{y:+.2f}}¢/gal<extra></extra>",
                ))
            fig.add_hline(y=0, line_color="#dee2e6", line_width=1)
            fig.update_layout(yaxis_title="Projected gross margin (¢/gal)")
            st.plotly_chart(fig, width='stretch', config=plotly_cfg())
            st.caption(
                "Ethanol follows its own CU forward curve; corn (Dec ZC) and gas (Oct NG) stay "
                "anchored to the Report Style tab's reference months, same as the current-margin "
                "snapshot. Corn oil follows the nearest ZL contract to each CU month, scaled by "
                f"the AMS-anchored ratio ({ratio*100:.1f}%). DDGS follows Dec ZC (fixed for the "
                f"whole curve, like corn) × the AMS-anchored ratio ({ddg_pct*100:.1f}%). Both "
                "ratios come from the Report Style tab."
            )
        else:
            st.info("Forward margin curve needs CU, ZC, NG, and ZL futures data.")
    else:
        st.info("Forward margin curve needs the corn-oil/SBO and DDG/corn ratios from the "
                "Report Style tab (requires an AMS report plus ZL and ZC futures) and CU futures.")

    st.divider()

    st.markdown("###### Ethanol vs Corn futures — historical")
    zc_archive = load_zc_dec_archive()
    eth_il = ethanol_grind.series(weekly, "Ethanol", "Illinois") if len(weekly) else pd.Series(dtype=float)

    if len(zc_archive) or len(eth_il):
        fig = plotly_base(380)
        if len(zc_archive):
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(d) for d in zc_archive.index],
                y=(zc_archive * 100).values,
                name="CBOT Corn (¢/bu)", mode="lines",
                line=dict(color=GOLD, width=1.5),
                yaxis="y2",
                hovertemplate="Corn: %{y:.1f}¢/bu<extra></extra>",
            ))
        if len(eth_il):
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(d) for d in eth_il.index],
                y=eth_il.values,
                name="Chicago Ethanol $/gal (AMS)", mode="lines",
                line=dict(color=JPSI_BLUE, width=2),
                hovertemplate="Ethanol: $%{y:.3f}/gal<extra></extra>",
            ))
        fig.update_layout(
            yaxis=dict(title="Ethanol $/gal", gridcolor="#e9ecef"),
            yaxis2=dict(title="Corn ¢/bu", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=-0.22),
        )
        st.plotly_chart(fig, width='stretch', config=plotly_cfg())
        st.caption("Corn: ZC December archive. Ethanol: AMS Illinois weekly cash.")
    else:
        st.info("No historical data available.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_margins(api_key: str) -> None:
    """The ethanol margin tabs: Report Style / Sensitivity Matrix / CU Curve & RINs."""
    weekly = load_ams_weekly(today.isoformat())
    plant_corn = load_ams_plant_corn(today.isoformat())

    if not len(weekly):
        st.error("No AMS ethanol data. Run `python ethanol_grind.py` in the ethanol-margin-matrix folder.")
        return

    tab1, tab2, tab3 = st.tabs(["Report Style", "Sensitivity Matrix", "CU Curve & RINs"])

    with tab1:
        report_ctx = render_report_tab(api_key, weekly)

    with tab2:
        render_matrix_tab(weekly, plant_corn)

    with tab3:
        render_curve_tab(api_key, weekly, report_ctx)


st.title("🌽 Ethanol")

api_key = get_api_key()
with st.sidebar:
    if api_key:
        st.success("Futures API: connected")
    else:
        st.error("MASSIVE_API_KEY not found — add to .env or Streamlit secrets")

# One view at a time. Only the selected view runs, so a click in one never re-runs the others
# (st.tabs runs every tab body on every interaction, which would pull the EIA data and the futures
# curves together). The view is kept in the URL (?view=exports) so a link opens straight to it.
VIEWS = {"margins": "Ethanol Margins", "exports": "Ethanol Exports", "weekly-eia": "Weekly EIA"}
if "eth_view" not in st.session_state:
    st.session_state["eth_view"] = VIEWS.get(st.query_params.get("view", ""), VIEWS["margins"])
view = st.segmented_control("View", list(VIEWS.values()), required=True, key="eth_view",
                            label_visibility="collapsed")
st.query_params["view"] = next(slug for slug, label in VIEWS.items() if label == view)

if view == "Ethanol Exports":
    render_ethanol_exports()
elif view == "Weekly EIA":
    render_weekly_eia()
else:
    render_margins(api_key)
