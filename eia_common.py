"""
Shared pieces for the EIA-based views of the Ethanol page (Weekly EIA report, Ethanol exports).

Copied from the EIA Energy Dashboard (eia-energy-dashboard/app.py) so both pages look and
read the same: the EIA v2 fetcher, the light house theme, chart/section helpers and the
PNG / Copy snapshot toolbar. One deliberate difference: the toolbar below also handles a
table that carries the snapshot id itself (the original only handled a wrapper around one,
so on a bare table it could grab the next chart on the page instead).

The EIA key is read from Streamlit secrets or the environment (EIA_API_KEY), the same key
ethanol_grind.py already uses for Henry Hub. ethanol_grind.py only runs locally, so on Streamlit
Cloud the key has to be added to the app's Secrets: the margin tabs never needed it.
"""
from __future__ import annotations

import base64
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from urllib3.util.retry import Retry

# ── Theme (light, matching the EIA Energy Dashboard and the other JSA dashboards) ─
DM_BG       = "#f7f9fa"
DM_SURFACE  = "#ffffff"
DM_SURFACE2 = "#f0f2f4"
DM_BORDER   = "#dde3e8"
DM_TEXT     = "#1f2328"
DM_MUTED    = "#6b7280"
JSA_BLUE    = "#0693e3"
COL_ETHANOL = "#f2b134"
COL_BIODSL  = "#5ec48c"
COL_REDSL   = "#8db8e0"
POS         = "#5ec48c"
NEG         = "#e0716f"

BASE_URL = "https://api.eia.gov/v2"


@st.cache_data
def _asset_uri(filename: str) -> str:
    p = Path(__file__).parent / "assets" / filename
    try:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    except OSError:
        return ""


def _secret(name: str) -> str:
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    return str(v or os.environ.get(name, "")).strip()


def inject_css() -> None:
    """House styling for the EIA views. Called at the top of each view, so it only exists on the
    page while that view is showing and never touches the margin tabs."""
    wm = _asset_uri("jsa_50yr.png")
    st.markdown(f"""
<style>
  [data-testid="stMetric"] {{
      background: {DM_SURFACE}; border: 1px solid {DM_BORDER};
      border-radius: 8px; padding: 12px 16px;
  }}
  [data-testid="stMetricLabel"] {{ color: {DM_MUTED} !important; font-size: 0.78rem !important; }}
  [data-testid="stMetricValue"] {{ color: {DM_TEXT} !important; font-size: 1.35rem !important; }}
  [data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}
  .section-header {{
      font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
      text-transform: uppercase; color: {DM_MUTED}; margin: 20px 0 6px 0;
  }}
  .note-text {{ color: {DM_MUTED}; font-size: 0.78rem; }}
  /* JSA watermark on charts and snapshot tables, painted on top at low opacity (Plotly draws an
     opaque background of its own, so a ::before behind it would be hidden). */
  [data-testid="stPlotlyChart"] {{ position: relative; }}
  [data-testid="stPlotlyChart"]::after {{
      content: ""; position: absolute; inset: 0;
      background: url('{wm}') center 50% / 28% auto no-repeat;
      opacity: 0.12; pointer-events: none; z-index: 5;
  }}
  table[id^="snap_"] {{ position: relative; }}
  table[id^="snap_"]::after {{
      content: ""; position: absolute; inset: 0;
      background: url('{wm}') center 50% / 24% auto no-repeat;
      opacity: 0.09; pointer-events: none; z-index: 5;
  }}
</style>
""", unsafe_allow_html=True)


# ── EIA v2 fetcher ────────────────────────────────────────────────────────────
@st.cache_resource
def _session():
    s = requests.Session()
    # EIA's gateway answers a burst with 429 + Retry-After: 1 and a slow query can take 30 s or more, so
    # retry rejections and gateway errors but never a read timeout (that restarts a query that is
    # still running and is how a 10-second load turned into a minutes-long one)
    retry = Retry(total=3, connect=3, read=0, status=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=24))
    return s


def _facet_params(facets: dict) -> list:
    out = []
    for key, vals in facets.items():
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            out.append((f"facets[{key}][]", v))
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _eia_fetch(route: str, facets: dict | None, start: str | None, end: str | None,
               max_rows: int) -> pd.DataFrame:
    """The cached fetch. It raises when a call fails, because st.cache_data never stores an exception:
    a slow EIA response must not be remembered for an hour as 'no data' (or as a half-read history).
    The hour matters: a cold load takes EIA's response time several times over, and the data only
    changes weekly or monthly (the views' Refresh buttons pull new data on demand)."""
    params_base = [("api_key", _secret("EIA_API_KEY")), ("data[0]", "value"),
                   ("sort[0][column]", "period"), ("sort[0][direction]", "desc")]
    if facets:
        params_base += _facet_params(facets)
    if start:
        params_base.append(("start", start))
    if end:
        params_base.append(("end", end))

    length = 5000

    def page(offset: int) -> dict:
        r = _session().get(f"{BASE_URL}/{route}/data/", timeout=(10, 150),
                           params=params_base + [("offset", offset), ("length", length)])
        r.raise_for_status()
        return r.json().get("response", {})

    first = page(0)
    rows = list(first.get("data", []))
    total = int(first.get("total", len(rows)) or 0)
    if len(rows) == length:         # a long history: the remaining pages are fetched side by side
        with ThreadPoolExecutor(max_workers=4) as pool:
            for resp in pool.map(page, range(length, min(total, max_rows), length)):
                rows.extend(resp.get("data", []))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df.sort_values("period").reset_index(drop=True)


def eia_get(route: str, facets: dict | None = None, start: str | None = None,
            end: str | None = None, max_rows: int = 30000) -> pd.DataFrame:
    """Every row for a route/facet combo from the EIA v2 API, paginated. Cached an hour (successful
    fetches only). Returns the columns EIA sends (period as datetime, value as float), sorted by period;
    an empty frame, with a warning on the page, when the call fails."""
    if not _secret("EIA_API_KEY"):
        st.error("No EIA_API_KEY found in Streamlit secrets or environment.")
        return pd.DataFrame()
    try:
        return _eia_fetch(route, facets, start, end, max_rows)
    except Exception as e:
        st.warning(f"EIA API error on {route}: {e}")
        return pd.DataFrame()


eia_get.clear = _eia_fetch.clear      # the views' Refresh buttons call eia_get.clear()


def eia_get_many(specs: dict) -> dict:
    """{name: (route, facets, start)} -> {name: DataFrame}, fetched side by side. EIA takes several seconds
    to answer a query whatever its size, so fifteen in a row cost minutes while fifteen at once cost about
    as long as the slowest few. The threads only fill the cache; every result is then read back through
    eia_get on this thread, so warnings and errors still show on the page (and a failed call gets one
    more try)."""
    ctx = get_script_run_ctx()

    def warm(spec):
        route, facets, start = spec
        try:
            _eia_fetch(route, facets, start, None, 30000)
        except Exception:
            pass                      # eia_get below repeats the call and reports it

    if _secret("EIA_API_KEY") and len(specs) > 1:
        with ThreadPoolExecutor(max_workers=6, initializer=add_script_run_ctx, initargs=(None, ctx)) as pool:
            list(pool.map(warm, specs.values()))
    return {name: eia_get(route, facets, start=start) for name, (route, facets, start) in specs.items()}


# ── Formatting / chart helpers ────────────────────────────────────────────────
def fmt_num(v, decimals=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{decimals}f}"


def latest_and_delta(df: pd.DataFrame, n_back: int = 1):
    """Latest value plus the change vs n_back periods prior."""
    if df.empty or len(df) <= n_back:
        return None, None, None
    s = df.sort_values("period")["value"].dropna()
    if len(s) <= n_back:
        return None, None, None
    latest, prior = s.iloc[-1], s.iloc[-1 - n_back]
    delta = latest - prior
    return latest, delta, ((delta / prior * 100) if prior else None)


def style_fig(fig, yaxis_title="", legend=True):
    fig.update_layout(
        plot_bgcolor=DM_SURFACE, paper_bgcolor=DM_SURFACE,
        font=dict(color=DM_TEXT, size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=DM_BORDER, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=DM_BORDER, showgrid=True, zeroline=False, title=yaxis_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)") if legend else None,
        height=420,
        hovermode="x unified",
    )


def line_fig(df, x, y, name, color, yaxis_title="", hover_fmt=",.2f"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=name,
                             line=dict(color=color, width=2),
                             hovertemplate=f"%{{x|%b %d, %Y}}<br>%{{y:{hover_fmt}}}<extra></extra>"))
    style_fig(fig, yaxis_title)
    return fig


def section_header(title: str):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


# ── Chart/table snapshot toolbar (PNG download + clipboard copy) ─────────────
# Client-side html2canvas screenshot of the real rendered element. Each toolbar is namespaced
# by snap_id and wrapped in an IIFE so several on one page don't collide.
def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', " ", str(name)).strip() or "chart"


def snap_anchor(snap_id: str):
    """Zero-height marker so a following Plotly chart can be found by the snapshot tool."""
    st.markdown(f'<div id="{snap_id}" style="height:0"></div>', unsafe_allow_html=True)


_SNAP_JS = """
<div style="display:flex;gap:6px;align-items:center;margin:2px 0 14px 0">
  <button id="dl__ID__" style="font:600 13px system-ui,sans-serif;color:#fff;
    background:__ACCENT__;border:none;border-radius:6px;padding:6px 12px;
    cursor:pointer">\U0001F4E5 PNG</button>
  <button id="cp__ID__" style="font:600 13px system-ui,sans-serif;color:__ACCENT__;
    background:transparent;border:1.5px solid __ACCENT__;border-radius:6px;padding:6px 12px;
    cursor:pointer">\U0001F4CB Copy</button>
  <span id="msg__ID__" style="font:12px system-ui,sans-serif;color:__MUTED__"></span>
</div>
<script>
(function(){
  const ANCHOR="__ID__", FN="__FN__";
  function ensureH2C(){
    if(window.html2canvas) return Promise.resolve();
    if(window.__h2c) return window.__h2c;
    window.__h2c=new Promise((res,rej)=>{
      const s=document.createElement("script");
      s.src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js";
      s.onload=()=>res(); s.onerror=()=>rej(new Error("load failed"));
      document.head.appendChild(s); setTimeout(()=>rej(new Error("timeout")),10000);
    });
    return window.__h2c;
  }
  // The element to shoot: the tagged element itself when it is, or holds, a table/canvas/svg;
  // otherwise the first Plotly chart that follows the anchor in document order.
  function target(){
    const el=document.getElementById(ANCHOR); if(!el) return null;
    if(el.tagName==="TABLE" || el.querySelector("table,canvas,svg")) return el;
    const charts=[...document.querySelectorAll('[data-testid="stPlotlyChart"]')];
    for(const c of charts){
      if(el.compareDocumentPosition(c) & Node.DOCUMENT_POSITION_FOLLOWING) return c;
    }
    return charts[0]||el;
  }
  async function shoot(){
    await ensureH2C();
    const el=target(); if(!el) throw new Error("nothing to capture");
    return await window.html2canvas(el,{scale:2,backgroundColor:"__BG__",logging:false,useCORS:true});
  }
  const msgEl=document.getElementById("msg__ID__");
  const msg=t=>{ if(msgEl){ msgEl.textContent=t; if(t) setTimeout(()=>{msgEl.textContent="";},2500);} };
  document.getElementById("dl__ID__").onclick=async()=>{
    msg("…"); try{ const c=await shoot(); const a=document.createElement("a");
      a.download=FN+".png"; a.href=c.toDataURL("image/png"); a.click(); msg("✓ saved"); }
    catch(e){ msg("⚠ "+e.message); }
  };
  // Promise-based ClipboardItem so the async html2canvas work keeps the click's user gesture.
  document.getElementById("cp__ID__").onclick=async()=>{
    msg("…");
    try{
      await navigator.clipboard.write([new ClipboardItem({"image/png":(async()=>{
        const c=await shoot();
        return await new Promise(r=>c.toBlob(r,"image/png"));
      })()})]);
      msg("✓ copied");
    }catch(e){ msg("⚠ "+(e.name||e.message)); }
  };
})();
</script>
"""


def snap_toolbar(snap_id: str, filename: str):
    """📥 PNG + 📋 Copy for the element tagged `snap_id` (a table, a wrapper around one, or a
    snap_anchor placed just before a Plotly chart)."""
    st.html(
        _SNAP_JS.replace("__ID__", snap_id).replace("__FN__", _safe_filename(filename))
                .replace("__ACCENT__", JSA_BLUE).replace("__MUTED__", DM_MUTED)
                .replace("__BG__", DM_SURFACE),
        unsafe_allow_javascript=True,
    )
