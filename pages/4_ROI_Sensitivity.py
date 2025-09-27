# pages/4_ROI_Sensitivity.py — ROI with Payback, Discounted Payback, IRR, Portfolio, Tornado, Charts & Excel pack

# --- Path bootstrap (must be first) ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Optional config (paths)
try:
    from config import PATHS
except Exception:
    class _Paths:
        tn_forecast_out = ROOT / "data" / "tn_ev_forecast.csv"
    PATHS = _Paths()

st.set_page_config(page_title="ROI Sensitivity", layout="wide")
st.title("💸 ROI — NPV, Payback, Discounted Payback, IRR, Portfolio & Tornado")

st.caption(
    "Tune unit economics. Optionally link to your Tunisia forecast to compute a portfolio NPV. "
    "See discounted payback visually and export everything to Excel."
)

# ---------------- Helpers ----------------
def annual_net_profit_per_charger(margin_tnd_per_kwh, kwh_per_session, sessions_per_day, opex_per_year):
    revenue_y = margin_tnd_per_kwh * kwh_per_session * sessions_per_day * 365.0
    return float(revenue_y - opex_per_year)

def cashflows_per_charger(capex, net_profit_per_year, years):
    """CF[0..years] undiscounted, capex at t=0 (negative), then yearly net profit."""
    cf = np.zeros(years + 1, dtype=float)
    cf[0] = -capex
    if years >= 1:
        cf[1:] = net_profit_per_year
    return cf

def npv_from_cf(cf, rate):
    disc = np.array([(1.0 + rate) ** t for t in range(len(cf))], dtype=float)
    return float(np.sum(cf / disc))

def payback_year(cf):
    """Undiscounted payback year (first t where cumulative >= 0). Returns int or None."""
    cum = np.cumsum(cf)
    idx = np.where(cum >= 0)[0]
    return int(idx[0]) if len(idx) else None

def discounted_payback_year(cf, rate):
    """Discounted payback (first t where discounted cumulative >= 0). Returns int or None."""
    disc = np.array([(1.0 + rate) ** t for t in range(len(cf))], dtype=float)
    cum_disc = np.cumsum(cf / disc)
    idx = np.where(cum_disc >= 0)[0]
    return int(idx[0]) if len(idx) else None

def irr_from_cf(cf, guess=0.1):
    """
    IRR using numpy_financial if available; otherwise robust bisection on [-0.99, 5.0].
    Returns float or None if no sign change / no root.
    """
    try:
        import numpy_financial as npf
        val = float(npf.irr(cf))
        if np.isfinite(val):
            return val
    except Exception:
        pass

    # Fallback: bisection
    def npv_at(r): return npv_from_cf(cf, r)
    lo, hi = -0.99, 5.0
    f_lo, f_hi = npv_at(lo), npv_at(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return None  # no guaranteed root
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = npv_at(mid)
        if not np.isfinite(f_mid): return None
        if abs(f_mid) < 1e-9: return float(mid)
        if f_lo * f_mid <= 0: hi, f_hi = mid, f_mid
        else: lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))

def build_portfolio_cashflows(adds_by_year, capex, net_profit_per_year, years):
    """
    Portfolio CF with installs over time:
    - capex applied in the install year for the number of new chargers
    - operating net profit each year = net_profit_per_year * chargers_in_service_that_year
    """
    T = years
    adds = np.zeros(T + 1, dtype=float)
    for t, n in adds_by_year.items():
        if 0 <= t <= T:
            adds[t] += float(max(n, 0))

    cf = np.zeros(T + 1, dtype=float)
    cf -= capex * adds  # capex at install
    in_service = np.cumsum(adds)
    cf[1:] += net_profit_per_year * in_service[1:]  # operating profit
    return cf, in_service

def make_tornado(rowspecs, pct=0.2):
    """rowspecs: list of (name, base_value, fn(value)) → returns DataFrame sorted by impact."""
    out = []
    for name, base, fn in rowspecs:
        low_v, high_v = base * (1 - pct), base * (1 + pct)
        npv_low, npv_high = float(fn(low_v)), float(fn(high_v))
        lo, hi = sorted([npv_low, npv_high])
        out.append({"param": name, "low": lo, "high": hi, "impact": abs(hi - lo)})
    df = pd.DataFrame(out).sort_values("impact", ascending=True)
    return df

def discounted_cumulative_df(cf, rate, label="Per-charger"):
    """Return table with year, undiscounted CF, discounted CF, and cumulative discounted CF."""
    years_idx = np.arange(0, len(cf))
    disc = np.array([(1.0 + rate) ** t for t in years_idx], dtype=float)
    disc_cf = cf / disc
    cum_disc = np.cumsum(disc_cf)
    return pd.DataFrame({
        "year": years_idx,
        "cashflow": cf,
        "discounted_cf": disc_cf,
        "cum_discounted_cf": cum_disc,
        "series": label
    })

def discounted_cum_chart(df, title="Discounted cumulative cashflow"):
    base = alt.Chart(df)
    line = base.mark_line(point=True).encode(
        x=alt.X("year:Q", title="Year (t)"),
        y=alt.Y("cum_discounted_cf:Q", title="Cumulative discounted cashflow (TND)"),
        color=alt.Color("series:N", title="Series"),
        tooltip=[
            alt.Tooltip("year:Q"),
            alt.Tooltip("cum_discounted_cf:Q", format=",.0f"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[6,3], color="black").encode(y="y:Q")
    return (line + zero).properties(title=title, height=320)

# ---------------- Inputs ----------------
c1, c2, c3, c4, c5 = st.columns(5)
capex = c1.number_input("CAPEX per charger (TND)", 2000, 200000, 36000, step=1000)
opex  = c2.number_input("OPEX per year (TND)", 200, 50000, 2500, step=100)
margin= c3.number_input("Margin per kWh (TND)", 0.05, 5.0, 0.55, step=0.05, format="%.2f")
kwh   = c4.number_input("kWh per session", 5, 150, 22, step=1)
sess  = c5.number_input("Sessions per day", 1, 100, 3, step=1)

years = st.slider("Project horizon (years)", 3, 20, 8, 1)
rate  = st.slider("Discount rate", 0.01, 0.30, 0.11, 0.01, format="%.2f")

net_profit_y = annual_net_profit_per_charger(margin, kwh, sess, opex)

# ---------------- Per-charger model ----------------
st.write("---")
st.subheader("Per-charger economics")

cf_unit = cashflows_per_charger(capex, net_profit_y, years)
npv_unit = npv_from_cf(cf_unit, rate)
pb_unit = payback_year(cf_unit)
dpb_unit = discounted_payback_year(cf_unit, rate)
irr_unit = irr_from_cf(cf_unit)

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("NPV per charger (TND)", f"{npv_unit:,.0f}")
mcol2.metric("Payback (undiscounted)", "No payback" if pb_unit is None else f"Year {pb_unit}")
mcol3.metric("Discounted payback", "No payback" if dpb_unit is None else f"Year {dpb_unit}")

mcol4, _ = st.columns([1,2])
mcol4.metric("IRR", "n/a" if irr_unit is None else f"{irr_unit*100:.1f}%")

st.caption("Cashflows per charger (undiscounted):")
df_unit_cf = pd.DataFrame({"year": np.arange(0, years + 1), "cashflow": cf_unit})
st.bar_chart(df_unit_cf.set_index("year"))

# NEW: discounted cumulative chart for per-charger
df_unit_disc = discounted_cumulative_df(cf_unit, rate, label="Per-charger")
st.altair_chart(discounted_cum_chart(df_unit_disc, "Discounted cumulative — per charger"), use_container_width=True)

st.download_button(
    "⬇️ Download per-charger cashflows (CSV)",
    data=df_unit_cf.to_csv(index=False).encode(),
    file_name="per_charger_cashflows.csv",
    use_container_width=True,
)

# ---------------- Portfolio (link to forecast) ----------------
st.write("---")
st.subheader("Portfolio mode (link to Tunisia forecast)")

use_portfolio = st.checkbox("Use forecast-linked portfolio NPV", value=False,
                            help="If checked, we load tn_ev_forecast.csv and compute NPV for a growing fleet.")

dfF = None
src_msg = None
if use_portfolio:
    if "forecast_df" in st.session_state and isinstance(st.session_state["forecast_df"], pd.DataFrame):
        dfF = st.session_state["forecast_df"].copy()
        src_msg = "Loaded forecast from session."
    else:
        default_p = Path(PATHS.tn_forecast_out)
        up = st.file_uploader("Upload tn_ev_forecast.csv (columns: year, ev_stock_tn, chargers_needed)",
                              type=["csv"], key="upload_forecast_roi")
        if up is not None:
            dfF = pd.read_csv(up)
            src_msg = f"Loaded uploaded file: {getattr(up, 'name', '')}"
        elif default_p.exists():
            dfF = pd.read_csv(default_p)
            src_msg = f"Loaded default: {default_p}"

if use_portfolio and (dfF is None or dfF.empty):
    st.warning("No forecast available. Uncheck portfolio mode or upload a forecast CSV.")
    use_portfolio = False

adds_by_year_idx = {}
if use_portfolio:
    st.caption(src_msg)
    need_cols = {"year", "chargers_needed"}
    missing = need_cols - set(dfF.columns)
    if missing:
        st.error(f"Forecast missing required columns: {', '.join(sorted(missing))}")
        use_portfolio = False
    else:
        dfF = dfF[["year", "chargers_needed"]].dropna().copy()
        dfF["year"] = pd.to_numeric(dfF["year"], errors="coerce")
        dfF["chargers_needed"] = pd.to_numeric(dfF["chargers_needed"], errors="coerce")
        dfF = dfF.dropna().sort_values("year").reset_index(drop=True)

if use_portfolio:
    st.dataframe(dfF.head(15), use_container_width=True)

    # Convert cumulative "chargers_needed" into yearly additions
    y0 = int(dfF["year"].iloc[0])
    cum = dfF["chargers_needed"].to_numpy()
    adds = np.r_[cum[0], np.diff(cum)]
    adds = np.where(adds < 0, 0, adds)

    # Build schedule dict: year_index (0..years) -> adds
    for i, y in enumerate(dfF["year"]):
        t = int(y - y0)
        if 0 <= t <= years:
            adds_by_year_idx[t] = adds[i]

    cf_port, in_service = build_portfolio_cashflows(adds_by_year_idx, capex, net_profit_y, years)
    npv_port = npv_from_cf(cf_port, rate)
    pb_port = payback_year(cf_port)
    dpb_port = discounted_payback_year(cf_port, rate)
    irr_port = irr_from_cf(cf_port)

    cA, cB, cC = st.columns(3)
    cA.metric("Portfolio NPV (TND)", f"{npv_port:,.0f}")
    cB.metric("Payback (undiscounted)", "No payback" if pb_port is None else f"Year {pb_port}")
    cC.metric("Discounted payback", "No payback" if dpb_port is None else f"Year {dpb_port}")

    cD, _ = st.columns([1,2])
    cD.metric("Portfolio IRR", "n/a" if irr_port is None else f"{irr_port*100:.1f}%")

    st.caption("Chargers in service and annual cashflow (portfolio):")
    df_port = pd.DataFrame({
        "year_index": np.arange(0, years + 1),
        "chargers_in_service": in_service,
        "cashflow": cf_port
    })
    st.line_chart(df_port.set_index("year_index")[["chargers_in_service"]])
    st.bar_chart(df_port.set_index("year_index")[["cashflow"]])

    # NEW: discounted cumulative chart for portfolio
    df_port_disc = discounted_cumulative_df(cf_port, rate, label="Portfolio")
    st.altair_chart(discounted_cum_chart(df_port_disc, "Discounted cumulative — portfolio"),
                    use_container_width=True)

    st.download_button(
        "⬇️ Download portfolio cashflows (CSV)",
        data=df_port.to_csv(index=False).encode(),
        file_name="portfolio_cashflows.csv",
        use_container_width=True,
    )
else:
    df_port = None
    df_port_disc = None

# ---------------- Tornado sensitivity ----------------
st.write("---")
st.subheader("Tornado sensitivity (±20%)")

if use_portfolio:
    # Sensitivity recomputing PORTFOLIO NPV
    def npv_with(capex_=None, opex_=None, margin_=None, kwh_=None, sess_=None, rate_=None):
        capex_i = capex if capex_ is None else capex_
        opex_i  = opex if opex_ is None else opex_
        rate_i  = rate if rate_ is None else rate_
        margin_i= margin if margin_ is None else margin_
        kwh_i   = kwh if kwh_ is None else kwh_
        sess_i  = sess if sess_ is None else sess_
        net_i = annual_net_profit_per_charger(margin_i, kwh_i, sess_i, opex_i)
        cf_i, _ = build_portfolio_cashflows(adds_by_year_idx, capex_i, net_i, years)
        return npv_from_cf(cf_i, rate_i)

    rowspecs = [
        ("CAPEX", capex, lambda v: npv_with(capex_=v)),
        ("OPEX", opex, lambda v: npv_with(opex_=v)),
        ("Margin/kWh", margin, lambda v: npv_with(margin_=v)),
        ("kWh/session", kwh, lambda v: npv_with(kwh_=v)),
        ("Sessions/day", sess, lambda v: npv_with(sess_=v)),
        ("Discount rate", rate, lambda v: npv_with(rate_=v)),
    ]
    base_cf = build_portfolio_cashflows(adds_by_year_idx, capex, net_profit_y, years)[0]
else:
    # Sensitivity recomputing PER-CHARGER NPV
    def npv_with(capex_=None, opex_=None, margin_=None, kwh_=None, sess_=None, rate_=None):
        capex_i = capex if capex_ is None else capex_
        opex_i  = opex if opex_ is None else opex_
        rate_i  = rate if rate_ is None else rate_
        margin_i= margin if margin_ is None else margin_
        kwh_i   = kwh if kwh_ is None else kwh_
        sess_i  = sess if sess_ is None else sess_
        net_i = annual_net_profit_per_charger(margin_i, kwh_i, sess_i, opex_i)
        cf_i = cashflows_per_charger(capex_i, net_i, years)
        return npv_from_cf(cf_i, rate_i)

    rowspecs = [
        ("CAPEX", capex, lambda v: npv_with(capex_=v)),
        ("OPEX", opex, lambda v: npv_with(opex_=v)),
        ("Margin/kWh", margin, lambda v: npv_with(margin_=v)),
        ("kWh/session", kwh, lambda v: npv_with(kwh_=v)),
        ("Sessions/day", sess, lambda v: npv_with(sess_=v)),
        ("Discount rate", rate, lambda v: npv_with(rate_=v)),
    ]
    base_cf = cashflows_per_charger(capex, net_profit_y, years)

df_tornado = make_tornado(rowspecs=rowspecs, pct=0.2)
base_npv = npv_from_cf(base_cf, rate)

base_line = alt.Chart(pd.DataFrame({"NPV": [base_npv]})).mark_rule(color="black").encode(x="NPV:Q")
bars = alt.Chart(df_tornado).mark_bar().encode(
    y=alt.Y("param:N", title="Parameter"),
    x=alt.X("low:Q", title="NPV (TND)"),
    x2="high:Q",
    tooltip=[alt.Tooltip("param:N"),
             alt.Tooltip("low:Q", format=",.0f"),
             alt.Tooltip("high:Q", format=",.0f")]
).properties(height=280)

st.altair_chart((bars + base_line).resolve_scale(x="shared"), use_container_width=True)

st.download_button(
    "⬇️ Download tornado table (CSV)",
    data=df_tornado.to_csv(index=False).encode(),
    file_name="tornado_table.csv",
    use_container_width=True,
)

# ---------------- Excel Export Pack ----------------
st.write("---")
st.subheader("📦 Export Excel pack")

# Build assumptions sheet
assumptions = pd.DataFrame({
    "parameter": ["CAPEX per charger (TND)", "OPEX per year (TND)",
                  "Margin per kWh (TND)", "kWh per session", "Sessions per day",
                  "Project horizon (years)", "Discount rate"],
    "value": [capex, opex, margin, kwh, sess, years, rate],
})

# Collect tables for export
sheets = {
    "Assumptions": assumptions,
    "PerCharger_CF": df_unit_cf,
    "PerCharger_Discounted": df_unit_disc[["year", "discounted_cf", "cum_discounted_cf"]],
    "Tornado": df_tornado,
}
if use_portfolio and df_port is not None:
    sheets["Portfolio_CF"] = df_port
    sheets["Portfolio_Discounted"] = df_port_disc[["year", "discounted_cf", "cum_discounted_cf"]].rename(columns={"year": "year_index"})

# Write to an in-memory Excel file
buf = io.BytesIO()
engine = None
for candidate in ("xlsxwriter", "openpyxl"):
    try:
        with pd.ExcelWriter(buf, engine=candidate) as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
        engine = candidate
        break
    except Exception:
        buf = io.BytesIO()  # reset and try next engine
        continue

if engine is None:
    st.error("Could not find an Excel engine. Try:  pip install xlsxwriter  (or openpyxl).")
else:
    st.success(f"Excel built with engine: {engine}")
    st.download_button(
        "⬇️ Download ROI pack (Excel)",
        data=buf.getvalue(),
        file_name="roi_pack.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
