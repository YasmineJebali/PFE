# pages/5_Deployment_ROI.py — Station-level deployment + ROI ranking

# --- Path bootstrap (must be first) ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Deployment ROI (per-station)", layout="wide")
st.title("🧭 Deployment ROI — station ranking from forecast")

st.caption(
    "Allocate chargers to stations year-by-year to hit your forecast, then compute NPV & payback per station. "
    "Use weighted or round-robin allocation. Download ranked tables for your report."
)

# ---------- Small finance helpers ----------
def annual_net_profit_per_charger(margin_tnd_per_kwh, kwh_per_session, sessions_per_day, opex_per_year):
    revenue_y = margin_tnd_per_kwh * kwh_per_session * sessions_per_day * 365.0
    return float(revenue_y - opex_per_year)

def npv_from_cf(cf, rate):
    disc = np.array([(1.0 + rate) ** t for t in range(len(cf))], dtype=float)
    return float(np.sum(cf / disc))

def payback_year(cf):
    cum = np.cumsum(cf)
    idx = np.where(cum >= 0)[0]
    return int(idx[0]) if len(idx) else None

def station_cashflow_for_schedule(installs_by_t, years, capex, net_profit_y):
    """
    installs_by_t: dict {t_index -> number_of_chargers_installed_that_year} for this station.
    CF timeline length = years+1 (t=0..years). Capex at install year; operating profit for chargers in service.
    """
    T = years
    adds = np.zeros(T + 1, dtype=float)
    for t, n in installs_by_t.items():
        if 0 <= t <= T and n > 0:
            adds[t] += n
    cf = np.zeros(T + 1, dtype=float)
    cf -= capex * adds
    in_service = np.cumsum(adds)
    cf[1:] += net_profit_y * in_service[1:]
    return cf, in_service

# ---------- Inputs: data sources ----------
with st.expander("📥 Inputs (auto-loads from session if available)", expanded=True):
    c1, c2 = st.columns(2)

    # Stations
    with c1:
        st.subheader("Stations")
        st.caption("Required columns: id, name, lat, lon. Optional: gov, nearest_charger_km, underserved_score")
        stations_df = None
        if "stations_df" in st.session_state and isinstance(st.session_state["stations_df"], pd.DataFrame):
            stations_df = st.session_state["stations_df"].copy()
            st.info("Loaded stations from session.")
        up_s = st.file_uploader("Or upload stations CSV", type=["csv"], key="stations_deploy_roi")
        if up_s is not None:
            stations_df = pd.read_csv(up_s)

    # Forecast
    with c2:
        st.subheader("Forecast (Tunisia)")
        st.caption("Required columns: year, chargers_needed (cumulative). Optional: ev_stock_tn")
        forecast_df = None
        if "forecast_df" in st.session_state and isinstance(st.session_state["forecast_df"], pd.DataFrame):
            forecast_df = st.session_state["forecast_df"].copy()
            st.info("Loaded forecast from session.")
        up_f = st.file_uploader("Or upload forecast CSV", type=["csv"], key="forecast_deploy_roi")
        if up_f is not None:
            forecast_df = pd.read_csv(up_f)

# Validation
if stations_df is None or stations_df.empty:
    st.error("No stations loaded. Load from session or upload a CSV.")
    st.stop()
if forecast_df is None or forecast_df.empty:
    st.error("No forecast loaded. Load from session or upload a CSV.")
    st.stop()

# Coerce & clean
stations_df.columns = [c.strip() for c in stations_df.columns]
forecast_df.columns = [c.strip().lower() for c in forecast_df.columns]
need_st = {"id", "name", "lat", "lon"}
need_fc = {"year", "chargers_needed"}
miss_st = need_st - set(stations_df.columns)
miss_fc = need_fc - set(forecast_df.columns)
if miss_st:
    st.error(f"Stations missing: {', '.join(sorted(miss_st))}")
    st.stop()
if miss_fc:
    st.error(f"Forecast missing: {', '.join(sorted(miss_fc))}")
    st.stop()

stations_df["lat"] = pd.to_numeric(stations_df["lat"], errors="coerce")
stations_df["lon"] = pd.to_numeric(stations_df["lon"], errors="coerce")
forecast_df["year"] = pd.to_numeric(forecast_df["year"], errors="coerce")
forecast_df["chargers_needed"] = pd.to_numeric(forecast_df["chargers_needed"], errors="coerce")
forecast_df = forecast_df.dropna().sort_values("year").reset_index(drop=True)

st.write("### Quick previews")
cA, cB = st.columns([1.2, 1])
with cA:
    st.dataframe(stations_df.head(15), use_container_width=True, height=280)
with cB:
    st.dataframe(forecast_df.head(15), use_container_width=True, height=280)
    if stations_df[["lat","lon"]].dropna().shape[0] > 0:
        st.caption("Stations map (approx. TN bounds)")
        m = stations_df.rename(columns={"lat":"latitude","lon":"longitude"})[["latitude","longitude"]]
        st.map(m)

# ---------- Assumptions ----------
st.write("---")
st.subheader("⚙️ Economic assumptions")

c1, c2, c3, c4, c5 = st.columns(5)
capex = c1.number_input("CAPEX per charger (TND)", 2000, 200000, 36000, step=1000)
opex  = c2.number_input("OPEX per year (TND)", 200, 50000, 2500, step=100)
margin= c3.number_input("Margin per kWh (TND)", 0.05, 5.0, 0.55, step=0.05, format="%.2f")
kwh   = c4.number_input("kWh per session", 5, 150, 22, step=1)
sess  = c5.number_input("Base sessions/day (per installed charger)", 1, 100, 3, step=1)

rate = st.slider("Discount rate", 0.01, 0.30, 0.11, 0.01, format="%.2f")

# Weighting (traffic proxy)
st.write("### Allocation strategy")
cL, cR = st.columns(2)
with cL:
    strategy = st.radio(
        "Distribute new chargers each year by:",
        ["Round-robin (even)", "Weighted 'need'"],
        horizontal=True
    )
with cR:
    need_col = st.selectbox(
        "Weight column (if 'Weighted')",
        options=["auto: underserved_score", "auto: nearest_charger_km", "constant (all=1)"] +
                [c for c in stations_df.columns if c not in {"id","name","lat","lon"}],
        index=0
    )
alpha = st.slider("Sessions/day uplift factor from weight (0=no effect, 2=strong)", 0.0, 2.0, 0.5, 0.1)

# Build weight vector
def pick_weights(df, need_col_choice):
    if need_col_choice == "auto: underserved_score" and "underserved_score" in df.columns:
        w = pd.to_numeric(df["underserved_score"], errors="coerce")
    elif need_col_choice == "auto: nearest_charger_km" and "nearest_charger_km" in df.columns:
        w = pd.to_numeric(df["nearest_charger_km"], errors="coerce")
    elif need_col_choice == "constant (all=1)":
        w = pd.Series(1.0, index=df.index)
    elif need_col_choice in df.columns:
        w = pd.to_numeric(df[need_col_choice], errors="coerce")
    else:
        # fallback
        w = pd.Series(1.0, index=df.index)
    w = w.fillna(w.median() if np.isfinite(w.median()) else 1.0)
    # normalize to 0..1 (avoid zero variance)
    if w.max() > w.min():
        w_norm = (w - w.min()) / (w.max() - w.min())
    else:
        w_norm = pd.Series(0.5, index=w.index)
    return w_norm

w_norm = pick_weights(stations_df, need_col)
stations_df["_weight"] = w_norm

# Sessions/day per station = base * (1 + alpha * weight)
stations_df["_sessions_day"] = sess * (1.0 + alpha * stations_df["_weight"])

# ---------- Build yearly install plan from forecast ----------
years_abs = forecast_df["year"].to_numpy()
start_year = int(years_abs.min())
end_year = int(years_abs.max())
horizon_years = end_year - start_year  # number of intervals; timeline indices t=0..horizon_years

cum = forecast_df["chargers_needed"].to_numpy()
adds = np.r_[cum[0], np.diff(cum)]
adds = np.where(adds < 0, 0, adds)  # guard against accidental negatives
adds_per_year = {int(y - start_year): int(a) for y, a in zip(years_abs, adds)}

st.write("### Yearly new chargers needed (from forecast)")
df_adds = pd.DataFrame({"year": years_abs, "adds": adds.astype(int), "cumulative": cum.astype(int)})
st.dataframe(df_adds, use_container_width=True, height=200)

# ---------- Allocate installs to stations ----------
N = len(stations_df)
station_ids = stations_df["id"].tolist()

def allocate_round_robin(adds_dict, N):
    """
    Returns dict: station_id -> {t_index -> installs}
    Evenly spreads new chargers each year across stations.
    """
    install = {sid: {} for sid in station_ids}
    rr_index = 0
    for t in range(0, horizon_years + 1):
        to_place = int(adds_dict.get(t, 0))
        for _ in range(to_place):
            sid = station_ids[rr_index % N]
            install[sid][t] = install[sid].get(t, 0) + 1
            rr_index += 1
    return install

def allocate_weighted(adds_dict, weights):
    """
    Returns dict: station_id -> {t_index -> installs}
    Distributes per year proportional to weights.
    """
    w = np.array(weights, dtype=float)
    w = np.where(w < 0, 0, w)
    if w.sum() == 0:
        w = np.ones_like(w)
    p = w / w.sum()

    install = {sid: {} for sid in station_ids}
    for t in range(0, horizon_years + 1):
        to_place = int(adds_dict.get(t, 0))
        if to_place <= 0:
            continue
        # multinomial draw (deterministic expectation rounding)
        exp = p * to_place
        base = np.floor(exp).astype(int)
        remainder = to_place - base.sum()
        # assign the remainder to the highest fractional parts
        frac_rank = np.argsort(-(exp - base))
        for i in range(remainder):
            base[frac_rank[i]] += 1
        # write into dict
        for idx, count in enumerate(base):
            if count > 0:
                sid = station_ids[idx]
                install[sid][t] = install[sid].get(t, 0) + int(count)
    return install

if strategy == "Round-robin (even)":
    schedule = allocate_round_robin(adds_per_year, N)
else:
    schedule = allocate_weighted(adds_per_year, stations_df["_weight"].to_numpy())

# ---------- Compute per-station ROI ----------
capex_f = float(capex)
rate_f = float(rate)

rows = []
per_station_cf = {}

for i, row in stations_df.iterrows():
    sid = row["id"]
    sname = row["name"]
    gov = row.get("gov", "")
    sessions_day_i = float(row["_sessions_day"])

    net_y = annual_net_profit_per_charger(margin, kwh, sessions_day_i, opex)
    inst_dict = schedule.get(sid, {})
    cf_i, in_serv_i = station_cashflow_for_schedule(inst_dict, horizon_years, capex_f, net_y)

    npv_i = npv_from_cf(cf_i, rate_f)
    pb_i = payback_year(cf_i)
    installs_total = int(sum(inst_dict.values()))
    first_t = min(inst_dict.keys()) if inst_dict else None
    first_year = (start_year + first_t) if first_t is not None else None

    rows.append({
        "station_id": sid,
        "name": sname,
        "gov": gov,
        "installs_total": installs_total,
        "first_year": first_year,
        "sessions_day": round(sessions_day_i, 2),
        "NPV_TND": npv_i,
        "payback_year_index": pb_i,
    })
    per_station_cf[sid] = {"cf": cf_i, "in_service": in_serv_i, "installs": inst_dict}

df_rank = pd.DataFrame(rows)
df_rank = df_rank.sort_values(["installs_total", "NPV_TND"], ascending=[False, False]).reset_index(drop=True)

st.write("---")
st.subheader("🏆 Ranked stations (by installs then NPV)")
st.dataframe(df_rank.head(50), use_container_width=True, height=360)

st.download_button(
    "⬇️ Download full station ranking (CSV)",
    data=df_rank.to_csv(index=False).encode(),
    file_name="station_roi_ranking.csv",
    use_container_width=True,
)

# ---------- Quick map of top candidates ----------
if stations_df[["lat","lon"]].dropna().shape[0] > 0 and not df_rank.empty:
    st.write("### Map — top 50 by NPV")
    top_ids = set(df_rank.head(50)["station_id"].tolist())
    m = stations_df[stations_df["id"].isin(top_ids)].rename(columns={"lat":"latitude","lon":"longitude"})
    if not m.empty:
        st.map(m[["latitude","longitude"]])

# ---------- Drill-down: one station ----------
st.write("---")
st.subheader("🔎 Drill-down a station")
options = df_rank["station_id"].astype(str) + " — " + df_rank["name"].astype(str)
pick_label = st.selectbox("Pick a station", options.tolist())
if pick_label:
    sid = df_rank.loc[options == pick_label, "station_id"].iloc[0]
    meta = df_rank[df_rank["station_id"] == sid].iloc[0].to_dict()
    cf = per_station_cf[sid]["cf"]
    in_serv = per_station_cf[sid]["in_service"]
    installs = per_station_cf[sid]["installs"]

    st.write(f"**{meta['name']}**  (gov: {meta.get('gov','')})")
    st.write(f"- Installs total: **{meta['installs_total']}**, first year: **{meta['first_year']}**")
    st.write(f"- Sessions/day assumed: **{meta['sessions_day']}**")
    st.write(f"- NPV (TND): **{meta['NPV_TND']:,.0f}**, Payback year index: **{meta['payback_year_index']}** (relative to {start_year})")

    # Build a per-year table for this station
    years_t = np.arange(0, horizon_years + 1, dtype=int)
    df_detail = pd.DataFrame({
        "year": start_year + years_t,
        "installs_this_year": [installs.get(int(t), 0) for t in years_t],
        "chargers_in_service": in_serv,
        "cashflow": cf
    })
    st.dataframe(df_detail, use_container_width=True, height=300)

    st.download_button(
        "⬇️ Download this station's schedule (CSV)",
        data=df_detail.to_csv(index=False).encode(),
        file_name=f"station_{sid}_schedule_cashflows.csv",
        use_container_width=True,
    )
