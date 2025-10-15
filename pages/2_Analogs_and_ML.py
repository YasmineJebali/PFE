# pages/2_Analogs_and_ML.py — Analogs fit + ML baseline + Tunisia scenario + comparison
# Updated: outlier toggles, milestones, correlated (r, t0) uncertainty, robust fallbacks.

# --- 0) Path bootstrap: ensure project root is on sys.path (MUST be first) ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]   # .../agil-ev-mvp
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------------------------------------

# --- 1) Standard imports ---
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.page_link("pages/6_Chatbot.py", label="💬 Ask the Assistant")

# --- 2) Local package imports (with safe fallbacks) ---
# Logger
try:
    from utils.logging_utils import get_logger
except Exception:
    import logging, sys as _sys
    def get_logger(name="agil"):
        log = logging.getLogger(name)
        if not log.handlers:
            log.setLevel(logging.INFO)
            h = logging.StreamHandler(_sys.stdout)
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            log.addHandler(h)
        return log
log = get_logger()

from utils.ui import set_page, sidebar_nav, breadcrumbs, status_pills
set_page("Analogs & ML Forecasting", icon="📊")
sidebar_nav(__file__)
breadcrumbs([
    ("Home", "pages/0_Home.py"),
    ("Analogs & ML", "pages/2_Analogs_and_ML.py"),
])
status_pills()


# Config paths
try:
    from config import PATHS
except Exception:
    class _Paths:
        analog_csv = ROOT / "data" / "analog_ev_full.csv"
        tn_tech_csv = ROOT / "data" / "tn_tech_real.csv"
        tn_forecast_out = ROOT / "data" / "tn_ev_forecast.csv"
    PATHS = _Paths()

# Models & helpers
from models.logistic_bass import logistic, fit_logistic, rough_fit, evaluate_fit
from sklearn.ensemble import RandomForestRegressor

# Optional: Pandera validation (auto-disabled if not installed)
try:
    from schemas import AnalogSchema, TnTechSchema  # your schemas.py
    import pandera as pa
    USE_PANDERA = True
except Exception:
    AnalogSchema = TnTechSchema = None
    USE_PANDERA = False
    log.info("Pandera/schemas not available — validation disabled.")

# Optional: shared analytics helpers (safe fallbacks if file not present)
try:
    from utils.analytics import (
        logistic_milestones, flag_outliers, draw_correlated, analog_rt0_stats
    )
    HAVE_ANALYTICS = True
except Exception:
    HAVE_ANALYTICS = False
    log.info("utils.analytics not found — using minimal internal fallbacks.")

    def logistic_milestones(K, r, t0, levels=(0.1, 0.5, 0.9)):
        out = {}
        for p in levels:  # t = t0 - (1/r)*ln(1/p - 1)
            out[int(round(p*100))] = float(t0 - (1.0/r) * np.log(1.0/p - 1.0))
        return out

    def flag_outliers(series, z=3.0):
        s = pd.Series(series, dtype=float)
        m, sd = s.mean(), s.std(ddof=1)
        sd = sd if sd > 0 else 1.0
        return (np.abs((s - m) / sd) > z)

    def analog_rt0_stats(fit_rows):
        arr = np.asarray(fit_rows, float)
        if arr.shape[0] < 2:
            mu = np.array([0.35, 2031.0])
            cov = np.diag([0.05**2, 1.5**2])
            return mu, cov
        return arr.mean(axis=0), np.cov(arr.T)

    def draw_correlated(mu, cov, n=2000, seed=42):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(mu, cov, size=n)

# --- 3) Streamlit page config ---
st.set_page_config(page_title="Analogs & ML Forecasting", layout="wide")
st.title("🔬 Analogs & ML Forecasting")

st.caption("Pick a country analog to estimate S-curve parameters, compare fits, then build a Tunisia scenario with uncertainty bands.")

# --- 4) Small helpers ---
@st.cache_data(show_spinner=False)
def read_csv_cached(path_or_file):
    return pd.read_csv(path_or_file)

def clean_analog_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize, coerce numerics, and drop rows missing country/year/ev_stock."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "Country": "country", "country": "country",
        "Year": "year", "YEAR": "year",
        "EV stock": "ev_stock", "EV_stock": "ev_stock", "ev_stock": "ev_stock",
        "Public charging points": "public_chargers", "public_chargers": "public_chargers",
        "ISO3": "iso3", "iso3": "iso3",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    for col in ["year", "ev_stock", "public_chargers"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    must_have = [c for c in ["country", "year", "ev_stock"] if c in df.columns]
    before = len(df)
    if must_have:
        df = df.dropna(subset=must_have)
    dropped = before - len(df)
    if dropped > 0:
        st.warning(f"🧹 Dropped {dropped} row(s) with missing {', '.join(must_have)}.")
    return df.reset_index(drop=True)

def clean_tntech_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize tech data: lower cols, numeric year/adoption_pct, drop nulls."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["year", "adoption_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["tech", "year", "adoption_pct"], how="any").reset_index(drop=True)
    return df

# --- 5) Inputs (upload or defaults) ---
st.write("---")
with st.expander("📥 Inputs", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Analog EV data")
        st.caption("Expected columns: country, iso3, year, ev_stock, public_chargers")
        up_a = st.file_uploader("Upload analogs CSV", type=["csv"], key="analog")
        if up_a is not None:
            dfA = read_csv_cached(up_a)
            st.info(f"Loaded uploaded file: {getattr(up_a, 'name', '')}")
        elif os.path.exists(PATHS.analog_csv):
            st.info(f"Loaded default: {PATHS.analog_csv}")
            dfA = read_csv_cached(str(PATHS.analog_csv))
        else:
            dfA = None

    with c2:
        st.subheader("Tunisia tech (WDI)")
        st.caption("Expected columns: tech (internet/mobile), year, adoption_pct")
        up_t = st.file_uploader("Upload Tunisia tech CSV", type=["csv"], key="tntech")
        if up_t is not None:
            dfTN = read_csv_cached(up_t)
            st.info(f"Loaded uploaded file: {getattr(up_t, 'name', '')}")
        elif os.path.exists(PATHS.tn_tech_csv):
            st.info(f"Loaded default: {PATHS.tn_tech_csv}")
            dfTN = read_csv_cached(str(PATHS.tn_tech_csv))
        else:
            dfTN = None

if dfA is None or dfA.empty:
    st.error("No analog EV dataset loaded. Upload a CSV or place one at the default path.")
    st.stop()

# --- 6) Clean & validate data ---
dfA = clean_analog_df(dfA)
if USE_PANDERA and AnalogSchema is not None:
    try:
        dfA = AnalogSchema.validate(dfA)
    except Exception as e:
        st.error("Analog dataset failed schema validation.")
        st.exception(e)
        st.stop()

if dfTN is not None and not dfTN.empty:
    dfTN = clean_tntech_df(dfTN)
    if USE_PANDERA and TnTechSchema is not None:
        try:
            dfTN = TnTechSchema.validate(dfTN)
        except Exception as e:
            st.warning("Tunisia tech dataset failed schema validation. Continuing without it.")
            st.exception(e)
            dfTN = None

# Only keep countries with ≥2 EV-stock points to avoid degenerate fits
counts = dfA.groupby("country")["ev_stock"].count()
countries = sorted([c for c, n in counts.items() if n >= 2])
if not countries:
    st.error("No countries with at least 2 EV stock points after cleaning.")
    st.stop()

# --- 7) Analog fit (Left) & Tunisia tech fit (Right) ---
st.write("---")
cL, cR = st.columns(2)

with cL:
    st.subheader("Analog fit (EV stock)")
    pick = st.selectbox("Pick a country", countries, index=0)

    sub = dfA[dfA["country"] == pick].sort_values("year").copy()

    # NEW: optional outlier filtering on analog EV stock
    hide_out = st.checkbox("Hide outliers (z>3) in EV stock", value=False, key="analog_outliers")
    if hide_out and len(sub) >= 6:
        mask_out = ~flag_outliers(sub["ev_stock"], z=3.0)
        dropped = int((~mask_out).sum())
        if dropped:
            st.info(f"Outliers hidden: {dropped} point(s).")
        sub = sub[mask_out].copy()

    st.caption(f"{pick} — {len(sub)} data points after filtering")
    st.dataframe(sub[["country", "iso3", "year", "ev_stock", "public_chargers"]].head(20),
                 use_container_width=True)

    years = sub["year"].to_numpy()
    y = sub["ev_stock"].to_numpy()

    if len(sub) >= 4:
        fit = fit_logistic(years, y, K0=None, r0=0.30, t0=None)
        st.success(
            f"{pick} fit → K={fit.K:.0f}, r={fit.r:.3f}, t0={fit.t0:.1f} "
            f"(status: {'OK' if fit.success else 'rough-init used'})"
        )
        rmse_v, mape_v = evaluate_fit(years, y, fit.K, fit.r, fit.t0)
        st.caption(f"Fit quality → RMSE={rmse_v:.1f}, MAPE={mape_v:.1%}")
        aK, aR, aT0 = float(fit.K), float(fit.r), float(fit.t0)
    else:
        K_r, r_r, t0_r = rough_fit(years, y)
        st.info(f"{pick} rough fit → K≈{K_r:.0f}, r≈{r_r:.3f}, t0≈{t0_r:.1f}")
        rmse_v, mape_v = evaluate_fit(years, y, K_r, r_r, t0_r)
        st.caption(f"Rough fit quality → RMSE≈{rmse_v:.1f}, MAPE≈{mape_v:.1%}")
        aK, aR, aT0 = float(K_r), float(r_r), float(t0_r)

    # NEW: Milestones (10/50/90%)
    ms = logistic_milestones(aK, aR, aT0, levels=(0.1, 0.5, 0.9))
    st.caption(f"Milestones → 10% in {int(round(ms[10]))}, 50% (t₀) ≈ {int(round(aT0))}, 90% in {int(round(ms[90]))}.")

    # Plot actual + fitted curve
    years_line = np.arange(int(years.min()) - 2, int(years.max()) + 6)
    pred_line = logistic(years_line, aK, aR, aT0)
    df_fit = pd.DataFrame({
        "year": np.r_[years, years_line],
        "series": ["actual"] * len(years) + ["fit"] * len(years_line),
        "value": np.r_[y, pred_line],
    })
    fit_chart = (
        alt.Chart(df_fit)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:Q"),
            y=alt.Y("value:Q", title="EV stock"),
            color="series:N",
            tooltip=["year:Q", "series:N", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=300)
    )
    # draw t0 vertical rule
    rule = alt.Chart(pd.DataFrame({"t0":[aT0]})).mark_rule(strokeDash=[6,3]).encode(x="t0:Q")
    st.altair_chart(fit_chart + rule, use_container_width=True)

    # ---- ML baseline (RandomForest) ----
    with st.expander("🤖 Extra ML baseline (RandomForest)"):
        st.caption("Illustrative baseline: train a small RF on years → EV stock.")
        if len(sub) >= 5:
            X = sub[["year"]].astype(float)
            yv = sub["ev_stock"].astype(float)
            rf = RandomForestRegressor(n_estimators=300, random_state=0)
            rf.fit(X, yv)
            yhat = rf.predict(X)
            rmse_rf = float(np.sqrt(np.mean((yv.to_numpy() - yhat) ** 2)))
            df_rf = pd.DataFrame({"year": sub["year"], "actual": yv, "RF_pred": yhat}).set_index("year")
            st.line_chart(df_rf)
            st.caption(f"RandomForest in-sample RMSE = {rmse_rf:.1f} (for illustration only)")
        else:
            st.info("Need ≥ 5 points for the ML demo.")

with cR:
    st.subheader("Tunisia tech adoption fit (optional)")
    if dfTN is None or dfTN.empty:
        st.info("No Tunisia tech file loaded.")
        r_tn_from_tech = None
        t0_tn_from_tech = None
    else:
        techs = sorted(dfTN["tech"].dropna().unique().tolist())
        pick_tech = st.selectbox("Pick a tech", techs, index=0)
        subT = dfTN[dfTN["tech"] == pick_tech].sort_values("year").copy()

        # NEW: optional outlier filtering on tech adoption
        hide_out_T = st.checkbox("Hide outliers (z>3) in adoption", value=False, key="tntech_outliers")
        if hide_out_T and len(subT) >= 6:
            maskT = ~flag_outliers(subT["adoption_pct"], z=3.0)
            droppedT = int((~maskT).sum())
            if droppedT:
                st.info(f"Outliers hidden (tech): {droppedT} point(s).")
            subT = subT[maskT].copy()

        st.caption(f"{pick_tech} — {len(subT)} points")
        st.dataframe(subT.head(20), use_container_width=True)

        # Detect whether values exceed 100 (e.g., mobile subscriptions per 100 people)
        max_val = float(subT["adoption_pct"].max())
        dynamic_K_cap = 200.0 if max_val > 100 else 100.0

        # Guard against outliers: clip to the chosen cap (so optimizer remains stable)
        if max_val > dynamic_K_cap:
            st.warning(f"{pick_tech}: values exceed {dynamic_K_cap:.0f}. Clipping for stability.")
            subT["adoption_pct"] = subT["adoption_pct"].clip(upper=dynamic_K_cap)

        yT = subT["adoption_pct"].to_numpy()
        xT = subT["year"].to_numpy()

        # Fit logistic with K bounded to dynamic cap
        fitT = fit_logistic(
            xT, yT, K0=min(dynamic_K_cap, max(50.0, yT.max())),
            r0=0.30, t0=None,
            bounds=((max(30.0, yT.min()*0.8), 0.01, xT.min() - 10),
                    (dynamic_K_cap,          1.50, xT.max() + 20))
        )

        rmseT, mapeT = evaluate_fit(xT, yT, fitT.K, fitT.r, fitT.t0)
        st.success(
            f"{pick_tech} fit → K≈{fitT.K:.1f}, r={fitT.r:.3f}, t0={fitT.t0:.1f} "
            f"(RMSE={rmseT:.2f}, MAPE={mapeT:.1%})"
        )

        r_tn_from_tech = float(fitT.r)
        t0_tn_from_tech = float(fitT.t0)

        # Plot: actual points + fitted curve (tech adoption)
        years_line_T = np.arange(int(xT.min()) - 2, int(xT.max()) + 6)
        pred_line_T = logistic(years_line_T, fitT.K, fitT.r, fitT.t0)

        df_points_T = pd.DataFrame({
            "year": xT,
            "series": "actual",
            "value": yT
        })
        df_fit_T = pd.DataFrame({
            "year": years_line_T,
            "series": "fit",
            "value": pred_line_T
        })
        df_plot_T = pd.concat([df_points_T, df_fit_T], ignore_index=True)

        tech_chart = (
            alt.Chart(df_plot_T)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=45))
            .encode(
                x=alt.X("year:Q", title="Year"),
                y=alt.Y("value:Q", title=f"{pick_tech} adoption"),
                color=alt.Color("series:N", title="Series"),
                tooltip=["year:Q", "series:N", alt.Tooltip("value:Q", format=",.2f")]
            )
            .properties(height=300)
        )
        ruleT = alt.Chart(pd.DataFrame({"t0":[fitT.t0]})).mark_rule(strokeDash=[6,3]).encode(x="t0:Q")
        st.altair_chart(tech_chart + ruleT, use_container_width=True)

# --- 8) Tunisia scenario builder ---
st.write("---")
st.subheader("🇹🇳 Build Tunisia EV scenario")

seed_r, seed_t0 = float(aR), float(aT0)

if (dfTN is not None) and (r_tn_from_tech is not None) and (t0_tn_from_tech is not None):
    st.caption("Tip: use Tunisia tech (internet/mobile) r,t₀ to seed the fields below.")

cA, cB, cC, cD = st.columns(4)
with cA:
    K_tn = st.number_input("K (max EV stock, vehicles)", min_value=1_000, max_value=2_000_000,
                           value=150_000, step=1_000)
with cB:
    r_tn = st.number_input("r (growth rate)", min_value=0.01, max_value=1.50,
                           value=round(seed_r, 3), step=0.01)
with cC:
    t0_tn = st.number_input("t₀ (midpoint year)", min_value=2000, max_value=2050,
                            value=int(round(seed_t0)))
with cD:
    ratio = st.slider("EVs per charger", min_value=8, max_value=30, value=18, step=1)

if (dfTN is not None) and (r_tn_from_tech is not None) and (t0_tn_from_tech is not None):
    if st.button("Use Tunisia tech fit (r,t₀)"):
        r_tn = r_tn_from_tech
        t0_tn = t0_tn_from_tech
        st.success(f"Applied: r={r_tn:.3f}, t₀={t0_tn:.1f} (from TN {pick_tech})")

yearsF = np.arange(2022, 2041)
ev_tn = logistic(yearsF, K_tn, r_tn, t0_tn)
chargers_needed = (ev_tn / max(ratio, 1)).round(0)

# NEW: milestones for the Tunisia scenario
ms_tn = logistic_milestones(K_tn, r_tn, t0_tn, levels=(0.1, 0.5, 0.9))
st.caption(f"Tunisia milestones → 10% in {int(round(ms_tn[10]))}, 50% (t₀) ≈ {int(round(t0_tn))}, 90% in {int(round(ms_tn[90]))}.")

dfF = pd.DataFrame({
    "year": yearsF,
    "ev_stock_tn": ev_tn.astype(int),
    "chargers_needed": chargers_needed.astype(int)
})
st.line_chart(dfF.set_index("year"))

cDL1, cDL2 = st.columns(2)
with cDL1:
    st.download_button("Download Tunisia forecast (CSV)",
                       data=dfF.to_csv(index=False).encode(),
                       file_name="tn_ev_forecast.csv")
with cDL2:
    if st.button("Save forecast to disk"):
        outp = str(PATHS.tn_forecast_out)
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        dfF.to_csv(outp, index=False)
        st.success(f"Saved: {outp}")

# --- 8.1) Advanced uncertainty based on correlated (r, t0) from analogs ---
with st.expander("🎲 Advanced uncertainty (correlated r, t₀ from analogs)", expanded=False):
    st.caption("We derive the joint distribution of (r, t₀) from the selected analog (and optionally more), then simulate Tunisia EV stock paths.")
    # If you want to allow adding more analogs to the prior, switch default=[pick] to a multiselect.
    include_ctrys = [pick]
    # include_ctrys = st.multiselect("Use these analogs to form (r,t₀) prior", countries, default=[pick])

    fit_rows = []
    for ctry in include_ctrys:
        d = dfA[dfA["country"] == ctry].sort_values("year")
        if len(d) >= 4:
            fr = fit_logistic(d["year"].to_numpy(), d["ev_stock"].to_numpy())
            fit_rows.append([float(fr.r), float(fr.t0)])

    if fit_rows:
        mu, cov = analog_rt0_stats(fit_rows)  # mean/cov of (r, t0)
        sims = st.number_input("Simulations", min_value=200, max_value=10000, value=2000, step=200)
        samples = draw_correlated(mu, cov, n=int(sims), seed=42)

        years_sim = yearsF
        EV = []
        for r_i, t0_i in samples:
            EV.append(logistic(years_sim, K_tn, float(r_i), float(t0_i)))
        EV = np.asarray(EV)  # sims x years

        # summarize (P10/P50/P90)
        p10 = np.percentile(EV, 10, axis=0)
        p50 = np.percentile(EV, 50, axis=0)
        p90 = np.percentile(EV, 90, axis=0)
        df_band = pd.DataFrame({
            "year": years_sim,
            "P10": p10.astype(int),
            "P50": p50.astype(int),
            "P90": p90.astype(int)
        })
        st.line_chart(df_band.set_index("year"))
        st.caption("EV stock uncertainty bands (P10/P50/P90) using correlated draws of (r, t₀) from analog(s).")
    else:
        st.info("Not enough analog points to infer (r, t₀) distribution. Pick another analog with ≥4 points.")

# --- 9) Browse analog curves (gallery) ---
with st.expander("📚 Browse analog curves (quick look)", expanded=False):
    all_countries = countries  # already filtered to ≥2 points
    sel = st.multiselect("Choose countries to display", all_countries, default=[pick][:1])
    if sel:
        df_small = dfA[dfA["country"].isin(sel)].copy()
        pts = alt.Chart(df_small).mark_point(size=40, opacity=0.6).encode(
            x="year:Q", y="ev_stock:Q", color="country:N",
            tooltip=["country:N", "year:Q", alt.Tooltip("ev_stock:Q", format=",.0f")]
        )
        # quick rough fit line for each selected country
        fits = []
        for c in sel:
            d = df_small[df_small["country"] == c].sort_values("year")
            if len(d) >= 2:
                Kc, rc, t0c = rough_fit(d["year"].to_numpy(), d["ev_stock"].to_numpy())
                xs = np.arange(int(d["year"].min()) - 1, int(d["year"].max()) + 5)
                ys = logistic(xs, Kc, rc, t0c)
                fits.append(pd.DataFrame({"year": xs, "ev_stock": ys, "country": c, "series": "rough_fit"}))
        df_fits = (pd.concat(fits, ignore_index=True)
                   if fits else pd.DataFrame(columns=["year", "ev_stock", "country", "series"]))
        line = alt.Chart(df_fits).mark_line().encode(
            x="year:Q", y="ev_stock:Q", color="country:N"
        )
        st.altair_chart((pts + line).properties(height=280), use_container_width=True)
    else:
        st.info("Pick at least one country to show curves.")

# --- 10) Comparison: Tunisia vs Analog (dual-axis, right axis=chargers) ---
st.write("---")
st.subheader("📈 Comparison: Tunisia scenario vs selected analog")

mode = st.radio("Analog curve mode", ["Analog own K", "Scale analog speed to Tunisia K"], horizontal=True)
if mode == "Analog own K":
    analog_curve = logistic(yearsF, aK, aR, aT0)
    analog_label = f"{pick} (own K≈{int(aK):,})"
else:
    analog_curve = logistic(yearsF, K_tn, aR, aT0)
    analog_label = f"{pick} (scaled to Tunisia K={int(K_tn):,})"

df_compare = pd.DataFrame({
    "year": yearsF,
    "Tunisia EV stock (scenario)": ev_tn.astype(int),
    analog_label: analog_curve.astype(int),
    "Tunisia chargers (scenario)": chargers_needed.astype(int),
})

# EV stock (lines, LEFT axis)
df_ev = df_compare.melt(
    id_vars="year",
    value_vars=["Tunisia EV stock (scenario)", analog_label],
    var_name="series",
    value_name="ev_stock"
)
line_ev = (
    alt.Chart(df_ev)
      .mark_line(point=False)
      .encode(
          x=alt.X("year:Q", title="Year"),
          y=alt.Y("ev_stock:Q", title="EV stock (vehicles)", axis=alt.Axis(titleColor="#4e79a7")),
          color=alt.Color("series:N", title="EV stock series"),
          tooltip=[alt.Tooltip("year:Q"),
                   alt.Tooltip("series:N"),
                   alt.Tooltip("ev_stock:Q", format=",.0f")]
      )
)

# Chargers (bars, RIGHT axis)
df_ch = df_compare[["year", "Tunisia chargers (scenario)"]].rename(
    columns={"Tunisia chargers (scenario)": "chargers"}
)
bar_ch = (
    alt.Chart(df_ch)
      .mark_bar(opacity=0.35)
      .encode(
          x="year:Q",
          y=alt.Y("chargers:Q", title="Chargers (units)", axis=alt.Axis(titleColor="#f28e2b", orient="right")),
          tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("chargers:Q", format=",.0f")]
      )
)

comp = alt.layer(line_ev, bar_ch).resolve_scale(y="independent").properties(
    width="container", height=380
)
st.altair_chart(comp, use_container_width=True)
st.caption("Lines (left axis): EV stock • Bars (right axis): required chargers — note the two axes.")

st.download_button(
    "Download comparison (CSV)",
    data=df_compare.to_csv(index=False).encode(),
    file_name="tn_vs_analog_comparison.csv"
)
