# pages/3_Modeling_Lab.py — Comparative Modeling Lab (Logistic vs Prophet vs XGBoost)
# Focus: transparent modeling, rolling-origin CV, clear explanations & visuals.

# --- 0) Path bootstrap ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- 1) Imports ---
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Optional deps (handled gracefully)
try:
    from prophet import Prophet  # pip install prophet (or prophet==1.1.5)
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

try:
    from xgboost import XGBRegressor  # pip install xgboost
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# Our logistic helper
from models.logistic_bass import logistic, fit_logistic, evaluate_fit

# Config paths
try:
    from config import PATHS
except Exception:
    class _Paths:
        analog_csv = ROOT / "data" / "analog_ev_full.csv"
    PATHS = _Paths()

# --- 2) Page meta ---
st.set_page_config(page_title="Modeling Lab • Logistic vs Prophet vs XGBoost", layout="wide")
st.page_link("pages/6_Chatbot.py", label="💬 Ask the Assistant")
st.title("🧪 Modeling Lab — Logistic vs Prophet vs XGBoost")
st.caption("Transparent, side-by-side comparison with rolling-origin cross-validation, clear metrics, and explainers.")

# --- 3) Small helpers ---
@st.cache_data(show_spinner=False)
def read_csv_cached(path_or_file):
    return pd.read_csv(path_or_file)

def clean_analogs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize headers
    ren = {
        "Country": "country", "ISO3":"iso3", "EV stock":"ev_stock",
        "Public charging points":"public_chargers", "Year":"year"
    }
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    df.columns = [c.strip().lower() for c in df.columns]
    # numerics
    for c in ["year", "ev_stock", "public_chargers"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop bad rows
    need = [c for c in ["country","year","ev_stock"] if c in df.columns]
    if need:
        df = df.dropna(subset=need)
    return df.reset_index(drop=True)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    if not np.any(mask): 
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def add_time_features(df: pd.DataFrame, target_col: str, max_lag: int = 3) -> pd.DataFrame:
    """Create lag features + rolling means to help tree models."""
    out = df.copy()
    out = out.sort_values("year").copy()
    for L in range(1, max_lag+1):
        out[f"{target_col}_lag{L}"] = out[target_col].shift(L)
    out[f"{target_col}_roll3"] = out[target_col].rolling(3).mean()
    out[f"{target_col}_roll5"] = out[target_col].rolling(5).mean()
    # time trend & polynomial
    out["t"] = np.arange(len(out), dtype=float)
    out["t2"] = out["t"]**2
    return out

def time_series_splits(n, n_splits=4, min_train=5, step=1, test_size=1):
    """
    Generator of (train_idx, test_idx) for rolling-origin splits.
    - min_train: minimum train size (years)
    - test_size: number of points to test at each split (usually 1)
    """
    start = min_train
    while start + test_size <= n and n_splits > 0:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, start + test_size)
        yield train_idx, test_idx
        start += step
        n_splits -= 1

# --- 4) Inputs ---
st.write("---")
with st.expander("📥 Data input", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("Upload analogs CSV (country, year, ev_stock[, public_chargers])", type=["csv"])
        if up is not None:
            df = read_csv_cached(up)
            src = f"Uploaded: {getattr(up, 'name','')}"
        elif os.path.exists(PATHS.analog_csv):
            df = read_csv_cached(PATHS.analog_csv)
            src = f"Default: {PATHS.analog_csv}"
        else:
            df = None
            src = "No file"
        if df is not None:
            st.caption(src)
    with c2:
        st.markdown("**What this page does**")
        st.markdown("""
- Choose a **country** and a **target series** (e.g., EV stock).
- We run **rolling-origin cross-validation**:
  - Split the time series into multiple expanding **train windows** and 1-step **validation** points.
  - Train each model on the train window, **forecast the next year**, and collect errors.
- Compare models on **RMSE/MAE/MAPE/R²** and visualize predictions vs actuals.
- For XGBoost, we build simple **lag & trend features** and show **feature importance**.
        """)

if df is None or df.empty:
    st.error("No dataset found. Upload a CSV or place one at PATHS.analog_csv.")
    st.stop()

df = clean_analogs(df)
countries = sorted(df["country"].unique().tolist())
if not countries:
    st.error("No country values found after cleaning.")
    st.stop()

st.subheader("🎯 Experiment setup")
colA, colB, colC, colD = st.columns(4)
with colA:
    country = st.selectbox("Country", countries, index=0)
with colB:
    target_col = st.selectbox("Target series", options=[c for c in ["ev_stock","public_chargers"] if c in df.columns], index=0)
with colC:
    n_splits = st.number_input("CV splits", min_value=2, max_value=10, value=5, step=1, help="How many rolling splits.")
with colD:
    min_train = st.number_input("Min train size (years)", min_value=3, max_value=20, value=5, step=1)

test_size = 1  # predict next year
step = 1       # roll by one year

sub = df[df["country"] == country].sort_values("year").reset_index(drop=True)
if len(sub) < (min_train + test_size + 1):
    st.warning("Not enough points for the chosen min_train and splits. Consider reducing min_train or splits.")
st.dataframe(sub.head(20), use_container_width=True)

# --- 5) Models config toggles ---
st.write("---")
st.subheader("⚙️ Models to compare")
c1, c2, c3 = st.columns(3)
use_logistic = c1.checkbox("Logistic (S-curve)", value=True,
                           help="Parametric adoption curve: K, r, t₀.")
use_prophet = c2.checkbox("Prophet", value=HAVE_PROPHET,
                          help="Additive trend + seasonality + holidays (if installed).")
use_xgb     = c3.checkbox("XGBoost", value=HAVE_XGB,
                          help="Gradient-boosted trees with lag/trend features (if installed).")

if use_prophet and not HAVE_PROPHET:
    st.info("Prophet not installed. Try:  pip install prophet")
if use_xgb and not HAVE_XGB:
    st.info("XGBoost not installed. Try:  pip install xgboost")

# --- 6) Rolling-origin CV runner ---
st.write("---")
st.subheader("🚦 Rolling-origin cross-validation")
st.caption("At each split: train up to year *t*, forecast year *t+1*. We collect out-of-sample predictions to compute unbiased errors.")

years_all = sub["year"].to_numpy()
y_all = sub[target_col].to_numpy()

records = []  # to aggregate oos predictions for each model
pred_store_plot = []  # for visualization (per split)

# helper: safe logistic fit on training window, predict next year
def _predict_logistic(train_years, train_y, next_year):
    if len(train_years) >= 4:
        fit = fit_logistic(train_years, train_y, K0=None, r0=0.30, t0=None)
        K, r, t0 = float(fit.K), float(fit.r), float(fit.t0)
    else:
        # rough
        # Reuse logistic’s rough behavior: fallback by hacking K/r/t0 from data
        K = max(float(np.nanmax(train_y)) * 1.5, 10.0)
        y_pos = np.maximum(train_y, 1e-6)
        t_center = train_years - train_years.mean()
        b, a = np.polyfit(t_center, np.log(y_pos), 1)
        r = float(np.clip(b, 0.05, 1.0))
        t0 = float(np.median(train_years))
    return float(logistic([next_year], K, r, t0)[0])

def _predict_prophet(train_df, next_year):
    # Prophet expects df with ds, y
    m = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
    dfp = train_df.rename(columns={"year":"ds", target_col:"y"}).copy()
    # cast ds to datetime (Jan 1 of that year)
    dfp["ds"] = pd.to_datetime(dfp["ds"].astype(int), format="%Y")
    m.fit(dfp[["ds","y"]])
    future = pd.DataFrame({"ds":[pd.to_datetime(int(next_year), format="%Y")]})
    yhat = float(m.predict(future)["yhat"].iloc[0])
    return yhat

def _predict_xgb(train_df, next_year):
    # Build features on TRAIN only; then build a single-row feature for the next year.
    tmp = train_df[["year", target_col]].copy()
    tmp = add_time_features(tmp, target_col=target_col, max_lag=3).dropna().reset_index(drop=True)
    if tmp.empty:
        return np.nan
    # split X,y from tmp (last row is the most recent train observation)
    feat_cols = [c for c in tmp.columns if c not in ["year", target_col]]
    X = tmp[feat_cols].values
    y = tmp[target_col].values
    model = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42
    )
    model.fit(X, y)
    # Build feature row for next_year:
    #  - Create a mini frame combining train + a placeholder next row to compute lags/rolls correctly.
    last = train_df.copy()
    next_row = pd.DataFrame({"year":[next_year], target_col:[np.nan]})
    ext = pd.concat([last, next_row], ignore_index=True)
    ext = add_time_features(ext, target_col=target_col, max_lag=3)
    row = ext[ext["year"] == next_year]
    row = row[feat_cols].tail(1)
    if row.empty or row.isna().any(axis=None):
        # if we can't compute lags (too short series), fallback to a simple trend
        return float(last[target_col].iloc[-1])
    pred = float(model.predict(row.values)[0])
    return pred, model, feat_cols

# Run splits
for tr_idx, te_idx in time_series_splits(len(sub), n_splits=int(n_splits), min_train=int(min_train), step=int(step), test_size=int(test_size)):
    train_years = years_all[tr_idx]
    train_y = y_all[tr_idx]
    test_years = years_all[te_idx]  # 1 step
    test_y = y_all[te_idx]
    assert len(test_years) == 1
    ny = int(test_years[0])  # next year

    # store predictions per model
    row_vis = {"year": ny, "actual": float(test_y[0])}

    # Logistic
    if use_logistic:
        yhat_l = _predict_logistic(train_years, train_y, ny)
        records.append({"model":"Logistic", "year":ny, "y_true": float(test_y[0]), "y_pred": yhat_l})
        row_vis["Logistic"] = yhat_l

    # Prophet
    if use_prophet and HAVE_PROPHET:
        train_df = sub.loc[tr_idx, ["year", target_col]].copy()
        try:
            yhat_p = _predict_prophet(train_df, ny)
        except Exception:
            yhat_p = np.nan
        records.append({"model":"Prophet", "year":ny, "y_true": float(test_y[0]), "y_pred": yhat_p})
        row_vis["Prophet"] = yhat_p

    # XGBoost
    if use_xgb and HAVE_XGB:
        train_df = sub.loc[tr_idx, ["year", target_col]].copy()
        try:
            res = _predict_xgb(train_df, ny)
            if isinstance(res, tuple):
                yhat_x, fitted_xgb, feat_cols = res
                # store feature importance from the last split (display below)
                last_feat_info = (fitted_xgb, feat_cols)
            else:
                yhat_x = res
        except Exception:
            yhat_x = np.nan
        records.append({"model":"XGBoost", "year":ny, "y_true": float(test_y[0]), "y_pred": float(yhat_x)})
        row_vis["XGBoost"] = yhat_x

    pred_store_plot.append(row_vis)

# Aggregate metrics
if not records:
    st.warning("No model was selected or predictions failed. Adjust choices and try again.")
    st.stop()

pred_df = pd.DataFrame(records)
metrics = []
for model in pred_df["model"].unique():
    d = pred_df[pred_df["model"] == model].sort_values("year")
    r = {
        "model": model,
        "n_points": len(d),
        "RMSE": rmse(d["y_true"], d["y_pred"]),
        "MAE":  mae(d["y_true"], d["y_pred"]),
        "MAPE": mape(d["y_true"], d["y_pred"]),
        "R2":   r2(d["y_true"], d["y_pred"]),
    }
    metrics.append(r)
metric_df = pd.DataFrame(metrics).sort_values("RMSE")

st.write("### 📊 Cross-validation summary (out-of-sample)")
st.dataframe(metric_df, use_container_width=True)

# --- 7) Visuals: per-split forecast accuracy ---
st.write("### 📈 OOS predictions vs actual (each CV split)")
plot_df = pd.DataFrame(pred_store_plot).sort_values("year")
plot_melt = plot_df.melt(id_vars=["year","actual"], var_name="model", value_name="yhat")
plot_melt = plot_melt[plot_melt["model"].isin(["Logistic","Prophet","XGBoost"])]

line_true = alt.Chart(plot_df).mark_line(point=True, color="#444").encode(
    x=alt.X("year:Q"), y=alt.Y("actual:Q", title=target_col.replace("_"," ").title()),
    tooltip=[alt.Tooltip("year:Q"), alt.Tooltip("actual:Q", format=",.0f")]
).properties(height=320)

line_models = alt.Chart(plot_melt).mark_line(point=True).encode(
    x=alt.X("year:Q", title="Validation year (t+1)"),
    y=alt.Y("yhat:Q", title=target_col.replace("_"," ").title()),
    color=alt.Color("model:N", title="Model"),
    tooltip=[alt.Tooltip("year:Q"), "model:N", alt.Tooltip("yhat:Q", format=",.0f")]
)
st.altair_chart(line_true + line_models, use_container_width=True)

# --- 8) XGBoost feature importance (if available) ---
if use_xgb and HAVE_XGB and 'last_feat_info' in locals() and last_feat_info:
    model_xgb, feat_cols = last_feat_info
    try:
        imp = model_xgb.get_booster().get_score(importance_type="gain")
        # align all features
        rows = []
        for f in feat_cols:
            rows.append({"feature": f, "gain": float(imp.get(f, 0.0))})
        imp_df = pd.DataFrame(rows).sort_values("gain", ascending=False)
        st.write("### 🌾 XGBoost — feature importance (gain)")
        st.dataframe(imp_df, use_container_width=True, height=280)
        imp_chart = alt.Chart(imp_df.head(12)).mark_bar().encode(
            x=alt.X("gain:Q", title="Importance (gain)"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("gain:Q", format=",.3f")]
        ).properties(height=280)
        st.altair_chart(imp_chart, use_container_width=True)
    except Exception:
        pass

# --- 9) Holistic final chart (fit on all history, forecast next N years) ---
st.write("---")
st.subheader("🔮 Final forecast (fit on all history)")
h = st.slider("Forecast horizon (years ahead)", 1, 10, 5)

# Build a common years range for plotting
start_y = int(sub["year"].min())
last_y = int(sub["year"].max())
future_years = np.arange(last_y + 1, last_y + h + 1, dtype=int)

final_rows = []
# Logistic final
if use_logistic:
    if len(sub) >= 4:
        fit_all = fit_logistic(sub["year"].to_numpy(), sub[target_col].to_numpy(), K0=None, r0=0.30, t0=None)
        Kf, rf, t0f = float(fit_all.K), float(fit_all.r), float(fit_all.t0)
    else:
        # rough fallback: (safer default)
        Kf = max(float(sub[target_col].max()) * 1.5, 10.0)
        y_pos = np.maximum(sub[target_col].to_numpy(), 1e-6)
        t_center = sub["year"].to_numpy() - sub["year"].to_numpy().mean()
        b, a = np.polyfit(t_center, np.log(y_pos), 1)
        rf = float(np.clip(b, 0.05, 1.0))
        t0f = float(np.median(sub["year"]))
    fitted_hist = logistic(sub["year"], Kf, rf, t0f)
    fitted_fut = logistic(future_years, Kf, rf, t0f)
    final_rows += [{"model":"Logistic","year":int(y),"value":float(v)} for y,v in zip(sub["year"], fitted_hist)]
    final_rows += [{"model":"Logistic","year":int(y),"value":float(v)} for y,v in zip(future_years, fitted_fut)]

# Prophet final
if use_prophet and HAVE_PROPHET:
    try:
        mp = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
        dfp = sub[["year", target_col]].rename(columns={"year":"ds", target_col:"y"}).copy()
        dfp["ds"] = pd.to_datetime(dfp["ds"].astype(int), format="%Y")
        mp.fit(dfp)
        fdf = pd.DataFrame({"ds": pd.to_datetime(np.r_[sub["year"].to_numpy(), future_years], format="%Y")})
        yhat_all = mp.predict(fdf)[["ds","yhat"]]
        # join back years
        yhat_all["year"] = yhat_all["ds"].dt.year
        final_rows += [{"model":"Prophet","year":int(r.year),"value":float(r.yhat)} for r in yhat_all.itertuples(index=False)]
    except Exception:
        pass

# XGBoost final
if use_xgb and HAVE_XGB:
    try:
        tmp_all = sub[["year", target_col]].copy()
        tmp_all = add_time_features(tmp_all, target_col=target_col, max_lag=3).dropna().reset_index(drop=True)
        if not tmp_all.empty:
            feat_cols = [c for c in tmp_all.columns if c not in ["year", target_col]]
            X = tmp_all[feat_cols].values
            y_tr = tmp_all[target_col].values
            m = XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.85, colsample_bytree=0.9,
                reg_lambda=1.0, random_state=7
            )
            m.fit(X, y_tr)
            # Predict for all years (history + future) by constructing features iteratively:
            hist_pred = m.predict(X)
            # For future, we’ll iterate, appending predictions to enable lag features
            ext = sub[["year", target_col]].copy()
            for y in future_years:
                # add placeholder -> generate features -> predict
                ext = pd.concat([ext, pd.DataFrame({"year":[y], target_col:[np.nan]})], ignore_index=True)
                ext_feat = add_time_features(ext, target_col=target_col, max_lag=3)
                row = ext_feat[ext_feat["year"] == y]
                rowX = row[feat_cols].tail(1)
                if rowX.empty or rowX.isna().any(axis=None):
                    # fallback to last known value
                    pred = float(ext[target_col].dropna().iloc[-1])
                else:
                    pred = float(m.predict(rowX.values)[0])
                ext.loc[ext["year"] == y, target_col] = pred
            # collect full series
            xgb_full = ext.rename(columns={target_col:"value"})
            xgb_full["model"] = "XGBoost"
            final_rows += xgb_full.to_dict("records")
    except Exception:
        pass

final_df = pd.DataFrame(final_rows)
if not final_df.empty:
    # Actuals
    act = sub[["year", target_col]].rename(columns={target_col:"value"})
    act["model"] = "Actual"
    show_df = pd.concat([act, final_df], ignore_index=True)

    chart = (
        alt.Chart(show_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("value:Q", title=target_col.replace("_"," ").title()),
            color=alt.Color("model:N", title="Series"),
            tooltip=[alt.Tooltip("year:Q"), "model:N", alt.Tooltip("value:Q", format=",.0f")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "⬇️ Download final forecast (CSV)",
        data=show_df.sort_values(["model","year"]).to_csv(index=False).encode(),
        file_name=f"{country.lower().replace(' ','_')}_{target_col}_final_forecast.csv",
        use_container_width=True
    )

# --- 10) Explain like I’m your supervisor (mini guide) ---
with st.expander("ℹ️ How to read these results (for supervisors)"):
    st.markdown("""
**Why rolling-origin CV?**  
Time-series models must be tested **forward in time**. We simulate a real deployment:
train on early years, predict the **next** year, move the window forward, repeat.  
This yields **out-of-sample** errors (RMSE/MAE/MAPE/R²) you can trust.

**Models:**
- **Logistic** (K, r, t₀): classic S-curve for technology diffusion. Great when adoption saturates.
- **Prophet**: additive model (trend + optional seasonality/holidays). Good general forecaster.
- **XGBoost**: tree ensembles on engineered features (lags, rolling means, trend). Flexible, can overfit if careless.

**How to compare:**
- **MAPE** rewards relative accuracy; **RMSE/MAE** reward absolute accuracy.
- **R²** shows variance explained (higher is better).
- Prefer models with consistently **lower out-of-sample** errors, not in-sample fit.

**Practical reading:**
- If **Logistic** wins, adoption likely follows an S-curve. Use its parameters to drive **EV→charger** planning.
- If **XGBoost** wins, short-term patterns/lag effects dominate. Keep monitoring frequently.
- **Prophet** doing well suggests smoother trend-driven dynamics.

**Next steps:**
- Blend models (e.g., average Logistic & XGBoost).
- Add exogenous drivers (GDP, prices, policy dummies) to XGBoost/Prophet.
- Stress-test with Monte Carlo over **K, r, t₀** (you already have this in the Risk tab).
    """)

with st.expander("🗂 Data & features used"):
    st.markdown(f"""
- **Dataset source**: {PATHS.analog_csv}
- **Target**: `{target_col}`
- **Country**: `{country}`
- **XGBoost features**:
  - Lags: `{target_col}_lag1`, `{target_col}_lag2`, `{target_col}_lag3`
  - Rolling means: `{target_col}_roll3`, `{target_col}_roll5`
  - Trend terms: `t`, `t2`
- **Prophet**:
  - Uses year as a date (Jan 1st), no seasonality by default (yearly/daily/weekly disabled here).
    """)

# A link back to Analogs page & ROI pages for the story flow
st.page_link("pages/2_Analogs_and_ML.py", label="⬅️ Back to Analogs & ML", icon="📊")
st.page_link("pages/4_ROI_Sensitivity.py", label="Next → ROI & Sensitivity", icon="💸")
st.page_link("pages/5_Deployment_ROI.py", label="Next → Deployment ROI", icon="🧭")
