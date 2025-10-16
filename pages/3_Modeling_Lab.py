# pages/3_Modeling_Lab.py — Comparative Modeling Lab (clear UI + explanations)
# Rolling-origin CV for Logistic + lightweight ML (RF/ET/GBR/SVR/MLP/KNN),
# with optional Prophet and XGBoost if installed.

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
    from prophet import Prophet  # pip install prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

try:
    from xgboost import XGBRegressor  # pip install xgboost
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# Logistic helper (your existing module)
from models.logistic_bass import logistic, fit_logistic

# Config paths
try:
    from config import PATHS
except Exception:
    class _Paths:
        analog_csv = ROOT / "data" / "analog_ev_full.csv"
    PATHS = _Paths()

# Scikit-learn models (all lightweight)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ------------------------------------------------------------
# PAGE META + SIDEBAR (guided steps)
# ------------------------------------------------------------
st.set_page_config(page_title="Modeling Lab • Forecasting Benchmarks", layout="wide")

st.sidebar.title("🧭 Guided steps")
st.sidebar.markdown("""
1. **Load data** (or use default)  
2. **Pick a country & target series**  
3. **Select models** to compare  
4. **Run cross-validation** (see metrics)  
5. **Fit on all history & forecast**  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")
_status = {"data": "❌", "country": "❌", "models": "❌", "cv": "⏳", "forecast": "⏳"}

st.page_link("pages/6_Chatbot.py", label="💬 Ask the Assistant")
st.title("🧪 Modeling Lab — Transparent Benchmarks")
st.caption(
    "Compare **Logistic S-curve** and **lightweight ML** (RandomForest, ExtraTrees, GradientBoosting, "
    "SVR, MLP, KNN). *Prophet* and *XGBoost* are available if installed. "
    "Evaluation uses **rolling-origin cross-validation** (train up to year *t*, predict *t+1*)."
)

# ------------------------------------------------------------
# Small helpers (metrics + features + CV)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv_cached(path_or_file):
    return pd.read_csv(path_or_file)

def clean_analogs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {
        "Country": "country", "ISO3":"iso3", "EV stock":"ev_stock",
        "Public charging points":"public_chargers", "Year":"year"
    }
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["year", "ev_stock", "public_chargers"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    need = [c for c in ["country","year","ev_stock"] if c in df.columns]
    if need:
        df = df.dropna(subset=need)
    return df.reset_index(drop=True)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    if not np.any(mask): return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def add_time_features(df: pd.DataFrame, target_col: str, max_lag: int = 3) -> pd.DataFrame:
    """Lag features + rolling means + trend terms for ML models."""
    out = df.copy().sort_values("year")
    for L in range(1, max_lag+1):
        out[f"{target_col}_lag{L}"] = out[target_col].shift(L)
    out[f"{target_col}_roll3"] = out[target_col].rolling(3).mean()
    out[f"{target_col}_roll5"] = out[target_col].rolling(5).mean()
    out["t"] = np.arange(len(out), dtype=float)
    out["t2"] = out["t"]**2
    return out

def time_series_splits(n, n_splits=4, min_train=5, step=1, test_size=1):
    """Yield expanding-window train and 1-step test indices."""
    start, done = min_train, 0
    while start + test_size <= n and done < n_splits:
        yield np.arange(0, start), np.arange(start, start + test_size)
        start += step
        done += 1

def build_supervised_matrix(train_df: pd.DataFrame, target_col: str, max_lag: int = 3):
    """Return (X, y, feat_cols, row_next) for ML models; row_next is next-year feature row."""
    tmp = add_time_features(train_df[["year", target_col]].copy(), target_col=target_col, max_lag=max_lag)
    tmp = tmp.dropna().reset_index(drop=True)
    if tmp.empty: return None, None, None, None
    feat_cols = [c for c in tmp.columns if c not in ["year", target_col]]
    X = tmp[feat_cols].values
    y = tmp[target_col].values
    last_year = int(train_df["year"].iloc[-1])
    nx = last_year + 1
    ext = pd.concat([train_df[["year", target_col]], pd.DataFrame({"year":[nx], target_col:[np.nan]})], ignore_index=True)
    ext = add_time_features(ext, target_col=target_col, max_lag=max_lag)
    row_next = ext[ext["year"] == nx][feat_cols].tail(1)
    if row_next.empty or row_next.isna().any(axis=None):
        return X, y, feat_cols, None
    return X, y, feat_cols, row_next.values

def predict_polynomial(train_years, train_y, next_year, degree=2):
    """Polynomial trend on time (degree 1=linear, 2=quadratic)."""
    t = np.asarray(train_years, float).reshape(-1, 1)
    X = np.hstack([t] + [t**d for d in range(2, degree+1)]) if degree > 1 else t
    lr = LinearRegression().fit(X, train_y)
    tn = np.array([[next_year]], float)
    Xn = np.hstack([tn] + [tn**d for d in range(2, degree+1)]) if degree > 1 else tn
    return float(lr.predict(Xn)[0])

def predict_treeish(model, train_df, target_col, next_year):
    X, y, feat_cols, row_next = build_supervised_matrix(train_df, target_col, max_lag=3)
    if X is None or row_next is None: return np.nan
    model.fit(X, y)
    return float(model.predict(row_next)[0])

def predict_pipeline(pipeline, train_df, target_col, next_year):
    X, y, feat_cols, row_next = build_supervised_matrix(train_df, target_col, max_lag=3)
    if X is None or row_next is None: return np.nan
    pipeline.fit(X, y)
    return float(pipeline.predict(row_next)[0])

# ------------------------------------------------------------
# TABS LAYOUT
# ------------------------------------------------------------
tab_setup, tab_cv, tab_forecast, tab_notes = st.tabs(
    ["📥 Setup", "🧪 Cross-Validation", "🔮 Final Forecast", "🗒️ Model Notes"]
)

# =========================
# 📥 SETUP
# =========================
with tab_setup:
    st.subheader("1) Load data & choose your series")
    c1, c2 = st.columns([1.2, 1])

    with c1:
        up = st.file_uploader("Upload analogs CSV (country, year, ev_stock[, public_chargers])", type=["csv"])
        if up is not None:
            df = read_csv_cached(up); src = f"Uploaded: {getattr(up, 'name','')}"
        elif os.path.exists(PATHS.analog_csv):
            df = read_csv_cached(PATHS.analog_csv); src = f"Default: {PATHS.analog_csv}"
        else:
            df = None; src = "No file found"
        if df is not None:
            st.caption(src)

    if df is None or df.empty:
        st.error("No dataset available. Upload a CSV or place one at PATHS.analog_csv.")
        st.stop()

    df = clean_analogs(df)
    countries = sorted(df["country"].unique().tolist())
    if not countries:
        st.error("No country values found after cleaning.")
        st.stop()

    with c2:
        st.markdown("**What this page will do**")
        st.info(
            "- Run **rolling-origin cross-validation** (train up to *t*, predict *t+1*)\n"
            "- Compare models on **RMSE, MAE, MAPE, R²**\n"
            "- Fit on all history and **forecast N years** ahead\n"
            "- Keep it **explainable** with clear visuals and notes"
        )

    st.markdown("---")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        country = st.selectbox("Country", countries, index=0, help="Pick one time series to evaluate.")
    with colB:
        target_col = st.selectbox(
            "Target series",
            options=[c for c in ["ev_stock","public_chargers"] if c in df.columns],
            index=0,
            help="What are we forecasting?"
        )
    with colC:
        n_splits = st.number_input("CV splits", 2, 10, 5, 1, help="How many expanding-window splits?")
    with colD:
        min_train = st.number_input("Min train size (years)", 3, 20, 5, 1, help="Small values = more splits but weaker models.")

    sub = df[df["country"] == country].sort_values("year").reset_index(drop=True)
    st.caption("**Data preview** (first rows)")
    st.dataframe(sub.head(15), use_container_width=True, height=260)

    # Sidebar status
    _status["data"] = "✅"
    _status["country"] = "✅"

    st.success("Setup complete. Move to **Cross-Validation** tab to benchmark models →")

# =========================
# 🧪 CROSS-VALIDATION
# =========================
with tab_cv:
    st.subheader("2) Select models & run rolling-origin CV")
    st.caption(
        "At each split, we train up to year *t* and forecast **t+1**. "
        "This simulates production usage and avoids look-ahead bias."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Core models**")
        col1, col2, col3 = st.columns(3)
        use_logistic = col1.checkbox("Logistic (S-curve)", value=True)
        use_poly2    = col2.checkbox("Polynomial trend (deg=2)", value=True)
        use_rf       = col3.checkbox("Random Forest", value=True)
        col4, col5 = st.columns(2)
        use_et       = col4.checkbox("Extra Trees", value=False)
        use_gbr      = col5.checkbox("Gradient Boosting", value=False)

    with c2:
        st.markdown("**Advanced ML**")
        col6, col7, col8 = st.columns(3)
        use_svr_rbf  = col6.checkbox("SVR (RBF)", value=False)
        use_mlp      = col7.checkbox("MLP (Neural Net)", value=False)
        use_knn      = col8.checkbox("KNN", value=False)
        st.markdown("**Optional (if installed)**")
        col9, col10 = st.columns(2)
        use_prophet  = col9.checkbox("Prophet", value=HAVE_PROPHET, disabled=not HAVE_PROPHET)
        use_xgb      = col10.checkbox("XGBoost", value=HAVE_XGB, disabled=not HAVE_XGB)
        if use_prophet and not HAVE_PROPHET:
            st.info("Prophet not installed. Try:  pip install prophet")
        if use_xgb and not HAVE_XGB:
            st.info("XGBoost not installed. Try:  pip install xgboost")

    # Hyperparameters (kept compact)
    st.markdown("---")
    cH1, cH2, cH3, cH4 = st.columns(4)
    poly_degree  = cH1.slider("Polynomial degree", 1, 3, 2)
    rf_trees     = cH2.slider("RF/ET: #trees", 50, 800, 300, 50)
    gbr_trees    = cH3.slider("GBR: #trees", 50, 800, 250, 50)
    knn_k        = cH4.slider("KNN: neighbors (k)", 1, 10, 3)

    years_all = sub["year"].to_numpy()
    y_all = sub[target_col].to_numpy()

    # Status
    any_model = any([use_logistic, use_poly2, use_rf, use_et, use_gbr, use_svr_rbf, use_mlp, use_knn, use_prophet, use_xgb])
    _status["models"] = "✅" if any_model else "❌"
    st.sidebar.markdown(f"- Models: **{_status['models']}**")

    # Prediction stores
    records = []          # aggregated OOS predictions
    pred_store_plot = []  # for chart
    last_feat_info = None # (xgb_model, feat_cols) if available

    # Model-specific predictors
    def _predict_logistic(train_years, train_y, next_year):
        if len(train_years) >= 4:
            fit = fit_logistic(train_years, train_y, K0=None, r0=0.30, t0=None)
            K, r, t0 = float(fit.K), float(fit.r), float(fit.t0)
        else:
            K = max(float(np.nanmax(train_y)) * 1.5, 10.0)
            y_pos = np.maximum(train_y, 1e-6)
            t_center = train_years - train_years.mean()
            b, _ = np.polyfit(t_center, np.log(y_pos), 1)
            r = float(np.clip(b, 0.05, 1.0)); t0 = float(np.median(train_years))
        return float(logistic([next_year], K, r, t0)[0])

    def _predict_prophet(train_df, next_year):
        m = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
        dfp = train_df.rename(columns={"year":"ds", target_col:"y"}).copy()
        dfp["ds"] = pd.to_datetime(dfp["ds"].astype(int), format="%Y")
        m.fit(dfp[["ds","y"]])
        future = pd.DataFrame({"ds":[pd.to_datetime(int(next_year), format="%Y")]})
        return float(m.predict(future)["yhat"].iloc[0])

    def _predict_xgb(train_df, next_year):
        tmp = train_df[["year", target_col]].copy()
        tmp = add_time_features(tmp, target_col=target_col, max_lag=3).dropna().reset_index(drop=True)
        if tmp.empty: return np.nan
        feat_cols = [c for c in tmp.columns if c not in ["year", target_col]]
        X = tmp[feat_cols].values; y = tmp[target_col].values
        model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0, random_state=42
        )
        model.fit(X, y)
        last = train_df.copy()
        next_row = pd.DataFrame({"year":[next_year], target_col:[np.nan]})
        ext = pd.concat([last, next_row], ignore_index=True)
        ext = add_time_features(ext, target_col=target_col, max_lag=3)
        row = ext[ext["year"] == next_year][feat_cols].tail(1)
        if row.empty or row.isna().any(axis=None):
            return float(last[target_col].iloc[-1])
        pred = float(model.predict(row.values)[0])
        return (pred, model, feat_cols)

    # Rolling-origin CV
    for tr_idx, te_idx in time_series_splits(
            len(sub), n_splits=int(n_splits), min_train=int(min_train), step=1, test_size=1
        ):
        train_years = years_all[tr_idx]
        train_y = y_all[tr_idx]
        test_year = int(years_all[te_idx][0])
        test_val = float(y_all[te_idx][0])
        train_df = sub.loc[tr_idx, ["year", target_col]].copy()
        row_vis = {"year": test_year, "actual": test_val}

        # Logistic
        if use_logistic:
            yhat = _predict_logistic(train_years, train_y, test_year)
            records.append({"model":"Logistic", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["Logistic"] = yhat

        # Polynomial
        if use_poly2:
            try: yhat = predict_polynomial(train_years, train_y, test_year, degree=int(poly_degree))
            except Exception: yhat = np.nan
            records.append({"model":"Polynomial", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["Polynomial"] = yhat

        # RF / ET / GBR
        if use_rf:
            rf = RandomForestRegressor(n_estimators=int(rf_trees), random_state=42)
            yhat = predict_treeish(rf, train_df, target_col, test_year)
            records.append({"model":"RandomForest", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["RandomForest"] = yhat

        if use_et:
            et = ExtraTreesRegressor(n_estimators=int(rf_trees), random_state=42)
            yhat = predict_treeish(et, train_df, target_col, test_year)
            records.append({"model":"ExtraTrees", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["ExtraTrees"] = yhat

        if use_gbr:
            gbr = GradientBoostingRegressor(n_estimators=int(gbr_trees), learning_rate=0.05, max_depth=2, random_state=42)
            yhat = predict_treeish(gbr, train_df, target_col, test_year)
            records.append({"model":"GradBoost", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["GradBoost"] = yhat

        # SVR / MLP / KNN
        if use_svr_rbf:
            svr = make_pipeline(StandardScaler(with_mean=False), SVR(kernel="rbf", C=5.0, epsilon=0.2, gamma="scale"))
            yhat = predict_pipeline(svr, train_df, target_col, test_year)
            records.append({"model":"SVR(RBF)", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["SVR(RBF)"] = yhat

        if use_mlp:
            mlp = make_pipeline(
                StandardScaler(with_mean=False),
                MLPRegressor(hidden_layer_sizes=(32,16), activation="relu",
                             alpha=1e-3, learning_rate_init=0.01, max_iter=1000, random_state=42)
            )
            yhat = predict_pipeline(mlp, train_df, target_col, test_year)
            records.append({"model":"MLP", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["MLP"] = yhat

        if use_knn:
            knn = KNeighborsRegressor(n_neighbors=int(knn_k))
            yhat = predict_treeish(knn, train_df, target_col, test_year)
            records.append({"model":"KNN", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["KNN"] = yhat

        # Prophet (optional)
        if use_prophet and HAVE_PROPHET:
            try: yhat = _predict_prophet(train_df, test_year)
            except Exception: yhat = np.nan
            records.append({"model":"Prophet", "year":test_year, "y_true": test_val, "y_pred": yhat})
            row_vis["Prophet"] = yhat

        # XGBoost (optional)
        if use_xgb and HAVE_XGB:
            try:
                res = _predict_xgb(train_df, test_year)
                if isinstance(res, tuple):
                    yhat, fitted_xgb, feat_cols = res
                    last_feat_info = (fitted_xgb, feat_cols)
                else:
                    yhat = res
            except Exception:
                yhat = np.nan
            records.append({"model":"XGBoost", "year":test_year, "y_true": test_val, "y_pred": float(yhat)})
            row_vis["XGBoost"] = yhat

        pred_store_plot.append(row_vis)

    if not records:
        st.warning("No model predictions were generated. Enable at least one model.")
        st.stop()

    _status["cv"] = "✅"
    st.sidebar.markdown(f"- Cross-validation: **{_status['cv']}**")

    # Metrics table
    pred_df = pd.DataFrame(records)
    metrics = []
    for model in pred_df["model"].unique():
        d = pred_df[pred_df["model"] == model].sort_values("year")
        metrics.append({
            "Model": model,
            "Points": len(d),
            "RMSE": rmse(d["y_true"], d["y_pred"]),
            "MAE":  mae(d["y_true"], d["y_pred"]),
            "MAPE": mape(d["y_true"], d["y_pred"]),
            "R²":   r2(d["y_true"], d["y_pred"]),
        })
    metric_df = pd.DataFrame(metrics).sort_values("RMSE")

    st.markdown("#### 📊 Cross-validation summary (out-of-sample)")
    st.caption("Lower RMSE/MAE/MAPE and higher R² are better. Values are computed only on predictions for t+1.")
    st.dataframe(metric_df, use_container_width=True)

    # Chart: actual vs OOS predictions
    st.markdown("#### 📈 OOS predictions vs actual (each split)")
    plot_df = pd.DataFrame(pred_store_plot).sort_values("year")
    plot_melt = plot_df.melt(id_vars=["year","actual"], var_name="model", value_name="yhat")
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

    # Optional: XGB feature importance
    if use_xgb and HAVE_XGB and last_feat_info:
        model_xgb, feat_cols = last_feat_info
        try:
            imp = model_xgb.get_booster().get_score(importance_type="gain")
            rows = [{"feature": f, "gain": float(imp.get(f, 0.0))} for f in feat_cols]
            imp_df = pd.DataFrame(rows).sort_values("gain", ascending=False)
            st.markdown("#### 🌾 XGBoost — feature importance (gain)")
            st.dataframe(imp_df, use_container_width=True, height=260)
            imp_chart = alt.Chart(imp_df.head(12)).mark_bar().encode(
                x=alt.X("gain:Q", title="Importance (gain)"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("gain:Q", format=",.3f")]
            ).properties(height=260)
            st.altair_chart(imp_chart, use_container_width=True)
        except Exception:
            pass

# =========================
# 🔮 FINAL FORECAST
# =========================
with tab_forecast:
    st.subheader("3) Fit on all history & forecast")
    st.caption("We refit selected models on **all past data** and project **N** future years.")
    h = st.slider("Forecast horizon (years ahead)", 1, 10, 5)

    # We reuse 'sub' from Setup tab
    start_y = int(sub["year"].min())
    last_y = int(sub["year"].max())
    future_years = np.arange(last_y + 1, last_y + h + 1, dtype=int)

    final_rows = []

    # Logistic
    if 'use_logistic' in locals() and use_logistic:
        if len(sub) >= 4:
            fit_all = fit_logistic(sub["year"].to_numpy(), sub[target_col].to_numpy(), K0=None, r0=0.30, t0=None)
            Kf, rf, t0f = float(fit_all.K), float(fit_all.r), float(fit_all.t0)
        else:
            Kf = max(float(sub[target_col].max()) * 1.5, 10.0)
            y_pos = np.maximum(sub[target_col].to_numpy(), 1e-6)
            t_center = sub["year"].to_numpy() - sub["year"].to_numpy().mean()
            b, _ = np.polyfit(t_center, np.log(y_pos), 1)
            rf = float(np.clip(b, 0.05, 1.0)); t0f = float(np.median(sub["year"]))
        fitted_hist = logistic(sub["year"], Kf, rf, t0f)
        fitted_fut = logistic(future_years, Kf, rf, t0f)
        final_rows += [{"model":"Logistic","year":int(y),"value":float(v)} for y,v in zip(sub["year"], fitted_hist)]
        final_rows += [{"model":"Logistic","year":int(y),"value":float(v)} for y,v in zip(future_years, fitted_fut)]

    # Polynomial
    if 'use_poly2' in locals() and use_poly2:
        vals_hist = []
        for y in sub["year"]:
            vals_hist.append(predict_polynomial(sub[sub["year"]<=y]["year"], sub[sub["year"]<=y][target_col], y, degree=int(poly_degree)))
        vals_fut = []
        for y in future_years:
            vals_fut.append(predict_polynomial(sub["year"], sub[target_col], y, degree=int(poly_degree)))
        final_rows += [{"model":"Polynomial","year":int(y),"value":float(v)} for y,v in zip(sub["year"], vals_hist)]
        final_rows += [{"model":"Polynomial","year":int(y),"value":float(v)} for y,v in zip(future_years, vals_fut)]

    # RandomForest (example ML in final)
    if 'use_rf' in locals() and use_rf:
        tmp_all = sub[["year", target_col]].copy()
        X, y_tr, feat_cols, _row_next = build_supervised_matrix(tmp_all, target_col, max_lag=3)
        if X is not None:
            rf_all = RandomForestRegressor(n_estimators=int(rf_trees), random_state=7)
            rf_all.fit(X, y_tr)
            # aligned history where features exist
            hist_years = tmp_all.dropna().reset_index(drop=True)["year"].iloc[-len(X):].tolist()
            hist_pred = rf_all.predict(X)
            final_rows += [{"model":"RandomForest","year":int(yy),"value":float(vv)} for yy, vv in zip(hist_years, hist_pred)]
            # iterative future
            ext = sub[["year", target_col]].copy()
            for y in future_years:
                ext = pd.concat([ext, pd.DataFrame({"year":[y], target_col:[np.nan]})], ignore_index=True)
                X_, y_, feat_cols2, row_next = build_supervised_matrix(ext, target_col, max_lag=3)
                pred = float(ext[target_col].dropna().iloc[-1]) if row_next is None else float(rf_all.predict(row_next)[0])
                ext.loc[ext["year"] == y, target_col] = pred
            rf_full = ext.rename(columns={target_col:"value"})
            rf_full["model"] = "RandomForest"
            final_rows += rf_full.iloc[len(sub):].to_dict("records")

    # Prophet (optional)
    if 'use_prophet' in locals() and use_prophet and HAVE_PROPHET:
        try:
            mp = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=False)
            dfp = sub[["year", target_col]].rename(columns={"year":"ds", target_col:"y"}).copy()
            dfp["ds"] = pd.to_datetime(dfp["ds"].astype(int), format="%Y")
            mp.fit(dfp)
            fdf = pd.DataFrame({"ds": pd.to_datetime(np.r_[sub["year"].to_numpy(), future_years], format="%Y")})
            yhat_all = mp.predict(fdf)[["ds","yhat"]]; yhat_all["year"] = yhat_all["ds"].dt.year
            final_rows += [{"model":"Prophet","year":int(r.year),"value":float(r.yhat)} for r in yhat_all.itertuples(index=False)]
        except Exception:
            pass

    # XGBoost (optional)
    if 'use_xgb' in locals() and use_xgb and HAVE_XGB:
        try:
            tmp_all = sub[["year", target_col]].copy()
            tmp_all = add_time_features(tmp_all, target_col=target_col, max_lag=3).dropna().reset_index(drop=True)
            if not tmp_all.empty:
                feat_cols = [c for c in tmp_all.columns if c not in ["year", target_col]]
                X = tmp_all[feat_cols].values; y_tr = tmp_all[target_col].values
                m = XGBRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=3,
                    subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0, random_state=7
                )
                m.fit(X, y_tr)
                ext = sub[["year", target_col]].copy()
                # history aligned
                hist_pred = m.predict(X)
                hist_years = ext.dropna().iloc[-len(hist_pred):]["year"].tolist()
                final_rows += [{"model":"XGBoost","year":int(yy),"value":float(vv)} for yy, vv in zip(hist_years, hist_pred)]
                # future iterative
                for y in future_years:
                    ext = pd.concat([ext, pd.DataFrame({"year":[y], target_col:[np.nan]})], ignore_index=True)
                    ext_feat = add_time_features(ext, target_col=target_col, max_lag=3)
                    row = ext_feat[ext_feat["year"] == y][feat_cols].tail(1)
                    pred = float(ext[target_col].dropna().iloc[-1]) if row.empty or row.isna().any(axis=None) else float(m.predict(row.values)[0])
                    ext.loc[ext["year"] == y, target_col] = pred
                xgb_full = ext.rename(columns={target_col:"value"})
                xgb_full["model"] = "XGBoost"
                final_rows += xgb_full.iloc[len(sub):].to_dict("records")
        except Exception:
            pass

    # Render
    if final_rows:
        final_df = pd.DataFrame(final_rows)
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

        cdl, cdr = st.columns([2,1])
        with cdl:
            st.download_button(
                "⬇️ Download final forecast (CSV)",
                data=show_df.sort_values(["model","year"]).to_csv(index=False).encode(),
                file_name=f"{country.lower().replace(' ','_')}_{target_col}_final_forecast.csv",
                use_container_width=True
            )
        with cdr:
            st.success("Forecast ready.")
            _status["forecast"] = "✅"
    else:
        st.warning("No final series to plot (no model selected?).")

# =========================
# 🗒️ MODEL NOTES
# =========================
with tab_notes:
    st.subheader("4) How to read & present these results")
    st.markdown("""
**Rolling-origin CV** avoids look-ahead bias: each prediction is made one step into the future using only past data.  
Report the **out-of-sample** metrics (RMSE/MAE/MAPE/R²) — they’re what supervisors care about.

**Model quick guide**
- **Logistic (K, r, t₀)**: S-curve technology diffusion. If this wins, adoption likely saturates; parameters are interpretable.
- **Polynomial trend**: simple macro trend; good baseline for smooth growth.
- **RandomForest / ExtraTrees / GradientBoosting**: tree ensembles on lag/rolling/trend features; capture nonlinearities.
- **SVR (RBF)**, **MLP**: smooth nonlinear approximators (with scaling). Use for short series with strong curvature.
- **Prophet / XGBoost**: available when installed; Prophet = additive trend, XGB = boosted trees.

**What to say in a meeting**
- Describe the **data** (country, years, target).  
- Explain that evaluation used **expanding windows** and **1-step-ahead** predictions.  
- Show the **metric table** (lower is better) and the **OOS prediction chart**.  
- Pick the **top 1–2 models** and show their **final forecast**.  
- Connect the forecast to downstream pages: **ROI** and **Deployment**.
    """)

# --- Sidebar status summary (final render) ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"- Data: **{_status['data']}**")
st.sidebar.markdown(f"- Country/Series: **{_status['country']}**")
st.sidebar.markdown(f"- Models selected: **{_status['models']}**")
st.sidebar.markdown(f"- CV run: **{_status['cv']}**")
st.sidebar.markdown(f"- Forecast: **{_status['forecast']}**")

# Navigation for the story flow
st.page_link("pages/2_Analogs_and_ML.py", label="⬅️ Back to Analogs & ML", icon="📊")
st.page_link("pages/4_ROI_Sensitivity.py", label="Next → ROI & Sensitivity", icon="💸")
st.page_link("pages/5_Deployment_ROI.py", label="Next → Deployment ROI", icon="🧭")
