# utils/analytics.py — small helpers you can import anywhere
import numpy as np
import pandas as pd

# ---- Metrics ----
def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if np.any(mask) else np.nan

def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

# ---- Logistic extras ----
def logistic_milestones(K, r, t0, levels=(0.1, 0.5, 0.9)):
    out = {}
    for p in levels:  # t = t0 - (1/r)*ln(1/p - 1)
        out[int(round(p * 100))] = float(t0 - (1.0 / r) * np.log(1.0 / p - 1.0))
    return out

# ---- CV helper: who wins most splits ----
def split_wins(pred_df):
    years = sorted(pred_df["year"].unique())
    models = pred_df["model"].unique()
    wins = {m: 0 for m in models}
    for y in years:
        d = pred_df[pred_df["year"] == y].copy()
        d["ae"] = (d["y_true"] - d["y_pred"]).abs()
        d = d.sort_values("ae")
        if len(d) > 0:
            wins[d.iloc[0]["model"]] += 1
    total = len(years)
    return {m: (w, (w / total if total else np.nan)) for m, w in wins.items()}

# ---- Outlier flag (simple & robust) ----
def flag_outliers(series, z=3.0):
    x = pd.Series(series, dtype=float)
    m, s = x.mean(), x.std(ddof=1)
    s = s if s > 0 else 1.0
    return (np.abs((x - m) / s) > z)

# ---- Correlated draws for (r, t0) from analogs ----
def analog_rt0_stats(fit_rows):  # fit_rows = list of (r, t0)
    arr = np.asarray(fit_rows, float)
    if arr.shape[0] < 2:
        mu = np.array([0.35, 2031.0])
        cov = np.diag([0.05**2, 1.5**2])
        return mu, cov
    return arr.mean(axis=0), np.cov(arr.T)

def draw_correlated(mu, cov, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=n)
