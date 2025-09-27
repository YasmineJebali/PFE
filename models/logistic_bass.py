# models/logistic_bass.py  — logistic & Bass, plus fit evaluation (RMSE/MAPE)
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.optimize import curve_fit

# ---------- Logistic ----------
def logistic(t, K, r, t0):
    """
    Standard logistic function.
    t : year or time (array-like)
    K : carrying capacity (max level)
    r : growth rate
    t0: inflection (midpoint) year
    """
    t = np.asarray(t, dtype=float)
    return K / (1.0 + np.exp(-r * (t - t0)))

@dataclass
class FitResult:
    K: float
    r: float
    t0: float
    success: bool

def fit_logistic(years, y, K0=None, r0=0.3, t0=None, bounds=None):
    """
    Fit logistic by nonlinear least squares.
    years: 1D array of years
    y    : 1D array of observed values (e.g., EV stock)
    K0   : initial guess for K (default ~1.5 * max(y))
    r0   : initial guess for r
    t0   : initial guess for t0 (default = median year)
    bounds: optional ((Kmin, rmin, t0min), (Kmax, rmax, t0max))
    """
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)

    if K0 is None:
        K0 = max(float(np.nanmax(y)) * 1.5, 10.0)
    if t0 is None:
        t0 = float(np.median(years))
    if bounds is None:
        # reasonable generic bounds
        lower = (max(1.0, float(np.nanmax(y)) * 0.8), 0.01, years.min() - 10)
        upper = (float(np.nanmax(y)) * 10.0, 1.5, years.max() + 20)
        bounds = (lower, upper)

    try:
        popt, _ = curve_fit(
            logistic, years, y,
            p0=(K0, r0, t0),
            bounds=bounds,
            maxfev=10000
        )
        K, r, t0 = map(float, popt)
        return FitResult(K=K, r=r, t0=t0, success=True)
    except Exception:
        return FitResult(K=float(K0), r=float(r0), t0=float(t0), success=False)

# ---------- Rough fit (when points are few) ----------
def rough_fit(years, y):
    """
    Quick, robust estimates when you have only 2–3 points:
    - K ≈ 1.5 * max(y)
    - r  from a linear fit on log(y) vs centered time (clipped)
    - t0 = median(year)
    """
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    K_r = max(float(np.nanmax(y)) * 1.5, 10.0)

    y_pos = np.maximum(y, 1e-6)
    t_center = years - years.mean()
    b, a = np.polyfit(t_center, np.log(y_pos), 1)
    r_r = float(np.clip(b, 0.05, 1.0))
    t0_r = float(np.median(years))
    return K_r, r_r, t0_r

# ---------- Bass (optional) ----------
def bass(t, p, q, m, t0=0.0):
    """
    Bass diffusion cumulative adoption curve.
    p : innovation coefficient
    q : imitation coefficient
    m : market potential
    t0: time shift
    """
    t = np.asarray(t, dtype=float) - t0
    # cumulative fraction F(t) = (1 - exp(-(p+q)t)) / (1 + (q/p)*exp(-(p+q)t))
    exp_term = np.exp(-(p + q) * np.maximum(t, 0.0))
    frac = (1.0 - exp_term) / (1.0 + (q / max(p, 1e-6)) * exp_term)
    return m * np.clip(frac, 0.0, 1.0)

# ---------- Evaluation ----------
def evaluate_fit(years, actual, K, r, t0):
    """
    Return (RMSE, MAPE) for a logistic fit on the provided data.
    No sklearn metrics; fully compatible with older versions.
    """
    years = np.asarray(years, dtype=float)
    actual = np.asarray(actual, dtype=float)
    pred = logistic(years, K, r, t0)

    # RMSE
    dif = actual - pred
    rmse = float(np.sqrt(np.nanmean(dif ** 2)))

    # MAPE (ignore zeros)
    mask = actual != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])))
    else:
        mape = float("nan")

    return rmse, mape
