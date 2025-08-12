import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import curve_fit

def logistic(t, K, r, t0):
    t = np.asarray(t, dtype=float)
    return K / (1.0 + np.exp(-r * (t - t0)))

@dataclass
class LogisticFit:
    K: float
    r: float
    t0: float
    cov: Optional[np.ndarray] = None

def fit_logistic(years, y, K0=None, r0=0.3, t00=None) -> LogisticFit:
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    if K0 is None:
        K0 = max(y) * 1.2
    if t00 is None:
        t00 = years[0] + (years[-1] - years[0]) / 2.0
    p0 = [K0, r0, t00]
    bounds = ([max(y), 0.001, years[0]-10], [1e9, 5.0, years[-1]+10])
    popt, pcov = curve_fit(logistic, years, y, p0=p0, bounds=bounds, maxfev=10000)
    K_hat, r_hat, t0_hat = popt
    return LogisticFit(K_hat, r_hat, t0_hat, cov=pcov)

def bass_cumulative(t, m, p, q):
    t = np.asarray(t, dtype=float)
    F = 1 - np.exp(-(p+q)*t) / (1 + (q/p) * (np.exp(-(p+q)*t) - 1))
    return m * F

def fit_bass(years, cumulative, m0=None, p0=0.03, q0=0.38):
    years = np.asarray(years, dtype=float)
    cumulative = np.asarray(cumulative, dtype=float)
    if m0 is None:
        m0 = float(max(cumulative) * 1.2)
    popt, pcov = curve_fit(
        bass_cumulative,
        years - years.min(),
        cumulative,
        p0=[m0, max(1e-4, p0), max(1e-4, q0)],
        bounds=([max(cumulative), 1e-4, 1e-4], [1e9, 1.0, 2.0]),
        maxfev=10000
    )
    m_hat, p_hat, q_hat = popt
    return (m_hat, p_hat, q_hat), pcov
