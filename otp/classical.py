"""Classical baselines producing the same 21 quantiles as Timestral.

- seasonal_naive: repeat the last season; Gaussian quantiles with sigma from in-sample
  seasonal-naive residuals, growing with sqrt(number of seasons ahead).
- ets: statsmodels ETSModel, additive error, additive damped trend, additive season when
  1 < m <= 128 and the context holds >= 3 seasons; Gaussian quantiles from the analytic
  forecast variance.
- arima: statsmodels ARIMA over a small fixed order grid, picked by AIC; Gaussian quantiles.

Classical fits use at most the last `CLASSICAL_MAX_CONTEXT` steps (fitting cost), and fall
back to seasonal naive if a fit fails. Each returns `(H, 21)` for one univariate series.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm

from otp.model import QUANTILE_LEVELS

Z = norm.ppf(np.asarray(QUANTILE_LEVELS))
CLASSICAL_MAX_CONTEXT = 1024
ARIMA_ORDERS = ((1, 0, 0), (2, 0, 1), (1, 1, 0), (2, 1, 1), (0, 1, 1))


def _gauss(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.maximum(np.nan_to_num(sd, nan=0.0), 1e-9)
    return mean[:, None] + sd[:, None] * Z[None, :]


def seasonal_naive(y: np.ndarray, horizon: int, m: int = 1) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if m < 1 or m >= len(y) // 2:
        m = 1
    last = y[-m:]
    reps = int(np.ceil(horizon / m))
    mean = np.tile(last, reps)[:horizon]
    resid = y[m:] - y[:-m]
    sigma = np.nanstd(resid) if resid.size > 1 else 1e-6
    k = np.arange(horizon) // m + 1
    return _gauss(mean, sigma * np.sqrt(k))


def ets(y: np.ndarray, horizon: int, m: int = 1) -> np.ndarray:
    import pandas as pd
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = np.asarray(y, dtype=float)[-CLASSICAL_MAX_CONTEXT:]
    seasonal = 1 < m <= 128 and len(y) >= 3 * m
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = ETSModel(pd.Series(y), error="add", trend="add", damped_trend=True,
                           seasonal="add" if seasonal else None, seasonal_periods=m if seasonal else None)
            res = mod.fit(disp=False, maxiter=200)
            p = res.get_prediction(start=len(y), end=len(y) + horizon - 1)
            mean = np.asarray(p.predicted_mean)
            sd = np.sqrt(np.asarray(p.var_pred_mean))
        if not np.all(np.isfinite(mean)):
            raise ValueError("non-finite ETS forecast")
        return _gauss(mean, sd)
    except Exception:
        return seasonal_naive(y, horizon, m)


def arima(y: np.ndarray, horizon: int, m: int = 1) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    y = np.asarray(y, dtype=float)[-CLASSICAL_MAX_CONTEXT:]
    best, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in ARIMA_ORDERS:
            try:
                r = ARIMA(y, order=order).fit()
                if np.isfinite(r.aic) and r.aic < best_aic:
                    best, best_aic = r, r.aic
            except Exception:
                continue
    if best is None:
        return seasonal_naive(y, horizon, m)
    f = best.get_forecast(horizon)
    mean, sd = np.asarray(f.predicted_mean), np.sqrt(np.asarray(f.var_pred_mean))
    if not np.all(np.isfinite(mean)):
        return seasonal_naive(y, horizon, m)
    return _gauss(mean, sd)


CLASSICAL = {"seasonal_naive": seasonal_naive, "ets": ets, "arima": arima}


def classical_forecast(name: str, context: np.ndarray, horizon: int, m: int = 1) -> np.ndarray:
    """`(B, V, T)` contexts to `(B, V, H, 21)` with one classical model per row."""
    fn = CLASSICAL[name]
    b, v, _ = context.shape
    out = np.empty((b, v, horizon, len(QUANTILE_LEVELS)))
    for i in range(b):
        for j in range(v):
            out[i, j] = fn(context[i, j], horizon, m)
    return out
