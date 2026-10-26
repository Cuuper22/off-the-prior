"""Metrics on 21-level quantile forecasts.

All forecasters (Timestral, classical models, the true-dynamics oracle) are scored from
the same `(..., H, 21)` quantile arrays, so the comparison is apples to apples.

Definitions (per series, per variate):

- MASE: mean |y - median| / scale, where scale is the in-sample mean absolute error of the
  seasonal naive forecast with period `m` on the context (m = 1 if none, or if m >= T/2).
- sCRPS: CRPS approximated by the average quantile (pinball) loss over the 21 levels, times
  2, divided by the same scale as MASE. "Scaled CRPS".
- WQL: GIFT-Eval style weighted quantile loss on levels 0.1..0.9: sum of 2*pinball over
  time and levels / (sum |y| * 9). Reported for leaderboard comparability; it breaks for
  series whose values sit near zero, so sCRPS is the headline probabilistic metric here.
- coverage[q]: fraction of horizon points with y <= quantile q.
- cov80 / cov98: fraction inside the [0.1, 0.9] / [0.01, 0.99] intervals.
- calib: mean over the 21 levels of |coverage[q] - q| (0 is perfectly calibrated).
"""

from __future__ import annotations

import numpy as np

from otp.model import QUANTILE_LEVELS, MEDIAN_INDEX

LEVELS = np.asarray(QUANTILE_LEVELS)
WQL_IDX = [QUANTILE_LEVELS.index(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
I10, I90 = QUANTILE_LEVELS.index(0.1), QUANTILE_LEVELS.index(0.9)
I01, I99 = QUANTILE_LEVELS.index(0.01), QUANTILE_LEVELS.index(0.99)


def naive_scale(context: np.ndarray, m: int = 1) -> np.ndarray:
    """In-sample seasonal-naive MAE along the last axis; `context` `(..., T)`."""
    t = context.shape[-1]
    if m < 1 or m >= t // 2:
        m = 1
    d = np.abs(context[..., m:] - context[..., :-m])
    s = np.nanmean(d, axis=-1)
    return np.maximum(s, 1e-8)


def pinball(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Pinball loss `(..., H, 21)` for truth `(..., H)` and quantiles `(..., H, 21)`."""
    diff = y[..., None] - q
    return np.maximum(LEVELS * diff, (LEVELS - 1.0) * diff)


def score(context: np.ndarray, future: np.ndarray, q: np.ndarray, m: int = 1) -> dict[str, np.ndarray]:
    """Metrics for one batch. Shapes: context `(..., T)`, future `(..., H)`, q `(..., H, 21)`.

    Returns arrays with the leading shape `(...)`, plus `coverage` `(..., 21)`.
    """
    scale = naive_scale(context, m)
    med = q[..., MEDIAN_INDEX]
    mae = np.mean(np.abs(future - med), axis=-1)
    pl = pinball(future, q)
    crps = 2.0 * pl.mean(axis=(-1, -2))
    wql = 2.0 * pl[..., WQL_IDX].sum(axis=(-1, -2)) / (np.abs(future).sum(axis=-1) * len(WQL_IDX) + 1e-12)
    cov = (future[..., None] <= q).mean(axis=-2)
    inside80 = ((future >= q[..., I10]) & (future <= q[..., I90])).mean(axis=-1)
    inside98 = ((future >= q[..., I01]) & (future <= q[..., I99])).mean(axis=-1)
    return {
        "mase": mae / scale,
        "scrps": crps / scale,
        "wql": wql,
        "cov80": inside80,
        "cov98": inside98,
        "calib": np.abs(cov - LEVELS).mean(axis=-1),
        "coverage": cov,
        "scale": scale,
    }


def per_step_abs_error(future: np.ndarray, q: np.ndarray) -> np.ndarray:
    """|y - median| per horizon step `(..., H)`, for error-growth curves."""
    return np.abs(future - q[..., MEDIAN_INDEX])


def geo_mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if x.size else float("nan")
