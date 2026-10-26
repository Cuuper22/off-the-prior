"""Turn a sweep's long `scores.csv` into failure curves: ratios/gaps to references, aggregated over seeds.

`ratios(df)` adds, for mase and scrps, per (sample, T, H, variate):
  <m>_ratio_oracle, <m>_gap_oracle, <m>_ratio_classical (vs the best classical model on that sample)
`curve(df, by)` aggregates Timestral rows over seeds: geometric mean of ratios, mean, std,
and a 95% bootstrap interval of the geometric mean (1000 resamples, fixed seed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSICAL_NAMES = ("seasonal_naive", "ets", "arima")
KEYS = ["sample_id", "T", "H", "variate"]


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for m in ("mase", "scrps"):
        orc = df[df.forecaster == "oracle"]
        # the oracle row is stored at the longest context only; it does not depend on T
        orc = orc[["sample_id", "H", "variate", m]].drop_duplicates(["sample_id", "H", "variate"])
        orc = orc.rename(columns={m: f"{m}_oracle"})
        cl = df[df.forecaster.isin(CLASSICAL_NAMES)].groupby(KEYS, as_index=False)[m].min()
        cl = cl.rename(columns={m: f"{m}_best_classical"})
        out = out.merge(orc, on=["sample_id", "H", "variate"], how="left").merge(cl, on=KEYS, how="left")
        out[f"{m}_ratio_oracle"] = out[m] / out[f"{m}_oracle"].clip(lower=1e-9)
        out[f"{m}_gap_oracle"] = out[m] - out[f"{m}_oracle"]
        out[f"{m}_ratio_classical"] = out[m] / out[f"{m}_best_classical"].clip(lower=1e-9)
    return out


def _gm(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else np.nan


def _boot_gm(x, n=1000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    lx = np.log(x)
    b = np.exp(lx[r.integers(0, lx.size, (n, lx.size))].mean(1))
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def curve(df: pd.DataFrame, by: list[str], metric: str = "mase_ratio_oracle") -> pd.DataFrame:
    """Aggregate a ratio column over seeds for every group in `by` (e.g. config, forecaster, T, H)."""
    rows = []
    for key, g in df.groupby(by, dropna=False):
        lo, hi = _boot_gm(g[metric])
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "n": len(g),
                     f"{metric}_gmean": _gm(g[metric]), f"{metric}_mean": float(g[metric].mean()),
                     f"{metric}_std": float(g[metric].std()), f"{metric}_lo95": lo, f"{metric}_hi95": hi})
    return pd.DataFrame(rows)
