"""Score a list of `Sample`s with Timestral sizes, classical models and the true-dynamics oracle.

Forecaster names:
- "timestral-<size>"       univariate/multivariate main rows, covariates ignored
- "timestral-<size>+cov"   same, with the sample's past/known covariates passed in
- "seasonal_naive", "ets", "arima"
- "oracle"                 the sample's Monte Carlo true-dynamics quantiles

Output: one row per (sample, forecaster, main-row variate) with metadata and metrics.
`with_reference_ratios` adds error ratios and gaps to the oracle and to the best classical
model for every metric. A failure only counts relative to those references.
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd

from otp import metrics
from otp.classical import CLASSICAL, classical_forecast
from otp.generators.base import Sample
from otp.model import forecast

METRIC_COLS = ("mase", "scrps", "wql", "cov80", "cov98", "calib")
CLASSICAL_NAMES = tuple(CLASSICAL)


def _group(samples: list[Sample]) -> dict:
    g = defaultdict(list)
    for i, s in enumerate(samples):
        key = (s.context.shape, s.H, None if s.past_cov is None else s.past_cov.shape,
               None if s.known_cov is None else s.known_cov.shape)
        g[key].append(i)
    return g


def predict(name: str, samples: list[Sample], max_context: int | None = None, batch_size: int = 32) -> list[np.ndarray]:
    """Quantiles `(V, H, 21)` per sample for one forecaster."""
    out: list[np.ndarray | None] = [None] * len(samples)
    if name == "oracle":
        return [s.oracle_q for s in samples]
    if name in CLASSICAL:
        for i, s in enumerate(samples):
            ctx = s.context if max_context is None else s.context[..., -max_context:]
            out[i] = classical_forecast(name, ctx[None], s.H, s.period)[0]
        return out
    assert name.startswith("timestral-"), f"unknown forecaster {name}"
    spec = name[len("timestral-"):]
    use_cov = spec.endswith("+cov")
    size = spec[:-4] if use_cov else spec
    for (_, h, _, _), idx in _group(samples).items():
        ctx = np.stack([samples[i].context for i in idx])
        pc = kc = None
        if use_cov:
            if samples[idx[0]].past_cov is not None:
                pc = np.stack([samples[i].past_cov for i in idx])
            if samples[idx[0]].known_cov is not None:
                kc = np.stack([samples[i].known_cov for i in idx])
        q = forecast(size, ctx, h, past_covariates=pc, known_covariates=kc, max_context=max_context,
                     batch_size=batch_size)
        for j, i in enumerate(idx):
            out[i] = q[j]
    return out


def score_samples(samples: list[Sample], forecasters: list[str], max_context: int | None = None,
                  extra: dict | None = None, keep_coverage: bool = False) -> pd.DataFrame:
    """Run every forecaster on every sample and return the long table of metrics.

    `max_context` truncates the context for every forecaster (the oracle is unaffected: it
    conditions on the true state). Metric scaling always uses the truncated context.
    """
    rows = []
    for name in forecasters:
        t0 = time.time()
        qs = predict(name, samples, max_context=max_context)
        dt = (time.time() - t0) / max(1, len(samples))
        for s, q in zip(samples, qs):
            ctx = s.context if max_context is None else s.context[..., -max_context:]
            sc = metrics.score(ctx, s.future, q, s.period)
            for v in range(s.context.shape[0]):
                r = {
                    "sample_id": s.meta.get("sample_id"),
                    "family": s.meta.get("family"),
                    "generator": s.meta.get("generator"),
                    "knob": s.meta.get("knob"),
                    "knob_value": s.meta.get("knob_value"),
                    "seed": s.meta.get("seed"),
                    "T": ctx.shape[-1],
                    "H": s.H,
                    "variate": v,
                    "forecaster": name,
                    "sec_per_sample": dt,
                }
                for k in METRIC_COLS:
                    r[k] = float(sc[k][v])
                if keep_coverage:
                    for li, lv in enumerate(metrics.LEVELS):
                        r[f"cov_q{lv:.2f}"] = float(sc["coverage"][v, li])
                if extra:
                    r.update(extra)
                rows.append(r)
    return pd.DataFrame(rows)


def with_reference_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add `<metric>_ratio_oracle`, `<metric>_gap_oracle`, `<metric>_ratio_classical` (best of
    the classical models present, chosen per sample and metric) for mase and scrps."""
    keys = ["sample_id", "variate", "T", "H"]
    out = df.copy()
    for m in ("mase", "scrps"):
        orc = df[df.forecaster == "oracle"][keys + [m]].rename(columns={m: f"{m}_oracle"})
        cl = df[df.forecaster.isin(CLASSICAL_NAMES)].groupby(keys, as_index=False)[m].min().rename(
            columns={m: f"{m}_best_classical"})
        out = out.merge(orc, on=keys, how="left").merge(cl, on=keys, how="left")
        out[f"{m}_ratio_oracle"] = out[m] / out[f"{m}_oracle"].clip(lower=1e-9)
        out[f"{m}_gap_oracle"] = out[m] - out[f"{m}_oracle"]
        out[f"{m}_ratio_classical"] = out[m] / out[f"{m}_best_classical"].clip(lower=1e-9)
    return out
