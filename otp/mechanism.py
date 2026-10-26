"""Mechanism measurements: why Timestral fails on a series, from inside the model.

For a batch of samples (one generator config, many seeds) this module records, from one
plain forward pass (`otp.internals.trace`):

1. **Readout vs. output.** Label: what the oracle expects over the first forecast patch,
   as a change from the last observed value, in units of the context's naive scale
   (`oracle_delta`). Two scores, both cross-validated R^2 across samples:
   - `readout_r2[layer]`: a ridge linear readout from the hidden state of the first
     forecast token after that layer.
   - `output_r2`: Timestral's own median, same quantity, used as a one-feature predictor
     (the R^2 of the best linear fit of the label on it, cross-validated).
   readout > output means the information the oracle uses is linearly present inside the
   model but does not reach the forecast; readout ~ output ~ low means the model does not
   represent it at all.
2. **Attention profile** of forecast tokens (recent-patch mass, seasonal-lag mass, mass after
   a change point, entropy), per time-attention layer.
3. **Template match**: correlation of Timestral's median with a "seasonal template"
   continuation (the average of the last cycles of the context at its dominant period,
   tiled forward), next to the same correlation for the oracle's median. A forecast that
   matches the template much more than the oracle does has collapsed onto a periodic
   pattern the dynamics do not follow.
"""

from __future__ import annotations

import numpy as np

from otp.internals import attention_profile, ridge_readout, trace
from otp.metrics import naive_scale, score
from otp.model import MEDIAN_INDEX, PATCH


def dominant_period(x: np.ndarray, min_p: int = 4, max_p: int = 1024) -> int:
    """Lag of the highest autocorrelation peak in [min_p, max_p] (on the last 4096 steps)."""
    x = np.asarray(x[-4096:], float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / max(ac[0], 1e-12)
    hi = min(max_p, n // 3)
    if hi <= min_p:
        return 1
    seg = ac[min_p:hi]
    # first local maximum region: pick the global max after the first zero crossing
    zc = np.argmax(ac[1:] < 0) + 1 if (ac[1:] < 0).any() else min_p
    lo = max(min_p, zc)
    if lo >= hi:
        return int(min_p + np.argmax(seg))
    return int(lo + np.argmax(ac[lo:hi]))


def template_forecast(x: np.ndarray, horizon: int, period: int, cycles: int = 4) -> np.ndarray:
    """Average of the last `cycles` periods of x, tiled over the horizon."""
    k = min(cycles, len(x) // period)
    if period < 2 or k < 1:
        return np.full(horizon, x[-1])
    segs = np.stack([x[len(x) - (i + 1) * period: len(x) - i * period] for i in range(k)])
    tmpl = segs.mean(0)
    return np.tile(tmpl, int(np.ceil(horizon / period)) + 1)[:horizon]


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _cv_r2_1d(x: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    return ridge_readout(x[:, None], y, alphas=(1e-6,), folds=folds, seed=seed)["r2"]


def measure(size: str, samples, horizon: int = 128, change_step: int | None = None,
            batch: int = 16) -> dict:
    """Run the measurements on `samples` (univariate Sample objects)."""
    ctx = np.stack([s.context[0] for s in samples])
    fut = np.stack([s.future[0, :horizon] for s in samples])
    orc = np.stack([s.oracle_q[0, :horizon] for s in samples])
    period = samples[0].period
    hidden_by_layer: dict[int, list] = {}
    profiles = []
    q_all = []
    for i in range(0, len(samples), batch):
        tr = trace(size, ctx[i:i + batch], horizon)
        q_all.append(tr["quantiles"])
        for li, h in tr["hidden"].items():
            hidden_by_layer.setdefault(li, []).append(h[:, tr["n_ctx_patches"], :])  # first forecast token
        profiles.append(attention_profile(tr, period=period if period >= 2 * PATCH else None, change_step=change_step))
    q = np.concatenate(q_all)
    prof = {k: np.concatenate([p[k] for p in profiles]) for k in profiles[0]}
    hidden = {li: np.concatenate(v) for li, v in hidden_by_layer.items()}

    scale = naive_scale(ctx, period)
    last = ctx[:, -1]
    oracle_delta = (orc[:, :PATCH, MEDIAN_INDEX].mean(1) - last) / scale
    model_delta = (q[:, :PATCH, MEDIAN_INDEX].mean(1) - last) / scale
    real_delta = (fut[:, :PATCH].mean(1) - last) / scale

    readout = {li: ridge_readout(h.astype(np.float64), oracle_delta)["r2"] for li, h in sorted(hidden.items())}
    out_r2 = _cv_r2_1d(model_delta, oracle_delta)
    # single-step labels: the oracle median at steps 1, 4, 16 (as change from the last value)
    step_labels = {}
    for h in (1, 4, 16):
        if h <= horizon:
            lab = (orc[:, h - 1, MEDIAN_INDEX] - last) / scale
            mod = (q[:, h - 1, MEDIAN_INDEX] - last) / scale
            best = max(ridge_readout(hh.astype(np.float64), lab)["r2"] for hh in hidden.values())
            step_labels[f"step{h}"] = {"best_readout_r2": best, "output_r2": _cv_r2_1d(mod, lab),
                                       "label_sd": float(lab.std())}

    sc_m = score(ctx, fut, q, period)
    sc_o = score(ctx, fut, orc, period)
    tm, to = [], []
    for j in range(len(samples)):
        p = dominant_period(ctx[j])
        tmpl = template_forecast(ctx[j], horizon, p)
        tm.append(_corr(q[j, :, MEDIAN_INDEX], tmpl))
        to.append(_corr(orc[j, :, MEDIAN_INDEX], tmpl))
    return {
        "n": len(samples),
        "mase_ratio_oracle_gmean": float(np.exp(np.mean(np.log(sc_m["mase"] / np.maximum(sc_o["mase"], 1e-9))))),
        "readout_r2": readout,
        "output_r2": out_r2,
        "step_labels": step_labels,
        "oracle_delta_sd": float(oracle_delta.std()),
        "realized_vs_oracle_r2": _cv_r2_1d(oracle_delta, real_delta),
        "attn": {k: np.nanmean(v, axis=0).tolist() for k, v in prof.items()},
        "attn_per_sample": prof,
        "template_corr_model": float(np.nanmean(tm)),
        "template_corr_oracle": float(np.nanmean(to)),
    }
