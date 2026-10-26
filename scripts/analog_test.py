"""Template or retrieval? Compare Timestral with two context-only forecasters on the chaos family.

- Seasonal template: the average of the last cycles of the context at its dominant period,
  tiled forward (what a forecaster does if it treats the series as a periodic pattern).
- Method of analogues: delay-embed the context (dimension m, lag chosen from the first zero of
  the autocorrelation, capped), find the k past states nearest to the current one, and
  average what followed each (what a forecaster does if it uses the context as a library of
  trajectories). No training, no knowledge of the equations.

For each sample: MASE of the median forecast of each method on the same context, and the
correlation of Timestral's median with each alternative's forecast. If Timestral tracks the
template more than the analogues, and analogues beat it, the context is used as a pattern to
continue rather than as trajectories to retrieve.

python scripts/analog_test.py --samples results/map/chaos/samples.npz --sizes 5M 28M --T 2048 --H 128 \
    --out results/mechanism/analog
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.generators.base import load_samples  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.mechanism import dominant_period, template_forecast  # noqa: E402
from otp.metrics import naive_scale  # noqa: E402
from otp.model import MEDIAN_INDEX, forecast, set_threads  # noqa: E402


def embed_lag(x: np.ndarray, max_lag: int = 50) -> int:
    x = x - x.mean()
    ac = np.correlate(x[-4096:], x[-4096:], "full")[len(x[-4096:]) - 1:]
    ac = ac / ac[0]
    neg = np.where(ac[1:max_lag + 1] <= 0)[0]
    return int(neg[0] + 1) if len(neg) else max_lag


def analog_forecast(x: np.ndarray, horizon: int, m: int = 6, k: int = 10) -> np.ndarray:
    lag = max(1, embed_lag(x) // 2)
    n = len(x)
    span = (m - 1) * lag
    idx = np.arange(span, n - horizon)  # states whose full continuation is inside the context
    if len(idx) < k + 1:
        return np.full(horizon, x[-1])
    emb = np.stack([x[idx - j * lag] for j in range(m)], axis=1)
    cur = np.array([x[n - 1 - j * lag] for j in range(m)])
    d = np.sqrt(((emb - cur) ** 2).sum(1))
    nn = idx[np.argsort(d)[:k]]
    futs = np.stack([x[i + 1:i + 1 + horizon] - x[i] for i in nn])  # continuation as change from the state
    return x[-1] + futs.mean(0)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sizes", nargs="+", default=["5M", "28M"])
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--out", default="results/mechanism/analog")
    a = ap.parse_args()
    set_threads(None)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = load_samples(a.samples)
    ctx = np.stack([s.context[0, -a.T:] for s in S])
    fut = np.stack([s.future[0, :a.H] for s in S])
    orc = np.stack([s.oracle_q[0, :a.H, MEDIAN_INDEX] for s in S])
    scale = naive_scale(ctx, 1)
    tmpl = np.stack([template_forecast(c, a.H, dominant_period(c)) for c in ctx])
    ana = np.stack([analog_forecast(c, a.H) for c in ctx])
    model = {sz: forecast(sz, ctx, a.H, batch_size=16)[:, 0, :, MEDIAN_INDEX] for sz in a.sizes}
    rows = []
    for i, s in enumerate(S):
        mae = lambda f: float(np.mean(np.abs(fut[i] - f)) / scale[i])  # noqa: E731
        r = {"sample_id": s.meta["sample_id"], "config_id": s.meta.get("config_id"), "generator": s.meta["generator"],
             "knob_value": s.meta["knob_value"], "mase_oracle": mae(orc[i]), "mase_template": mae(tmpl[i]),
             "mase_analog": mae(ana[i])}
        for sz, f in model.items():
            r[f"mase_{sz}"] = mae(f[i])
            r[f"corr_{sz}_template"] = _corr(f[i], tmpl[i])
            r[f"corr_{sz}_analog"] = _corr(f[i], ana[i])
            r[f"corr_{sz}_oracle"] = _corr(f[i], orc[i])
        r["corr_oracle_template"] = _corr(orc[i], tmpl[i])
        r["corr_oracle_analog"] = _corr(orc[i], ana[i])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_sample.csv"), index=False)
    gm = lambda x: float(np.exp(np.log(np.clip(x, 1e-12, None)).mean()))  # noqa: E731
    agg = {"n": ("sample_id", "size"), "mase_oracle": ("mase_oracle", gm), "mase_template": ("mase_template", gm),
           "mase_analog": ("mase_analog", gm), "corr_oracle_template": ("corr_oracle_template", "mean"),
           "corr_oracle_analog": ("corr_oracle_analog", "mean")}
    for sz in a.sizes:
        agg[f"mase_{sz}"] = (f"mase_{sz}", gm)
        for c in ("template", "analog", "oracle"):
            agg[f"corr_{sz}_{c}"] = (f"corr_{sz}_{c}", "mean")
    summ = df.groupby(["config_id", "generator", "knob_value"]).agg(**agg).reset_index()
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    write_json(os.path.join(a.out, "provenance.json"),
               provenance("scripts/analog_test.py", argv=sys.argv, n_samples=len(S), wall_sec=round(time.time() - t0, 1)))
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
