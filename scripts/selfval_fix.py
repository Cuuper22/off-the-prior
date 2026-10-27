"""A zero-training fix: switch from Timestral to the method of analogues when the series' own
context says the analogues forecast it better.

For each series (context c of length T, horizon H) and each of K backtest windows
k = 1..K (window k = forecast c[-kH:-(k-1)H] from everything before it):
  e_T[k] = mean absolute error of Timestral's median, e_A[k] = same for the analogues.
Per sample we store these errors and the scores of two final forecasts on the real horizon:
  "timestral": Timestral as released;
  "switch":    the analogue forecast as the median, with Timestral's quantiles shifted to it
               (so the spread is Timestral's).
A decision rule (e.g. "switch if e_A[k] < m * e_T[k] for every k") is then evaluated offline by
scripts/selfval_rules.py, which tunes m on one half of the seeds and reports on the other.
Only the context is used (no oracle, no future), so any rule is deployable as is.

python scripts/selfval_fix.py --samples results/map/chaos/samples.npz --sizes 5M 209M --T 2048 --H 128 \
    --windows 2 --out results/fix/selfval/chaos
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp import metrics  # noqa: E402
from otp.generators.base import load_samples  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import MEDIAN_INDEX, forecast, set_threads  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analog_test import analog_forecast  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sizes", nargs="+", default=["5M"])
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    set_threads(None)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = load_samples(a.samples)
    if a.max_samples:
        S = S[: a.max_samples]
    S = [s for s in S if s.context.shape[0] == 1 and s.T >= a.T]
    H, K = a.H, a.windows
    ctx = np.stack([s.context[0, -a.T:] for s in S])
    fut = np.stack([s.future[0, :H] for s in S])
    orc = np.stack([s.oracle_q[0, :H] for s in S])
    per = [s.period for s in S]
    ana = np.stack([analog_forecast(c, H) for c in ctx])
    bt = []  # (context, truth, analogue forecast) per window
    for k in range(1, K + 1):
        c_k = ctx[:, : a.T - k * H]
        y_k = ctx[:, a.T - k * H: a.T - (k - 1) * H]
        bt.append((c_k, y_k, np.stack([analog_forecast(c, H) for c in c_k])))
    rows = []
    for size in a.sizes:
        q = forecast(size, ctx, H, batch_size=16)[:, 0]
        q_bt = [forecast(size, c_k, H, batch_size=16)[:, 0] for c_k, _, _ in bt]
        for i, s in enumerate(S):
            med = q[i, :, MEDIAN_INDEX]
            q_sw = q[i] + (ana[i] - med)[:, None]
            sc_t = metrics.score(ctx[i][None], fut[i][None], q[i][None], per[i])
            sc_s = metrics.score(ctx[i][None], fut[i][None], q_sw[None], per[i])
            sc_o = metrics.score(ctx[i][None], fut[i][None], orc[i][None], per[i])
            r = {"sample_id": s.meta["sample_id"], "config_id": s.meta.get("config_id"), "seed": s.meta.get("seed"),
                 "family": s.meta["family"], "generator": s.meta["generator"], "size": size,
                 "mase_timestral": float(sc_t["mase"][0]), "mase_switch": float(sc_s["mase"][0]),
                 "mase_oracle": float(sc_o["mase"][0]), "scrps_timestral": float(sc_t["scrps"][0]),
                 "scrps_switch": float(sc_s["scrps"][0]), "scrps_oracle": float(sc_o["scrps"][0]),
                 "cov80_timestral": float(sc_t["cov80"][0]), "cov80_switch": float(sc_s["cov80"][0])}
            for k, ((c_k, y_k, a_k), qb) in enumerate(zip(bt, q_bt), start=1):
                r[f"bt{k}_mae_timestral"] = float(np.mean(np.abs(y_k[i] - qb[i, :, MEDIAN_INDEX])))
                r[f"bt{k}_mae_analog"] = float(np.mean(np.abs(y_k[i] - a_k[i])))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_sample.csv"), index=False)
    write_json(os.path.join(a.out, "provenance.json"),
               provenance("scripts/selfval_fix.py", argv=sys.argv, n_samples=len(S), wall_sec=round(time.time() - t0, 1)))
    print(f"wrote {len(df)} rows to {a.out}/per_sample.csv ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
