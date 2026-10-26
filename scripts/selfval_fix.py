"""A zero-training fix: choose between Timestral and the method of analogues by backtesting on
the series' own context.

For each series (context c of length T, horizon H):
1. backtest: forecast the last H steps of c from c[:-H] with Timestral (median) and with the
   method of analogues; record each method's mean absolute error there;
2. blend weights w = softmax(-MAE / (temp * mean MAE)) over the two methods, for several
   temperatures (`--temps`), plus a hard pick of the lower-error method ("hard");
3. final median = w_T * Timestral median + w_A * analogue forecast; the final quantiles are
   Timestral's quantiles shifted by (final median - Timestral median), so the spread is
   Timestral's.

Only the context is used (no oracle, no future), so the fix is deployable as is. Reported per
sample: MASE and sCRPS of Timestral alone and of the blend, plus the analogue weight.

python scripts/selfval_fix.py --samples results/map/chaos/samples.npz --sizes 5M 209M --T 2048 --H 128 \
    --out results/fix/selfval/chaos
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
    ap.add_argument("--temps", nargs="+", type=float, default=[0.05, 0.1, 0.25])
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
    ctx = np.stack([s.context[0, -a.T:] for s in S])
    fut = np.stack([s.future[0, :a.H] for s in S])
    orc = np.stack([s.oracle_q[0, :a.H] for s in S])
    per = [s.period for s in S]
    bt_ctx, bt_fut = ctx[:, :-a.H], ctx[:, -a.H:]
    ana_bt = np.stack([analog_forecast(c, a.H) for c in bt_ctx])
    ana = np.stack([analog_forecast(c, a.H) for c in ctx])
    rows = []
    for size in a.sizes:
        q_bt = forecast(size, bt_ctx, a.H, batch_size=16)[:, 0]
        q = forecast(size, ctx, a.H, batch_size=16)[:, 0]
        for i, s in enumerate(S):
            e_t = np.mean(np.abs(bt_fut[i] - q_bt[i, :, MEDIAN_INDEX]))
            e_a = np.mean(np.abs(bt_fut[i] - ana_bt[i]))
            med = q[i, :, MEDIAN_INDEX]
            sc_t = metrics.score(ctx[i][None], fut[i][None], q[i][None], per[i])
            sc_o = metrics.score(ctx[i][None], fut[i][None], orc[i][None], per[i])
            for variant in ["hard"] + [f"t{tp:g}" for tp in a.temps]:
                if variant == "hard":
                    w_a = float(e_a < e_t)
                else:
                    tp = float(variant[1:])
                    z = -np.array([e_t, e_a]) / (tp * max(1e-12, 0.5 * (e_t + e_a)))
                    z = np.exp(z - z.max())
                    w_a = float(z[1] / z.sum())
                blend_med = (1 - w_a) * med + w_a * ana[i]
                q_blend = q[i] + (blend_med - med)[:, None]
                sc_b = metrics.score(ctx[i][None], fut[i][None], q_blend[None], per[i])
                rows.append({"sample_id": s.meta["sample_id"], "config_id": s.meta.get("config_id"),
                             "family": s.meta["family"], "generator": s.meta["generator"], "size": size,
                             "variant": variant, "analog_weight": w_a, "backtest_mae_timestral": e_t,
                             "backtest_mae_analog": e_a, "mase_timestral": float(sc_t["mase"][0]),
                             "mase_blend": float(sc_b["mase"][0]), "mase_oracle": float(sc_o["mase"][0]),
                             "scrps_timestral": float(sc_t["scrps"][0]), "scrps_blend": float(sc_b["scrps"][0]),
                             "scrps_oracle": float(sc_o["scrps"][0]), "cov80_timestral": float(sc_t["cov80"][0]),
                             "cov80_blend": float(sc_b["cov80"][0])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_sample.csv"), index=False)
    gm = lambda x: float(np.exp(np.log(np.clip(np.asarray(x, float), 1e-12, None)).mean()))  # noqa: E731
    g = df.groupby(["config_id", "size", "variant"]).apply(lambda d: pd.Series({
        "n": len(d), "analog_weight_mean": d.analog_weight.mean(),
        "mase_ratio_oracle_timestral": gm(d.mase_timestral / d.mase_oracle),
        "mase_ratio_oracle_blend": gm(d.mase_blend / d.mase_oracle),
        "scrps_ratio_oracle_timestral": gm(d.scrps_timestral / d.scrps_oracle),
        "scrps_ratio_oracle_blend": gm(d.scrps_blend / d.scrps_oracle),
        "blend_over_timestral_mase": gm(d.mase_blend / d.mase_timestral),
        "blend_over_timestral_scrps": gm(d.scrps_blend / d.scrps_timestral)}), include_groups=False).reset_index()
    g.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    write_json(os.path.join(a.out, "provenance.json"),
               provenance("scripts/selfval_fix.py", argv=sys.argv, n_samples=len(S), wall_sec=round(time.time() - t0, 1)))
    pd.set_option("display.width", 220)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
