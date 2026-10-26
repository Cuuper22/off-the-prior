"""Does Timestral's cumulative normalization anchor it to the pre-switch regime?

Timestral normalizes each patch with the mean and std of *everything observed since the
start of the context* (not a recent window), and denormalizes the forecast with the
statistics at the last context patch. Hypothesis: after a variance switch inside the
context, that cumulative std stays dominated by the old regime, so intervals come out too
wide after a drop and too narrow after a rise, and more pre-switch history makes it worse.

Causal test: forecast the same samples with (a) the full context, (b) the context cut to
the post-switch part only (the last `d` steps), (c) the last `2d` steps. If cutting the
pre-switch history fixes calibration, the anchoring is the cause. (b) is also a
zero-training fix: change-point-aware context truncation.

Also records `scale_ratio` = cumulative std at the end of the context / std of the
post-switch part, the quantity the hypothesis says drives the miscalibration.

python scripts/normalization_test.py --sizes 5M 28M --seeds 24 --out results/mechanism/normalization
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp import metrics  # noqa: E402
from otp.generators.base import make_sample  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import forecast, set_threads  # noqa: E402


def cases():
    from otp.generators import regime, tails

    out = []
    for factor in (0.25, 4.0):
        for d in (64, 256, 1024):
            out.append((f"variance_jump/f{factor:g}/d{d}", d,
                        lambda s, f=factor, d=d: regime.VarianceJump(factor=f, placement="context", d=d,
                                                                     **regime.randomized_base(s))))
    for delta in (4.0,):
        for d in (64, 256, 1024):
            out.append((f"mean_jump/delta{delta:g}/d{d}", d,
                        lambda s, de=delta, d=d: regime.MeanJump(delta=de, placement="context", d=d,
                                                                 **regime.randomized_base(s))))
    # heavy tails: no switch; "d" = a recent window, testing whether old outliers anchor the scale
    for alpha in (2.0, 1.3):
        out.append((f"stable_ar/alpha{alpha:g}/w256", 256, lambda s, a=alpha: tails.StableAR(alpha=a)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["5M", "28M"])
    ap.add_argument("--contexts", nargs="+", type=int, default=[2048, 8192])
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--out", default="results/mechanism/normalization")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)
    set_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rows = []
    tmax = max(a.contexts)
    for cid, d, factory in cases():
        samples = [make_sample(factory(s), tmax, a.H, seed=zlib.crc32(f"norm/{cid}/{s}".encode()) % 2**31,
                               n_paths=1000, sample_id=f"{cid}/s{s}") for s in range(a.seeds)]
        for T in a.contexts:
            conds = {"full": T, "post_only": d, "post_2x": 2 * d}
            for cond, L in conds.items():
                if L > T:
                    continue
                ctx = np.stack([s.context[:, -L:] for s in samples])
                for size in a.sizes:
                    q = forecast(size, ctx, a.H, batch_size=8)
                    for i, s in enumerate(samples):
                        full = s.context[0, -T:]
                        sc = metrics.score(full, s.future[0], q[i, 0], s.period)  # same scale for every condition
                        so = metrics.score(full, s.future[0], s.oracle_q[0], s.period)
                        cum_sd = full.std(ddof=1)
                        post_sd = full[-d:].std(ddof=1)
                        rows.append({"case": cid, "d": d, "T": T, "condition": cond, "context_used": L,
                                     "size": size, "seed": i, "mase_ratio_oracle": sc["mase"] / so["mase"],
                                     "scrps_ratio_oracle": sc["scrps"] / so["scrps"], "cov80": sc["cov80"],
                                     "cov98": sc["cov98"], "calib": sc["calib"], "oracle_cov80": so["cov80"],
                                     "scale_ratio": cum_sd / max(post_sd, 1e-12)})
        print(f"{cid} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "scores.csv"), index=False)
    g = df.groupby(["case", "T", "condition", "size"])
    summ = g.agg(n=("seed", "size"), cov80=("cov80", "mean"), cov98=("cov98", "mean"), calib=("calib", "mean"),
                 oracle_cov80=("oracle_cov80", "mean"), scale_ratio=("scale_ratio", "median"),
                 scrps_ratio_gmean=("scrps_ratio_oracle", lambda x: float(np.exp(np.log(x).mean()))),
                 mase_ratio_gmean=("mase_ratio_oracle", lambda x: float(np.exp(np.log(x).mean())))).reset_index()
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    write_json(os.path.join(a.out, "provenance.json"),
               provenance("scripts/normalization_test.py", argv=sys.argv, wall_sec=round(time.time() - t0, 1)))
    print(summ.to_string())


if __name__ == "__main__":
    main()
