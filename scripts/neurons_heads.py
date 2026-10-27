"""Attention-head atlas for one Timestral size: every time-attention and variate-attention head.

    python scripts/neurons_heads.py --size 5M --corpus results/neurons/corpus/main.npz \
        --mv results/neurons/corpus/mv.npz --out results/colab_out/nn_heads_5M

Writes `heads_time.csv` (one row per layer, head, query type) and `heads_variate.csv`; see `otp/neurons/heads.py`.
Known variables used for scoring: the period of seasonal series, the switch patch of regime series and the coupling
strength of coupled systems.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.model import set_threads  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import heads as Hd  # noqa: E402
from otp.neurons import variables as V  # noqa: E402


def known(c):
    sc = {k: i for i, k in enumerate(V.SERIES_COLS)}
    per = 2.0 ** c["sv"][:, sc["s_log2_period"]]
    lam = np.where(np.isfinite(per) & (per >= 64), per / 32, 0.0)
    L = c["T"] // 32
    swp = c["sv"][:, sc["s_switch_pos"]]
    switch = np.where(np.isfinite(swp), np.floor(swp * L), -1).astype(int)
    rng = np.random.default_rng(11)
    pl = rng.integers(8, max(9, L - 8), len(lam)).astype(int)
    switch_pl = np.where(switch >= 0, pl, -1)
    return lam, switch, switch_pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--mv", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-series", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    set_threads()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    c, specs = C.load_corpus(a.corpus)
    if a.max_series:
        c = {k: (v[:a.max_series] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(specs) else v)
             for k, v in c.items()}
    lam, switch, switch_pl = known(c)
    arr = Hd.collect_time(a.size, c["values"], c["T"], lam, switch, switch_pl, batch=a.batch, device=a.device)
    rows = Hd.summarize_time(arr, lam, switch, a.size)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "heads_time.csv"), index=False, float_format="%.5g")
    print(f"[{time.time() - t0:.0f}s] time heads: {len(rows)} rows", flush=True)
    vrows = []
    if a.mv:
        mc, mspecs = C.load_corpus(a.mv)
        vo, vl = Hd.collect_variate(a.size, mc["values"], mc["T"], mc["gid"], mc["row"], device=a.device)
        sc = {k: i for i, k in enumerate(V.SERIES_COLS)}
        coup = mc["sv"][:, sc["s_coupling"]]
        lag = mc["sv"][:, sc["s_lag"]]
        gen = np.asarray([s["generator"] for s in mspecs])
        for vi, li in enumerate(vl):
            for h in range(vo.shape[2]):
                for qt, qn in ((0, "context"), (1, "forecast")):
                    row = {"size": a.size, "layer": li, "head": h, "query": qn}
                    for r, rn in ((0, "y_row"), (1, "x_row")):
                        m = mc["row"] == r
                        row[f"other_mass_{rn}"] = float(vo[m, vi, h, qt].mean())
                        for g in ("lagged_driver", "feedback_linear"):
                            mm = m & (gen == g)
                            x, y = coup[mm], vo[mm, vi, h, qt]
                            ok = np.isfinite(x)
                            rr = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 5 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan
                            row[f"corr_coupling_{g}_{rn}"] = float(rr)
                    row["uniform_baseline"] = 0.5
                    vrows.append(row)
        pd.DataFrame(vrows).to_csv(os.path.join(a.out, "heads_variate.csv"), index=False, float_format="%.5g")
    write_json(os.path.join(a.out, "meta.json"), provenance("scripts/neurons_heads.py", size=a.size,
               corpus=os.path.basename(a.corpus), n_series=int(c["values"].shape[0]), wall_s=round(time.time() - t0, 1),
               n_time_rows=len(rows), n_variate_rows=len(vrows)))
    print("done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
