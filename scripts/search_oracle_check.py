"""Check the top genomes of the failure search for small-oracle-error inflation.

A fitness is MASE(Timestral) / MASE(oracle). When a genome is nearly deterministic the oracle's MASE is tiny and
the ratio can be large even if Timestral's own error is ordinary. For the top genomes of every search run this
rebuilds the three search samples (same seeds and settings as otp/search/objective.py), forecasts them with
Timestral-5M, and reports absolute MASE of Timestral, the oracle and seasonal naive, next to the ratio.

    python scripts/search_oracle_check.py [--top 5] [--out results/search/oracle_check.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp import metrics  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import forecast  # noqa: E402
from otp.search import objective as O  # noqa: E402


def gm(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean())) if x.size and np.isfinite(x).all() and (x > 0).all() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--size", default="5M")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "search", "oracle_check.csv"))
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "search", "runs", "*"))):
        h = pd.read_csv(os.path.join(d, "history.csv"))
        h = h[h.feasible.astype(str) == "True"].sort_values("fitness", ascending=False).drop_duplicates("hash").head(a.top)
        for r in h.itertuples(index=False):
            genome = np.array(json.loads(r.genome))
            recs = [O._gen_record((genome, s, O.T_CTX, O.HORIZON, O.N_PATHS)) for s in O.SEARCH_SEEDS]
            ctx = np.stack([rc["sample"].context for rc in recs])        # (S, 1, T)
            q = forecast(a.size, ctx, O.HORIZON)                           # (S, 1, H, 21)
            mt = [float(metrics.score(rc["sample"].context, rc["sample"].future, q[i], rc["period"])["mase"][0]) for i, rc in enumerate(recs)]
            mo = [rc["mase_orc"] for rc in recs]
            ms = [rc["mase_sn"] for rc in recs]
            rows.append({"run": os.path.basename(d), "hash": r.hash, "fitness_search": r.fitness,
                         "ratio_recomputed": gm(np.array(mt) / np.maximum(mo, 1e-9)),
                         "mase_timestral_gm": gm(mt), "mase_oracle_gm": gm(mo), "mase_snaive_gm": gm(ms),
                         "mase_oracle_min": float(min(mo)), "timestral_over_snaive": gm(np.array(mt) / np.array(ms)),
                         "oracle_over_snaive": gm(np.array(mo) / np.array(ms)), "signature": r.signature, "family": r.family})
            print(rows[-1]["run"], rows[-1]["hash"], f"{r.fitness:.1f}", f"orc {rows[-1]['mase_oracle_gm']:.4g}",
                  f"ts/sn {rows[-1]['timestral_over_snaive']:.3g}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    write_json(os.path.join(os.path.dirname(a.out), "provenance_oracle_check.json"),
               provenance("scripts/search_oracle_check.py", argv=sys.argv, n_genomes=len(df)))


if __name__ == "__main__":
    main()
