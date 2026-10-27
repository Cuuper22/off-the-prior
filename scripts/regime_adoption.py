"""Adoption diagnostic: how much of a post-switch level shift does Timestral's median reflect at H = 16?

For every sample of generator mean_jump with placement "context" in a sweep folder (it reads
`<dir>/samples.npz`, re-created by scripts/sweep.py with the folder's command), run Timestral at each
(size, context length T) and compute, over the first 16 horizon steps,

    adoption = 1 + mean_h (m_timestral(h) - m_oracle(h)) / shift

where m is the median forecast, m_oracle is the true-dynamics oracle median (new regime, true
state), and shift = delta x (stationary std of the AR noise) is the size of the level jump.
Reading: 1 = the median sits at the oracle (post-switch) level, 0 = the median sits where the
oracle would be had the old level continued (oracle median minus shift). If the switch is before
the visible window (d > T) the context holds only post-switch data, and adoption is measured
against the same post-switch oracle.

Outputs in the sweep folder:
  adoption_by_seed.csv  one row per size x T x sample
  adoption.csv          mean, median, std over seeds per size x T x delta x d (and over all deltas: delta = "all")
  adoption_provenance.json

Usage:
  python scripts/regime_adoption.py --dir results/map/regime --sizes 5M 28M --contexts 512 2048 8192
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
from otp.generators.regime import stationary_sd  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.model import MEDIAN_INDEX, forecast, set_threads  # noqa: E402

H_ADOPT = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sizes", nargs="+", default=["5M", "28M"])
    ap.add_argument("--contexts", nargs="+", type=int, default=[512, 2048, 8192])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    set_threads()
    t_start = time.time()
    samples = [s for s in load_samples(os.path.join(a.dir, "samples.npz"))
               if s.meta["generator"] == "mean_jump" and s.meta["params"]["placement"] == "context"]
    print(f"{len(samples)} mean_jump / context samples", flush=True)
    rows = []
    for size in a.sizes:
        for t in a.contexts:
            ctx = np.stack([s.context[..., -t:] for s in samples])
            q = forecast(size, ctx, H_ADOPT, batch_size=a.batch_size)       # (B, 1, 16, 21)
            for s, qs in zip(samples, q):
                p = s.meta["params"]
                shift = p["delta"] * stationary_sd(p["rho"], p["sigma"])
                diff = qs[0, :H_ADOPT, MEDIAN_INDEX] - s.oracle_q[0, :H_ADOPT, MEDIAN_INDEX]
                rows.append({"size": size, "T": t, "config_id": s.meta["config_id"], "seed": s.meta["seed"],
                             "delta": p["delta"], "d": int(p["d"]), "shift": shift,
                             "adoption": 1.0 + float(diff.mean()) / shift})
            print(f"{size} T={t} done ({time.time() - t_start:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["switch_visible"] = df.d <= df["T"]
    df.to_csv(os.path.join(a.dir, "adoption_by_seed.csv"), index=False)
    agg = lambda g: pd.Series({"n": len(g), "adoption_mean": g.adoption.mean(), "adoption_median": g.adoption.median(),
                               "adoption_std": g.adoption.std()})
    by_delta = df.groupby(["size", "T", "delta", "d", "switch_visible"]).apply(agg, include_groups=False).reset_index()
    pooled = df.groupby(["size", "T", "d", "switch_visible"]).apply(agg, include_groups=False).reset_index()
    pooled.insert(2, "delta", "all")
    out = pd.concat([by_delta.astype({"delta": object}), pooled], ignore_index=True)
    out.to_csv(os.path.join(a.dir, "adoption.csv"), index=False)
    write_json(os.path.join(a.dir, "adoption_provenance.json"),
               provenance("scripts/regime_adoption.py", argv=sys.argv, n_samples=len(samples), n_rows=len(df),
                          wall_sec=round(time.time() - t_start, 1)))
    print(f"wrote adoption.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
