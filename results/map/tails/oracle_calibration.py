"""Check that each config's oracle is calibrated against its own realized futures.

For every config in `otp.grids.tails`, build N independent samples (context 1024, horizon 128, 500 oracle
paths) and report the mean coverage of the oracle's [0.1, 0.9] and [0.01, 0.99] intervals on the realized
future. The oracle is the true conditional distribution (up to Monte Carlo error and, for fbm, the 2048-step
conditioning window), so the means should approach 0.80 and 0.98 as N grows.

Usage: python results/map/tails/oracle_calibration.py --n 100 --out results/map/tails/oracle_calibration.csv
"""
import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from otp import metrics  # noqa: E402
from otp.generators.base import make_sample  # noqa: E402
from otp.grids.tails import configs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = []
for cid, factory in configs():
    c80, c98 = [], []
    for k in range(a.n):
        s = make_sample(factory(k), 1024, 128, seed=zlib.crc32(f"calib/{cid}/{k}".encode()) % (2**31), n_paths=500)
        sc = metrics.score(s.context, s.future, s.oracle_q, s.period)
        c80.append(float(sc["cov80"][0]))
        c98.append(float(sc["cov98"][0]))
    rows.append({"config_id": cid, "n_samples": a.n, "oracle_cov80": np.mean(c80), "oracle_cov98": np.mean(c98),
                 "cov80_se": np.std(c80) / np.sqrt(a.n), "cov98_se": np.std(c98) / np.sqrt(a.n)})
    print(cid, rows[-1]["oracle_cov80"], rows[-1]["oracle_cov98"], flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
