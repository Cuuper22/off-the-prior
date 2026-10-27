"""Compare two per-sample result files of the same samples (e.g. laptop GPU vs. VM CPU).

python scripts/parity_check.py A.csv B.csv --key sample_id --out parity.json
Reports, per shared numeric column, the max and median relative difference.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otp.io import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--key", default="sample_id")
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    a, b = pd.read_csv(x.a), pd.read_csv(x.b)
    m = a.merge(b, on=x.key, suffixes=("_a", "_b"))
    res = {"a": x.a, "b": x.b, "n_shared": len(m), "columns": {}}
    for c in a.columns:
        if c == x.key or c + "_b" not in m or not np.issubdtype(m[c + "_a"].dtype, np.number):
            continue
        r = ((m[c + "_b"] - m[c + "_a"]).abs() / m[c + "_a"].abs().clip(lower=1e-12)).replace([np.inf], np.nan)
        res["columns"][c] = {"max_rel_diff": float(r.max()), "median_rel_diff": float(r.median())}
    write_json(x.out, res)
    print(res)


if __name__ == "__main__":
    main()
