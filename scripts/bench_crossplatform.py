"""Compare two builds of the same benchmark file (e.g. Windows laptop vs. Linux VM), per config.

python scripts/bench_crossplatform.py A.npz B.npz --label-a "..." --label-b "..." --out bench/data/cross_platform_diag.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otp.io import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    A, B = np.load(x.a), np.load(x.b)
    assert list(A["series_id"]) == list(B["series_id"])
    diff, tot = collections.Counter(), collections.Counter()
    for i in range(len(A["series_id"])):
        cid = json.loads(str(A["meta"][i]))["config_id"]
        tot[cid] += 1
        same = all(np.array_equal(A[k][i], B[k][i], equal_nan=True) for k in ("context", "future", "oracle_q"))
        diff[cid] += 0 if same else 1
    res = {"a": x.label_a, "b": x.label_b, "n_series": int(sum(tot.values())), "n_configs": len(tot),
           "identical_configs": sum(1 for k in tot if diff[k] == 0),
           "differing": {k: f"{diff[k]}/{tot[k]}" for k in sorted(tot) if diff[k]},
           "series_identical": int(sum(tot.values()) - sum(diff.values()))}
    write_json(x.out, res)
    print(res)


if __name__ == "__main__":
    main()
