"""Rebuild the oracle rows of a sweep's scores.csv for every context length T used.

Early sweeps stored the oracle's rows at the longest context only, but MASE and sCRPS are
scaled by the naive error of the context the forecaster saw, so ratios at shorter contexts
mixed two scales. The oracle's quantiles do not depend on T, so its rows can be recomputed
exactly from samples.npz without running any model.

python scripts/repair_oracle_rows.py <sweep_folder> [--samples <folder with samples.npz>] [--extra scores.csv ...]
`--extra`: other score files whose T values also need oracle rows (e.g. big-model runs on the same samples).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.generators.base import load_samples  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--samples", default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    import importlib.util

    spec = importlib.util.spec_from_file_location("sweep", os.path.join(os.path.dirname(__file__), "sweep.py"))
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    path = os.path.join(a.folder, "scores.csv")
    df = pd.read_csv(path)
    Ts = set(df["T"].unique())
    for e in a.extra:
        Ts |= set(pd.read_csv(e)["T"].unique())
    Hs = sorted(df["H"].unique())
    samples = load_samples(os.path.join(a.samples or a.folder, "samples.npz"))
    rows = []
    for s in samples:
        for t in sorted(Ts):
            if t <= s.T:
                rows += sweep.rows_for(s, s.oracle_q, "oracle", int(t), [int(h) for h in Hs], {})
    new = pd.DataFrame(rows)
    keep = df[df.forecaster != "oracle"]
    out = pd.concat([keep, new[keep.columns.intersection(new.columns)]], ignore_index=True)
    n_old = int((df.forecaster == "oracle").sum())
    out.to_csv(path, index=False)
    write_json(os.path.join(a.folder, "provenance_oracle_repair.json"),
               provenance("scripts/repair_oracle_rows.py", argv=sys.argv, oracle_rows_before=n_old,
                          oracle_rows_after=len(new), context_lengths=sorted(int(t) for t in Ts)))
    print(f"{path}: oracle rows {n_old} -> {len(new)} for T in {sorted(int(t) for t in Ts)}")


if __name__ == "__main__":
    main()
