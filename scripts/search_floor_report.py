"""Failure-search summaries with a floor on the oracle's error.

The search fitness is MASE(Timestral) / MASE(oracle). Feasibility only caps the oracle from above
(oracle sCRPS at most 0.8 x seasonal naive's, otp/search/objective.py), so nearly noise-free genomes, whose
oracle error is close to zero, reach very large ratios with ordinary Timestral errors
(`results/search/oracle_check.csv`). This script recomputes the per-method summaries keeping only genomes whose
oracle sCRPS is at least `floor` x seasonal naive's (column `orc_sn` of the search histories), for several floors,
and the transfer summary at the same floors.

    python scripts/search_floor_report.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp.io import provenance, write_json  # noqa: E402
from otp.search import objective as O  # noqa: E402

FLOORS = (0.0, 0.01, 0.05, 0.1)


def main():
    out = os.path.join(ROOT, "results", "search")
    per_run = []
    for d in sorted(glob.glob(os.path.join(out, "runs", "*"))):
        run = os.path.basename(d)
        method, seed = run.rsplit("_seed", 1)
        h = pd.read_csv(os.path.join(d, "history.csv"))
        h = h[h.feasible.astype(str) == "True"].drop_duplicates("hash")
        for fl in FLOORS:
            q = h[h.orc_sn >= fl]
            cells = {}
            for r in q.itertuples(index=False):
                c = O.cell_of(r.family, r.spec_entropy)
                cells[c] = max(cells.get(c, 0.0), r.fitness)
            per_run.append({"method": method, "seed": int(seed), "floor": fl, "n_feasible": len(q),
                            "best_fitness": float(q.fitness.max()) if len(q) else np.nan,
                            "median_top20": float(q.fitness.nlargest(20).median()) if len(q) else np.nan,
                            "cells_ge_3": int(sum(v >= 3 for v in cells.values())),
                            "cells_ge_2": int(sum(v >= 2 for v in cells.values()))})
    pr = pd.DataFrame(per_run)
    pr.to_csv(os.path.join(out, "comparison_floored_runs.csv"), index=False)
    g = pr.groupby(["method", "floor"]).agg(seeds=("seed", "count"), best_mean=("best_fitness", "mean"),
                                           best_median=("best_fitness", "median"), median_top20_mean=("median_top20", "mean"),
                                           cells_ge_3_mean=("cells_ge_3", "mean"), n_feasible_mean=("n_feasible", "mean")).reset_index()
    g.to_csv(os.path.join(out, "comparison_floored.csv"), index=False)
    t = pd.read_csv(os.path.join(out, "transfer.csv"))
    t = t[t.feasible.astype(str) == "True"]
    rows = []
    for fl in FLOORS:
        for (m, sz), q in t.groupby(["method", "size"]):
            q = q[q.orc_sn >= fl]
            rows.append({"floor": fl, "method": m, "size": sz, "n": len(q), "frac_gt_1.5": float((q.fitness > 1.5).mean()) if len(q) else np.nan, "frac_gt_2": float((q.fitness > 2).mean()) if len(q) else np.nan,
                         "median_fitness": float(q.fitness.median()) if len(q) else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(out, "transfer_floored.csv"), index=False)
    write_json(os.path.join(out, "provenance_floor_report.json"), provenance("scripts/search_floor_report.py", argv=sys.argv, floors=list(FLOORS)))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
