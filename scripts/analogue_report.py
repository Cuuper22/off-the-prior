"""Timestral's error ratio to the best classical model on the real analogue windows, next to the synthetic
family's ratio from the failure-map sweeps.

Real side: for every analogue window, ratio = MASE (or sCRPS) of a Timestral size divided by the lower of the
seasonal naive, ETS and ARIMA values on that window; aggregated per family and per family/dataset as the geometric
mean over windows (the same definition as `mase_ratio_classical_gmean` in the sweep summaries).

Synthetic side: `mase_ratio_classical_gmean` / `scrps_ratio_classical_gmean` rows of
`results/map/<grid>/summary_all_sizes.csv` at context T = 2048, for the generators mapped to each family, taken at
horizon 16 and at horizon 128 (geometric mean over generators and knob values).

python scripts/analogue_report.py [--scores results/regression_baseline/scores.csv] [--out results/regression_baseline/analogue_vs_synthetic.csv]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# analogue family -> (summary folder, generators)
SYNTHETIC = {
    "delay_feedback": ("chaos", ["mackey_glass"]),
    "coupled_oscillators": ("chaos", ["kuramoto"]),
    "heavy_tails": ("tails", ["fbm", "hawkes", "msm_cascade", "stable_ar"]),
    "regime_switches": ("regime", ["frequency_jump", "markov_ar", "mean_jump", "variance_jump"]),
}
CLASSICAL = ["seasonal_naive", "ets", "arima"]


def gmean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else float("nan")


def real_side(scores: pd.DataFrame) -> pd.DataFrame:
    d = scores[scores.set == "analogues"].copy()
    d["group_dataset"] = d["generator"].astype(str)
    out = []
    for m in ("mase", "scrps"):
        piv = d.pivot_table(index=["window_index", "family_analogue", "group_dataset"], columns="forecaster", values=m)
        have = [c for c in CLASSICAL if c in piv.columns]
        # windows scored by every classical model only, so "best classical" means the same thing everywhere
        piv = piv.dropna(subset=have)
        best = piv[have].min(axis=1)
        for f in [c for c in piv.columns if c.startswith("timestral")]:
            r = (piv[f] / best).rename("ratio").reset_index()
            for level, keys in (("family", ["family_analogue"]), ("family/dataset", ["family_analogue", "group_dataset"])):
                for k, g in r.groupby(keys):
                    k = k if isinstance(k, tuple) else (k,)
                    out.append({"level": level, "family_analogue": k[0], "dataset": k[1] if len(k) > 1 else "ALL", "forecaster": f,
                                "metric": m, "n_windows": len(g), "real_ratio_to_best_classical_gmean": gmean(g["ratio"])})
    return pd.DataFrame(out)


def synthetic_side(map_dir: str) -> pd.DataFrame:
    rows = []
    for fam, (folder, gens) in SYNTHETIC.items():
        path = os.path.join(map_dir, folder, "summary_all_sizes.csv")
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path)
        s = s[s.generator.isin(gens) & (s["T"] == 2048)]
        for (f, h), g in s.groupby(["forecaster", "H"]):
            if h not in (16, 128):
                continue
            for m in ("mase", "scrps"):
                rows.append({"family_analogue": fam, "forecaster": f, "metric": m, "synthetic_H": int(h),
                             "synthetic_ratio_to_best_classical_gmean": gmean(g[f"{m}_ratio_classical_gmean"]),
                             "synthetic_generators": "+".join(gens), "synthetic_source": f"results/map/{folder}/summary_all_sizes.csv"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/regression_baseline/scores.csv")
    ap.add_argument("--map-dir", default="results/map")
    ap.add_argument("--out", default="results/regression_baseline/analogue_vs_synthetic.csv")
    a = ap.parse_args()
    scores = pd.read_csv(a.scores)
    real = real_side(scores)
    syn = synthetic_side(a.map_dir)
    wide = syn.pivot_table(index=["family_analogue", "forecaster", "metric", "synthetic_generators", "synthetic_source"], columns="synthetic_H",
                           values="synthetic_ratio_to_best_classical_gmean").reset_index()
    wide.columns = [c if not isinstance(c, (int, np.integer)) else f"synthetic_T2048_H{c}" for c in wide.columns]
    out = real.merge(wide, on=["family_analogue", "forecaster", "metric"], how="left")
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
