"""Timestral's error ratio to the best classical model on the real analogue windows, next to the synthetic
family's ratio from the failure-map sweeps.

Real side: for every analogue window, ratio = MASE (or sCRPS) of a Timestral size divided by the lower of the
seasonal naive, ETS and ARIMA values on that window; aggregated per family and per family/dataset as the geometric
mean over windows (the same definition as `mase_ratio_classical_gmean` in the sweep summaries).

Synthetic side: `mase_ratio_classical_gmean` / `scrps_ratio_classical_gmean` rows of
`results/map/<grid>/summary_all_sizes.csv` at context T = 2048, for the generators mapped to each family, taken at
horizon 16 and at horizon 128 (geometric mean over generators and knob values).

Output: `analogue_vs_synthetic.csv` (family and family/dataset rows) and a section "Analogues versus synthetic
families" written into `results/regression_baseline/README.md` between two marker lines (replaced on each run).
Run `scripts/score_regression.py --summarize-only` first (it writes scores.csv and the README).

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


BEGIN = "<!-- analogue-report:begin -->"
END = "<!-- analogue-report:end -->"
SIZE_ORDER = ["timestral-5M", "timestral-28M", "timestral-123M", "timestral-209M"]
FAMILY_ORDER = list(SYNTHETIC)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and not np.isfinite(v)) else (format(v, ".3f") if isinstance(v, float) else str(v))
                                     for v in r) + " |")
    return "\n".join(out)


def readme_section(out: pd.DataFrame) -> str:
    fam = out[out.level == "family"].copy()
    fam["_f"] = fam.family_analogue.map({f: i for i, f in enumerate(FAMILY_ORDER)})
    fam["_s"] = fam.forecaster.map({f: i for i, f in enumerate(SIZE_ORDER)})
    text = [BEGIN, "## Analogues versus synthetic families",
            "",
            "Real analogue windows (the `analogues` set; no oracle exists for them) against the synthetic failure family each stands in for. "
            "For every window, the ratio is a Timestral size's MASE (or sCRPS) divided by the lowest of seasonal naive, ETS and ARIMA on that "
            "window; the table gives the geometric mean of the ratio over the family's windows (below 1: Timestral is lower than the best "
            "classical model). The synthetic columns are the same quantity (`*_ratio_classical_gmean`, geometric mean over the generators listed "
            "and their knob values) read from `results/map/<chaos|tails|regime>/summary_all_sizes.csv` at context 2048 and horizon 16 or 128. "
            "The pairing of families to generators is: delay_feedback to mackey_glass, coupled_oscillators to kuramoto, heavy_tails to "
            "fbm, hawkes, msm_cascade and stable_ar, regime_switches to frequency_jump, markov_ar, mean_jump and variance_jump. "
            "The real windows have their own horizons and periods (see `bench/regression/manifest.csv`), so the columns are placed side by "
            "side, not matched window to window. Per-dataset rows are in `analogue_vs_synthetic.csv`.", ""]
    for m, lab in (("mase", "MASE"), ("scrps", "sCRPS")):
        t = fam[fam.metric == m].sort_values(["_f", "_s"])
        t = t[["family_analogue", "forecaster", "n_windows", "real_ratio_to_best_classical_gmean", "synthetic_T2048_H16", "synthetic_T2048_H128"]]
        t.columns = ["family_analogue", "forecaster", "real windows", f"real {lab} ratio", f"synthetic {lab} ratio, H16", f"synthetic {lab} ratio, H128"]
        text += [f"### {lab} ratio to the best classical model", "", _md(t), ""]
    text.append(END)
    return "\n".join(text) + "\n"


def write_section(readme_path: str, section: str) -> None:
    txt = open(readme_path).read() if os.path.exists(readme_path) else ""
    if BEGIN in txt and END in txt:
        a, b = txt.index(BEGIN), txt.index(END) + len(END)
        txt = txt[:a] + section.rstrip("\n") + txt[b:]
    else:
        txt = txt.rstrip("\n") + "\n\n" + section
    with open(readme_path, "w") as f:
        f.write(txt)


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
    readme = os.path.join(os.path.dirname(a.out), "README.md")
    write_section(readme, readme_section(out))
    print(f"wrote analogue section into {readme}")


if __name__ == "__main__":
    main()
