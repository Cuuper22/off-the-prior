"""Follow-up B of the neuron atlas: how much does the one-job share depend on the corpus, the number of series and
the variable library? Reads `scores_partial.csv` of capture runs (scripts/neurons_run.py) and classifies poles
with the atlas's rule (otp/neurons/score.py): dead if fewer than MIN_ACTIVE active label tokens; E = max(single,
pair, additive) R^2; unexplained if E < floor; one-job if r2_cv / E >= theta; mixed otherwise.

    python scripts/neurons_followup_b.py --runs <dir with run folders> --out results/neurons/followup

Run folders are named <tag>, each with scores_partial.csv and meta.json; `runs.csv` in --runs lists
tag, size, corpus, library, n_series, seed (written by the run script of this follow-up).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.io import provenance, write_json  # noqa: E402
from otp.neurons.labels import MIN_ACTIVE  # noqa: E402


def classify(df: pd.DataFrame, theta: float, floor: float) -> dict:
    dead = df.n_active_label < MIN_ACTIVE
    live = df[~dead]
    E = np.maximum.reduce([live.r2_cv.clip(lower=0).values, live.r2_pair.fillna(0).clip(lower=0).values,
                           live.r2_add.fillna(0).clip(lower=0).values])
    ratio = np.where(E > 0, live.r2_cv.clip(lower=0).values / np.where(E > 0, E, 1), 0.0)
    unexpl = E < floor
    one = (~unexpl) & (ratio >= theta)
    n = len(live)
    return {"n_live": n, "one_job": float(one.mean()) if n else np.nan, "unexplained": float(unexpl.mean()) if n else np.nan,
            "mixed": float(((~unexpl) & ~one).mean()) if n else np.nan, "E_mean": float(E.mean()) if n else np.nan,
            "E_median": float(np.median(E)) if n else np.nan, "r2_label_median": float(np.median(live.r2_cv)) if n else np.nan}


def write_readme(out: pd.DataFrame, out_dir: str) -> None:
    q = out[(out.theta == 0.7) & (out.floor == 0.1)].copy()
    cols = ["tag", "size", "corpus", "library", "n_series", "seed", "n_live", "one_job", "unexplained", "E_mean", "same_label_as_reference"]
    cols = [c for c in cols if c in q.columns]

    def md(d):
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in d[cols].itertuples(index=False):
            lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(lines)

    g = q.groupby(["size", "role", "corpus", "n_series", "library"]).agg(one_job_mean=("one_job", "mean"), one_job_min=("one_job", "min"),
                                                              one_job_max=("one_job", "max"), E_mean=("E_mean", "mean"),
                                                              runs=("tag", "count")).reset_index()
    gl = ["| " + " | ".join(g.columns) + " |", "|" + "---|" * len(g.columns)]
    for r in g.itertuples(index=False):
        gl.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
    text = f"""# Neuron atlas follow-up B: corpus, number of series and variable library

Question: the atlas's one-job share was 4.5% on the main corpus but 24 to 47% on the short, long and xlong corpora
(`../context_comparison.csv`). Is that a property of the neurons, of the variable library, or of the corpus size?

Runs (all with `scripts/neurons_run.py` at commit c4eecd7 on the laptop GPU; plan in `plan.csv`, run folders not
committed): the main corpus with the full library (reference) and with the library common to all four corpora
(`libraries/common.txt`, 82 of the 101 variables); short, long and xlong with the common library; the main corpus
subsampled at random to 1034, 620 and 48 series (the series counts of short, long and xlong), 3 seeds each; random
libraries of 10, 25 and 50 variables drawn from the common library, 3 seeds each. 28M: reference, common library,
1034 and 48 series. Classification as in the atlas (theta 0.7, floor 0.1 below; other thresholds in
`corpus_library_control.csv`), written by `scripts/neurons_followup_b.py`. `same_label_as_reference`: share of poles
whose best label equals the reference run's (same size).

## Summary (theta 0.7, floor 0.1)

{chr(10).join(gl)}

## All runs (theta 0.7, floor 0.1)

{md(q)}
"""
    open(os.path.join(out_dir, "README.md"), "w").write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None, help="run folder whose labels the others are compared with (per size)")
    a = ap.parse_args()
    runs = pd.read_csv(os.path.join(a.runs, "runs.csv"))
    rows, labels = [], {}
    for r in runs.itertuples(index=False):
        f = os.path.join(a.runs, r.tag, "scores_partial.csv")
        if not os.path.exists(f):
            print("missing", f)
            continue
        df = pd.read_csv(f, usecols=["layer", "neuron", "sign", "n_active_label", "label", "r2_cv", "r2_pair", "r2_add"])
        labels[r.tag] = df.set_index(["layer", "neuron", "sign"]).label
        for theta in (0.5, 0.7, 0.9):
            for floor in (0.05, 0.1):
                rows.append({**r._asdict(), "theta": theta, "floor": floor, **classify(df, theta, floor)})
    out = pd.DataFrame(rows)
    # label agreement with the reference run of the same size (full library, main corpus, all series)
    agree = []
    for r in runs.itertuples(index=False):
        ref = runs[(runs["size"] == r.size) & (runs.role == "reference")]
        if r.tag not in labels or ref.empty or ref.tag.iloc[0] not in labels:
            continue
        a_, b_ = labels[r.tag].align(labels[ref.tag.iloc[0]], join="inner")
        ok = a_.notna() & b_.notna()
        agree.append({"tag": r.tag, "same_label_as_reference": float((a_[ok] == b_[ok]).mean()), "n_compared": int(ok.sum())})
    if agree:
        out = out.merge(pd.DataFrame(agree), on="tag", how="left")
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, "corpus_library_control.csv"), index=False)
    write_json(os.path.join(a.out, "provenance_followup_b.json"), provenance("scripts/neurons_followup_b.py", argv=sys.argv))
    write_readme(out, a.out)
    print(out[(out.theta == 0.7) & (out.floor == 0.1)][["tag", "size", "corpus", "library", "n_series", "seed", "n_live", "one_job",
                                                       "unexplained", "E_mean"] + (["same_label_as_reference"] if agree else [])].to_string())


if __name__ == "__main__":
    main()
