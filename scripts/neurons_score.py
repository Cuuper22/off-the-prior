"""Summary tables from the per-pole label files of all sizes.

    python scripts/neurons_score.py --runs results/colab_out --corpus-dir results/neurons/corpus --out results/neurons

Reads `<runs>/nn_<size>_<context>/scores_partial.csv` (context = main, short, long, xlong) and writes
  <out>/<size>/poles.parquet        one row per pole of the main-context run (exemplar columns included)
  <out>/<size>/poles_<context>.parquet   same for the other contexts (without the position profile columns)
  <out>/one_job_curves.csv, scaling.csv, heldout_check.csv, pole_pairs.csv, sign_regimes.csv, kinds.csv,
  <out>/label_groups.csv, context_comparison.csv, motif_cohesion.csv
Definitions are in `otp/neurons/score.py`.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from otp.io import provenance, read_json, write_json  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import labels as Lb  # noqa: E402
from otp.neurons import score as Sc  # noqa: E402

SIZES = ("5M", "28M", "123M", "209M")


def find_runs(runs: str) -> dict:
    out = {}
    for d in sorted(glob.glob(os.path.join(runs, "nn_*"))):
        parts = os.path.basename(d).split("_")
        if len(parts) != 3 or not os.path.exists(os.path.join(d, "scores_partial.csv")):
            continue
        out[(parts[1], parts[2])] = d
    return out


def load_corpus_for(meta: dict, corpus_dir: str):
    files = [os.path.join(corpus_dir, f) for f in meta["corpus"].split(",")]
    parts = [C.load_corpus(f) for f in files]
    return parts[0] if len(parts) == 1 else C.merge_corpora([x[0] for x in parts], [x[1] for x in parts])


def save_table(t, path):
    """Parquet with float32 numbers and zstd compression (keeps every file well below the 50 MB limit)."""
    t = t.copy()
    f = t.select_dtypes("float64").columns
    t[f] = t[f].astype("float32")
    t.to_parquet(path, index=False, compression="zstd", compression_level=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/colab_out")
    ap.add_argument("--corpus-dir", default="results/neurons/corpus")
    ap.add_argument("--out", default="results/neurons")
    a = ap.parse_args()
    runs = find_runs(a.runs)
    print("runs:", sorted(runs), flush=True)
    tabs: dict[tuple, pd.DataFrame] = {}
    for (size, ctx), d in runs.items():
        df = pd.read_csv(os.path.join(d, "scores_partial.csv"))
        df = df.sort_values(["layer", "sign", "neuron"]).reset_index(drop=True)
        tabs[(size, ctx)] = Sc.add_derived(df)
        print(size, ctx, len(df), "poles", flush=True)
    sizes = [s for s in SIZES if (s, "main") in tabs]
    main_all = pd.concat([tabs[(s, "main")] for s in sizes], ignore_index=True)
    # cohesion of exemplar lists (needs the corpus used by the run)
    coh_rows = []
    for s in sizes:
        meta = read_json(os.path.join(runs[(s, "main")], "meta.json"))
        c, _ = load_corpus_for(meta, a.corpus_dir)
        tab = Lb.token_table(c)
        t = tabs[(s, "main")]
        coh = Sc.motif_cohesion(t, tab["X"], tab["split"] == 0)
        tabs[(s, "main")] = pd.concat([t, coh], axis=1)
        cls = Sc.klass(tabs[(s, "main")], 0.7, 0.1)
        x = tabs[(s, "main")]
        for sign in ("pos", "neg"):
            m = (x["sign"] == sign).to_numpy() & (cls != "dead")
            for name, mm in (("all_live", m), ("one_job", m & (cls == "one_job")), ("mixed", m & (cls == "mixed")),
                             ("unexplained", m & (cls == "unexplained"))):
                if mm.sum():
                    coh_rows.append({"size": s, "sign": sign, "poles": name, "n": int(mm.sum()),
                                     "coh_top_mean": float(x.loc[mm, "coh_top"].mean()),
                                     "coh_mid_mean": float(x.loc[mm, "coh_mid"].mean()),
                                     "coh_top_median": float(x.loc[mm, "coh_top"].median()),
                                     "coh_mid_median": float(x.loc[mm, "coh_mid"].median())})
    main_all = pd.concat([tabs[(s, "main")] for s in sizes], ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    for s in sizes:
        os.makedirs(os.path.join(a.out, s), exist_ok=True)
        save_table(tabs[(s, "main")], os.path.join(a.out, s, "poles.parquet"))
    for (s, ctx), t in tabs.items():
        if ctx != "main":
            keep = [c_ for c_ in ("size", "layer", "neuron", "sign", "n_active_label", "mean", "std", "max", "label", "label_corr", "r2_cv",
                                  "r2_heldout", "pair_label", "r2_pair", "r2_add", "dead", "E", "ratio", "group") if c_ in t.columns]
            save_table(t[keep], os.path.join(a.out, s, f"poles_{ctx}.parquet"))
    cv = Sc.curves(main_all)
    cv.to_csv(os.path.join(a.out, "one_job_curves.csv"), index=False, float_format="%.5g")
    Sc.scaling(cv).to_csv(os.path.join(a.out, "scaling.csv"), index=False, float_format="%.5g")
    Sc.heldout_check(main_all).to_csv(os.path.join(a.out, "heldout_check.csv"), index=False, float_format="%.5g")
    Sc.pole_pairs(main_all).to_csv(os.path.join(a.out, "pole_pairs.csv"), index=False, float_format="%.5g")
    Sc.sign_regimes(main_all).to_csv(os.path.join(a.out, "sign_regimes.csv"), index=False, float_format="%.5g")
    Sc.kinds(main_all).to_csv(os.path.join(a.out, "kinds.csv"), index=False, float_format="%.5g")
    Sc.group_table(main_all).to_csv(os.path.join(a.out, "label_groups.csv"), index=False, float_format="%.5g")
    pd.DataFrame(coh_rows).to_csv(os.path.join(a.out, "motif_cohesion.csv"), index=False, float_format="%.5g")
    ctx_tabs = {}
    for ctx in ("main", "short", "long", "xlong"):
        ts = [tabs[(s, ctx)] for s in sizes if (s, ctx) in tabs]
        if ts:
            ctx_tabs[ctx] = pd.concat(ts, ignore_index=True)
    if len(ctx_tabs) > 1:
        Sc.context_comparison(ctx_tabs).to_csv(os.path.join(a.out, "context_comparison.csv"), index=False, float_format="%.5g")
    write_json(os.path.join(a.out, "provenance_score.json"), provenance(
        "scripts/neurons_score.py", sizes=sizes, runs={f"{s}_{c_}": read_json(os.path.join(d, "meta.json"))
                                                       for (s, c_), d in runs.items() if os.path.exists(os.path.join(d, "meta.json"))}))
    print("done", flush=True)


if __name__ == "__main__":
    main()
