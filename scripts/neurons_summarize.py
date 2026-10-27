"""Collect the steering, failure-link, cross-size, head and squeeze outputs into `results/neurons/` and summarize them.

    python scripts/neurons_summarize.py --runs results/colab_out --out results/neurons

Inputs (whatever exists): `<runs>/nn_steer/steering_{results,effects}_<size>.csv`, `<runs>/nn_failure/*`, `<runs>/nn_match/*`,
`<runs>/nn_heads_<size>/*`, `<runs>/nn_squeeze_w*_main/scores_partial.csv`, `<out>/<size>/poles.parquet` (from
`scripts/neurons_score.py`). Writes steering_results.csv, steering_effects.csv, steering_summary.csv, steering_checks.csv,
failure_link.csv, failure_configs.csv, failure_samples.csv, transfer.csv, transfer_by_group.csv, scale_match.csv,
heads_time.csv, heads_variate.csv, heads_summary.csv, squeeze_summary.csv, and adds the `up*` columns to each poles.parquet.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.neurons import score as Sc  # noqa: E402
from otp.neurons import summary as Sm  # noqa: E402

SIZES = ("5M", "28M", "123M", "209M")


def cat(pattern):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/colab_out")
    ap.add_argument("--out", default="results/neurons")
    a = ap.parse_args()
    o = lambda f: os.path.join(a.out, f)          # noqa: E731
    fk = dict(index=False, float_format="%.5g")
    poles = {s: pd.read_parquet(o(f"{s}/poles.parquet")) for s in SIZES if os.path.exists(o(f"{s}/poles.parquet"))}
    allp = pd.concat(poles.values(), ignore_index=True)
    done = []
    res = cat(os.path.join(a.runs, "nn_steer", "steering_results_*.csv"))
    eff = cat(os.path.join(a.runs, "nn_steer", "steering_effects_*.csv"))
    up = cat(os.path.join(a.runs, "nn_match", "upstream.csv"))
    mt = cat(os.path.join(a.runs, "nn_match", "match_sizes.csv"))
    if res is not None and eff is not None:
        res.to_csv(o("steering_results.csv"), **fk)
        eff.to_csv(o("steering_effects.csv"), **fk)
        Sm.steering_summary(res).to_csv(o("steering_summary.csv"), **fk)
        Sm.steering_checks(res, eff, up, allp).to_csv(o("steering_checks.csv"), **fk)
        done.append("steering")
    for src, dst in (("failure_link_*.csv", "failure_link.csv"), ("failure_configs_*.csv", "failure_configs.csv"),
                     ("failure_samples_*.csv", "failure_samples.csv")):
        d = cat(os.path.join(a.runs, "nn_failure", src))
        if d is not None:
            d.to_csv(o(dst), **fk)
            done.append(dst)
    if mt is not None:
        Sm.scale_match(mt, allp).to_csv(o("scale_match.csv"), **fk)
        done.append("scale_match")
    tr = Sm.transfer(allp)
    if len(tr):
        tr.to_csv(o("transfer.csv"), **fk)
        Sm.transfer_by_group(allp).to_csv(o("transfer_by_group.csv"), **fk)
        done.append("transfer")
    ht = cat(os.path.join(a.runs, "nn_heads_*", "heads_time.csv"))
    if ht is not None:
        ht.to_csv(o("heads_time.csv"), **fk)
        Sm.heads_summary(ht).to_csv(o("heads_summary.csv"), **fk)
        hv = cat(os.path.join(a.runs, "nn_heads_*", "heads_variate.csv"))
        if hv is not None:
            hv.to_csv(o("heads_variate.csv"), **fk)
        done.append("heads")
    sq = {}
    for f in sorted(glob.glob(os.path.join(a.runs, "nn_squeeze_w*_main", "scores_partial.csv"))):
        name = os.path.basename(os.path.dirname(f)).split("_")[2]
        sq[name] = Sc.add_derived(pd.read_csv(f))
    if sq:
        Sm.squeeze(sq).to_csv(o("squeeze_summary.csv"), **fk)
        done.append("squeeze")
    if up is not None:
        for s, t in poles.items():
            u = up[up["size"] == s].drop(columns=["label", "group", "r2_cv"])
            t = t.drop(columns=[c for c in t.columns if c.startswith("up")]).merge(u, on=["size", "layer", "sign", "neuron"], how="left")
            f32 = t.select_dtypes("float64").columns
            t[f32] = t[f32].astype("float32")
            t.to_parquet(o(f"{s}/poles.parquet"), index=False, compression="zstd", compression_level=15)
        done.append("upstream columns")
    write_json(o("provenance_summarize.json"), provenance("scripts/neurons_summarize.py", produced=done))
    print("done:", done)


if __name__ == "__main__":
    main()
