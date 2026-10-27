"""Compact JSON of the atlas for a neuron browser: one file per size and layer.

    python scripts/neurons_export_browser.py --poles results/neurons --corpus results/neurons/corpus/main.npz,results/neurons/corpus/mv.npz \
        --real results/neurons/real/real.npz --steering results/neurons/steering_results.csv --out results/neurons/browser

`<out>/index.json` describes the sizes, layers, field order and decoding of the exemplar snippets;
`<out>/<size>/layer_<l>.json` has one row per pole (field order in `fields`). A snippet is the 32 raw values of one
patch, scaled to 0..15 between its own minimum and maximum (32 hexadecimal digits, one per value), with the minimum and maximum and the
pole's activation on that token. Every file is under 50 MB.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import json  # noqa: E402

from otp.io import provenance  # noqa: E402
from otp.neurons import capture as Cap  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import score as Sc  # noqa: E402

PATCH = 32


def write_json(path, obj):
    """Compact JSON (no indentation) so that the shards stay small; `otp.io.write_json` indents."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(json.JSONEncoder(separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode(obj))


FIELDS = ["neuron", "sign", "label", "r2", "E", "ratio", "heldout", "corr_sign", "pair_label", "class", "regime_mass",
          "transfer_corr", "steer", "top", "mid", "real"]


def snippet(vals, T, tid, P, act):
    pos, t = divmod(int(tid), P)
    L = T // PATCH
    s = PATCH * t if t < L else T + PATCH * (t - L)
    v = np.asarray(vals[pos][s:s + PATCH], float)
    lo, hi = float(v.min()), float(v.max())
    q = np.clip(np.rint((v - lo) / max(hi - lo, 1e-12) * 15), 0, 15).astype(np.uint8)
    return [pos, t, round(float(act), 4), "".join("0123456789abcdef"[i] for i in q), float(f"{lo:.4g}"), float(f"{hi:.4g}")]


def ex_list(row, key_tok, key_act, vals, T, k):
    toks, acts = row.get(key_tok), row.get(key_act)
    if not isinstance(toks, str) or not isinstance(acts, str):
        return []
    P = T // PATCH + Cap.N_FC
    return [snippet(vals, T, int(t), P, float(a)) for t, a in zip(toks.split()[:k], acts.split()[:k])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poles", default="results/neurons", help="folder with <size>/poles.parquet")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--real", default=None)
    ap.add_argument("--steering", default=None)
    ap.add_argument("--out", default="results/neurons/browser")
    ap.add_argument("--sizes", nargs="*", default=["5M", "28M", "123M", "209M"])
    ap.add_argument("--k-top", type=int, default=3)
    ap.add_argument("--k-mid", type=int, default=1)
    ap.add_argument("--k-real", type=int, default=1)
    a = ap.parse_args()
    parts = [C.load_corpus(f) for f in a.corpus.split(",")]
    c, _ = parts[0] if len(parts) == 1 else C.merge_corpora([p[0] for p in parts], [p[1] for p in parts])
    cr = C.load_corpus(a.real)[0] if a.real else None
    st = pd.read_csv(a.steering) if a.steering and os.path.exists(a.steering) else None
    index = {"fields": FIELDS, "snippet": "[series_position, token, activation, 32 hexadecimal digits (patch scaled 0..15, one digit per value), min, max]",
             "class": {"o": "one_job", "m": "mixed", "u": "unexplained", "d": "dead"}, "theta": 0.7, "floor": 0.1,
             "sizes": {}, "provenance": provenance("scripts/neurons_export_browser.py")}
    os.makedirs(a.out, exist_ok=True)
    for size in a.sizes:
        f = os.path.join(a.poles, size, "poles.parquet")
        if not os.path.exists(f):
            continue
        t = pd.read_parquet(f)
        t["cls"] = Sc.klass(t, 0.7, 0.1)
        stm = {}
        if st is not None:
            for r in st[st["size"] == size].itertuples():
                if isinstance(r.quantity, str) and r.quantity and r.role != "control":
                    stm[(r.layer, r.neuron, r.sign)] = [r.quantity, int(bool(r.match)), round(float(r.rho), 3)]
        labels = sorted(set(t["label"]) | set(t["pair_label"]))
        lix = {l: i for i, l in enumerate(labels)}
        n_layers = int(t["layer"].max()) + 1
        for li in range(n_layers):
            d = t[t["layer"] == li].sort_values(["neuron", "sign"], ascending=[True, False])
            rows = []
            for r in d.to_dict("records"):
                mass = ([r["mass_g1u1"], r["mass_g0u0"]] if r["sign"] == "pos" else [r["mass_g1u0"], r["mass_g0u1"]])
                rows.append([int(r["neuron"]), 0 if r["sign"] == "pos" else 1, lix[r["label"]], round(r["r2_cv"], 4) if np.isfinite(r["r2_cv"]) else None,
                             round(r["E"], 4) if np.isfinite(r["E"]) else None, round(r["ratio"], 3) if np.isfinite(r["ratio"]) else None,
                             round(r["r2_heldout"], 4) if np.isfinite(r["r2_heldout"]) else None,
                             (1 if r["label_corr"] >= 0 else -1) if np.isfinite(r["label_corr"]) else 0, lix[r["pair_label"]],
                             {"one_job": "o", "mixed": "m", "unexplained": "u", "dead": "d"}[r["cls"]],
                             [round(float(x), 2) for x in mass], round(r["tr_corr"], 3) if "tr_corr" in r and np.isfinite(r["tr_corr"]) else None,
                             stm.get((li, int(r["neuron"]), r["sign"])),
                             ex_list(r, "top_tok", "top_act", c["values"], c["T"], a.k_top),
                             ex_list(r, "mid_tok", "mid_act", c["values"], c["T"], a.k_mid),
                             ex_list(r, "real_top_tok", "real_top_act", cr["values"], cr["T"], a.k_real) if cr is not None else []])
            out = os.path.join(a.out, size, f"layer_{li}.json")
            write_json(out, {"size": size, "layer": li, "labels": labels, "poles": rows})
            assert os.path.getsize(out) < 50e6, out
        index["sizes"][size] = {"layers": n_layers, "hidden": int(t["neuron"].max()) + 1, "poles": len(t)}
        print(size, len(t), "poles", flush=True)
    write_json(os.path.join(a.out, "index.json"), index)


if __name__ == "__main__":
    main()
