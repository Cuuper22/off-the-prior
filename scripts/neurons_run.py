"""Capture activations of every neuron of one Timestral size on an audit corpus and label every pole.

    python scripts/neurons_run.py --size 5M --corpus results/neurons/corpus/main.npz[,results/neurons/corpus/mv.npz] --out results/colab_out/nn_5M_main \
        [--real results/neurons/real/real.npz] [--layers-per-chunk 4] [--resume]

Writes `scores_partial.csv` (one row per pole, appended after every chunk of layers; with `--resume` layers already in
the file are skipped) and `meta.json`. The row format is described in `otp/neurons/labels.py` (`summarize_sign`);
exemplar columns hold corpus token ids (series position * tokens-per-series + token) and activations.
"""
import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.model import default_device, set_threads  # noqa: E402
from otp.neurons import capture as Cap  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import labels as Lb  # noqa: E402


def fmt(a, nd=4):
    return " ".join(f"{x:.{nd}g}" for x in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real", default=None)
    ap.add_argument("--layers-per-chunk", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-series", type=int, default=0, help="use only the first n series (tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--subsample", type=int, default=0, help="use a random subset of n series (seeded; for token-count controls)")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--keep-vars", default=None,
                    help="file with a comma-separated list of library variables to keep; the others are treated as undefined")
    ap.add_argument("--acts-dir", default=None, help="where acts_sub_<size>.npz goes (default: --out)")
    ap.add_argument("--acts-sub", type=int, default=0,
                    help="also save the signed activations of every neuron on this many fixed tokens (seeded, same for all sizes)")
    a = ap.parse_args()
    set_threads()
    dev = a.device or default_device()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    parts = [C.load_corpus(f) for f in a.corpus.split(",")]
    c, specs = (parts[0] if len(parts) == 1 else C.merge_corpora([x[0] for x in parts], [x[1] for x in parts]))
    if a.max_series:
        keep = np.arange(min(a.max_series, len(specs)))
        c = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(specs) else v)
             for k, v in c.items()}
    if a.subsample and a.subsample < len(specs):
        keep = np.sort(np.random.default_rng(a.subsample_seed).choice(len(specs), a.subsample, replace=False))
        c = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(specs) else v)
             for k, v in c.items()}
    T = c["T"]
    tab = Lb.token_table(c)
    if a.keep_vars:
        from otp.neurons import variables as V
        keep_names = [x.strip() for x in open(a.keep_vars).read().split(",") if x.strip()]
        unknown = [x for x in keep_names if x not in V.VAR_INDEX]
        assert not unknown, f"unknown variables: {unknown}"
        drop = [i for i, n in enumerate(V.LIBRARY) if n not in set(keep_names)]
        tab["X"][:, drop] = np.nan
    prep = Lb.prepare(tab, dev)
    real = None
    if a.real:
        cr, _ = C.load_corpus(a.real)
        tr = Lb.token_table(cr)
        real_codes = torch.as_tensor(Lb.apply_bins(tr["X"], prep["edges"]))
        ev_path = os.path.join(os.path.dirname(a.real), "real_events.npy")
        ev = np.load(ev_path).reshape(-1, 3) if os.path.exists(ev_path) else None
        real = {"codes": real_codes, "values": cr["values"], "T": cr["T"], "events": ev}
    n_blocks, hidden = Cap.layer_dims(a.size, dev)
    part = os.path.join(a.out, "scores_partial.csv")
    done = set()
    if a.resume and os.path.exists(part):
        done = set(pd.read_csv(part, usecols=["layer"]).layer.unique().tolist())
        print("resume: layers already done:", sorted(done), flush=True)
    todo = [li for li in range(n_blocks) if li not in done]
    tok = T // 32 + Cap.N_FC
    info = {}
    sub_ix = None
    sub = {}
    if a.acts_sub:
        n_tok = c["values"].shape[0] * tok
        label_tok = np.where(np.repeat(c["split"], tok) == 0)[0]
        sub_ix = np.sort(np.random.default_rng(12345).choice(label_tok, min(a.acts_sub, len(label_tok)), replace=False))
        sub_path = os.path.join(a.acts_dir or a.out, f"acts_sub_{a.size}.npz")
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        if a.resume and os.path.exists(sub_path):
            with np.load(sub_path) as zz:
                sub = {k: zz[k] for k in zz.files}
    for i in range(0, len(todo), a.layers_per_chunk):
        chunk = tuple(todo[i:i + a.layers_per_chunk])
        print(f"[{time.time() - t0:.0f}s] capture layers {chunk}", flush=True)
        z, code, info = Cap.capture_corpus(a.size, c["values"], T, batch=a.batch, device=dev, layers=chunk,
                                              gid=c["gid"] if len(parts) > 1 else None)
        zr = None
        if real is not None:
            zr, _, _ = Cap.capture_corpus(a.size, real["values"], real["T"], batch=a.batch, device=dev, layers=chunk)
        if sub_ix is not None:
            for li in chunk:
                sub[f"L{li}"] = z[li][torch.as_tensor(sub_ix)].numpy()
            sub["token_ids"] = sub_ix
            np.savez(sub_path, **sub)
        for li in chunk:
            rl = None if real is None else {"Z": zr[li], "codes": real["codes"], "events": real["events"]}
            res = Lb.label_layer(prep, z[li], code[li], real=rl)
            rows = []
            for sign in ("pos", "neg"):
                o = res[sign]
                row = Lb.summarize_sign(o, li, sign, a.size)
                df = pd.DataFrame(row)
                gid = prep["order"]
                for name, ix_key, act_key in (("top", "top_idx", "top_act"), ("mid", "mid_idx", "mid_act")):
                    ids = gid[o[ix_key]]
                    df[f"{name}_tok"] = [" ".join(map(str, r)) for r in ids]
                    df[f"{name}_act"] = [fmt(r) for r in o[act_key]]
                if "real_top_idx" in o:
                    for name in ("top", "mid"):
                        df[f"real_{name}_tok"] = [" ".join(map(str, r)) for r in o[f"real_{name}_idx"]]
                        df[f"real_{name}_act"] = [fmt(r) for r in o[f"real_{name}_act"]]
                    df["real_n_active"] = o["real_n_active"]
                rows.append(df)
            out = pd.concat(rows, ignore_index=True)
            out.to_csv(part, mode="a", header=not os.path.exists(part), index=False, float_format="%.5g")
            print(f"[{time.time() - t0:.0f}s] layer {li} labeled ({len(out)} poles)", flush=True)
            del res
        del z, code, zr
        gc.collect()
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    meta = provenance("scripts/neurons_run.py", size=a.size, corpus=",".join(os.path.basename(f) for f in a.corpus.split(",")), T=T,
                      n_series=int(c["values"].shape[0]), tokens_per_series=tok, hidden=hidden, blocks=n_blocks,
                      device=dev, wall_s=round(time.time() - t0, 1), capture=info,
                      n_label_series=int((c["split"] == 0).sum()), n_heldout_series=int((c["split"] == 1).sum()))
    write_json(os.path.join(a.out, "meta.json"), meta)
    print("done", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
