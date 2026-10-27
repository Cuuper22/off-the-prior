"""Write a sample of the neuron atlas in the closest format Neuronpedia accepts.

    python scripts/neurons_export_neuronpedia.py --size 5M --poles results/neurons/5M/poles.parquet \
        --corpus results/neurons/corpus/main.npz,results/neurons/corpus/mv.npz --real results/neurons/real/real.npz \
        --out results/neurons/neuronpedia --n-poles 12

Neuronpedia's feature upload (`POST /api/feature/upload-batch`, Python client `neuronpedia.np_feature.Feature`) takes, per
source (one layer of one model) and feature index: `density`, `topLogits` / `bottomLogits` (token strings with values),
`explanations` (text of 1 to 512 characters) and `activations` (parallel lists `tokens` / `values`, at most 1024 entries,
with the optional `quantileMax`, `quantileMin`, `quantileFraction`), at most 128 features per request. See
`results/neurons/README.md` for what was read and what does not fit.

Mapping used here
- model id `timestral-<size>`, layers = number of blocks; two source sets per model, `ts<size>-mlp-pos` and `ts<size>-mlp-neg`
  (so sources are `<layer>-ts<size>-mlp-pos` and `<layer>-ts<size>-mlp-neg`); feature index = neuron index in the MLP.
- one activation entry = a window of up to 9 patch tokens ending at the exemplar token. A token string is a sparkline of the
  patch's 32 values (eight block characters, scaled to the window's range), prefixed with an arrow for forecast tokens
  (the realized next patch). `values` = the pole's activation on each token (recomputed with the plain forward pass).
- `quantile*` fields: top exemplars, mid-range exemplars and real-data exemplars each form one bin.
- the file `patches/<source>.json` carries what the schema has no field for: the 32 raw values of every token; the PNG
  of each exemplar is in `sparklines/`; `poles_extra.json` carries labels and scores.
Nothing is uploaded.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.model import default_device, set_threads  # noqa: E402
from otp.neurons import capture as Cap  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import score as Sc  # noqa: E402

BLOCKS = "▁▂▃▄▅▆▇█"
WINDOW = 9
PATCH = 32
MAX_EXPL = 512


def spark(v: np.ndarray, lo: float, hi: float) -> str:
    q = np.clip(((v - lo) / max(hi - lo, 1e-12) * 7.999).astype(int), 0, 7)
    return "".join(BLOCKS[i] for i in q)


def patch_values(vals: np.ndarray, T: int, t: int) -> np.ndarray:
    """32 raw values of token t of a series row: context patch, or the realized next patch for a forecast token."""
    L = T // PATCH
    s = PATCH * t if t < L else T + PATCH * (t - L)
    return vals[s:s + PATCH]


def acts_of_series(size, c, pos, layer, neuron, sign, dev):
    """Pole activation on every token of series `pos` (rows of its group are run together)."""
    gid = c.get("gid")
    rows = [pos]
    if gid is not None:
        rows = list(np.where(gid == gid[pos])[0])
    x = torch.as_tensor(c["values"][rows, :c["T"]], dtype=torch.float32)
    g = None if len(rows) == 1 else torch.zeros(len(rows), dtype=torch.long)
    _, zs, _ = Cap.forward_with_mlp(size, x, device=dev, keep=(layer,), gid=g)
    z = zs[layer][rows.index(pos), :, neuron].float().cpu().numpy()
    return np.maximum(z if sign == "pos" else -z, 0.0)


def parse(s):
    return [float(x) for x in str(s).split()] if isinstance(s, str) else []


def choose(t: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    t = t.copy()
    t["cls"] = Sc.klass(t, 0.7, 0.1)
    rng = np.random.default_rng(seed)
    picks = []
    oj = t[t["cls"] == "one_job"].sort_values("r2_cv", ascending=False)
    for g, d in oj.groupby("group", sort=False):
        if len(picks) >= n - 4:
            break
        picks.append(d.iloc[0])
    for cls in ("mixed", "unexplained"):
        d = t[t["cls"] == cls]
        for j in rng.permutation(len(d))[:2]:
            picks.append(d.iloc[j])
    return pd.DataFrame(picks).drop_duplicates(["layer", "neuron", "sign"])


def validate(request_body: dict) -> None:
    """The constraints of Neuronpedia's feature upload-batch route (apps/webapp/app/api/feature/upload-batch/route.tsx)."""
    fs = request_body["features"]
    assert 1 <= len(fs) <= 128 and request_body["modelId"] == request_body["modelId"].lower() and request_body["source"] == request_body["source"].lower()
    assert len({f["index"] for f in fs}) == len(fs)
    for f in fs:
        assert 0 <= f["index"] <= 10_000_000 and 0 <= f["density"] <= 1 and len(f["activations"]) >= 1
        for e in f["explanations"]:
            assert 1 <= len(e["text"]) <= 512
        for ac in f["activations"]:
            assert len(ac["tokens"]) == len(ac["values"]) and 1 <= len(ac["tokens"]) <= 1024
            assert 0 <= ac["quantileFraction"] <= 1


def png(path, vals, acts, hit, lo_t):
    fig, ax = plt.subplots(2, 1, figsize=(4.2, 1.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(vals, lw=0.8, color="#2a4d8f")
    ax[0].axvspan(hit * PATCH, (hit + 1) * PATCH, color="#f2a900", alpha=0.35, lw=0)
    ax[0].set_yticks([])
    ax[1].bar(np.arange(len(acts)) * PATCH + PATCH / 2, acts, width=PATCH * 0.8, color="#8f2a4d")
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    for a in ax:
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--poles", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--real", default=None)
    ap.add_argument("--out", default="results/neurons/neuronpedia")
    ap.add_argument("--n-poles", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    set_threads()
    dev = a.device or default_device()
    t = pd.read_parquet(a.poles) if a.poles.endswith(".parquet") else pd.read_csv(a.poles)
    t = t[t["size"] == a.size]
    if "E" not in t.columns:
        t = Sc.add_derived(t)
    parts = [C.load_corpus(f) for f in a.corpus.split(",")]
    c, _ = parts[0] if len(parts) == 1 else C.merge_corpora([p[0] for p in parts], [p[1] for p in parts])
    T = c["T"]
    P = T // PATCH + Cap.N_FC
    n_label_tok = int((c["split"] == 0).sum()) * P
    cr = C.load_corpus(a.real)[0] if a.real else None
    n_blocks, hidden = Cap.layer_dims(a.size, dev)
    sel = choose(t, a.n_poles, a.seed)
    # add the other pole of the first three neurons
    extra = []
    for r in sel.head(3).itertuples():
        o = t[(t.layer == r.layer) & (t.neuron == r.neuron) & (t.sign != r.sign)]
        extra.append(o)
    sel = pd.concat([sel] + extra).drop_duplicates(["layer", "neuron", "sign"]).reset_index(drop=True)
    mid = f"timestral-{a.size.lower()}"
    os.makedirs(a.out, exist_ok=True)
    feats: dict[str, list] = {}
    patches: dict[str, dict] = {}
    extras: dict[str, dict] = {}
    for r in sel.itertuples():
        src = f"{r.layer}-ts{a.size.lower()}-mlp-{r.sign}"
        entries, pat = [], []
        groups = [("top", "top_tok", "top_act", c, 4), ("mid", "mid_tok", "mid_act", c, 2)]
        if cr is not None and isinstance(getattr(r, "real_top_tok", None), str):
            groups.append(("real_top", "real_top_tok", "real_top_act", cr, 2))
        n_act = int(r.n_active_label) if getattr(r, "n_active_label", 0) else 1
        for kind, kt, ka, cc, k in groups:
            toks = [int(x) for x in str(getattr(r, kt)).split()][:k]
            acts = parse(getattr(r, ka))[:k]
            Pk = cc["T"] // PATCH + Cap.N_FC
            tvals = []
            for j, (tid, act) in enumerate(zip(toks, acts)):
                pos, tk = divmod(tid, Pk)
                full = acts_of_series(a.size, cc, pos, int(r.layer), int(r.neuron), r.sign, dev)
                lo_t = max(0, tk - WINDOW + 1)
                idx = list(range(lo_t, tk + 1))
                pv = [patch_values(cc["values"][pos], cc["T"], i) for i in idx]
                lo, hi = float(min(p.min() for p in pv)), float(max(p.max() for p in pv))
                tokens = [("→" if i >= cc["T"] // PATCH else "") + spark(p, lo, hi) for i, p in zip(idx, pv)]
                vals = [float(full[i]) for i in idx]
                tvals.append(vals[-1])
                pth = os.path.join(a.out, "sparklines", a.size, src, f"{r.neuron}_{kind}{j}.png")
                png(pth, np.concatenate(pv), np.asarray(vals), len(idx) - 1, lo_t)
                entries.append({"kind": kind, "tokens": tokens, "values": vals})
                pat.append({"kind": kind, "series_pos": int(pos), "token": int(tk), "stored_activation": float(act),
                            "tokens": [{"t": int(i), "activation": float(v), "values": [round(float(x), 5) for x in p]}
                                       for i, v, p in zip(idx, vals, pv)],
                            "png": os.path.relpath(pth, a.out)})
            ents = [e for e in entries if e["kind"] == kind]
            if ents:
                vmax = max(max(e["values"]) for e in ents)
                vmin = min(max(e["values"]) for e in ents)
                for e in ents:
                    e["quantileMax"], e["quantileMin"] = float(vmax), float(vmin)
                    e["quantileFraction"] = float(min(1.0, len(ents) / n_act))
        txt = (f"{r.label} ({r.group}); cross-validated R2 {r.r2_cv:.3f}, explainable {r.E:.3f}, ratio {r.ratio:.2f}, "
               f"held-out R2 {r.r2_heldout:.3f}; runner-up {r.label_runner_up} {r.r2_runner_up:.3f}; "
               f"other pole of this neuron is the {'negative' if r.sign == 'pos' else 'positive'} pole (same index in "
               f"source {r.layer}-ts{a.size.lower()}-mlp-{'neg' if r.sign == 'pos' else 'pos'})")[:MAX_EXPL]
        feats.setdefault(src, []).append({
            "index": int(r.neuron), "density": float(min(1.0, n_act / max(n_label_tok, 1))), "topLogits": None, "bottomLogits": None,
            "explanations": [{"text": txt, "methodName": None, "modelName": None}],
            "activations": [{k: e[k] for k in ("tokens", "values", "quantileMax", "quantileMin", "quantileFraction") if k in e}
                            for e in entries]})
        patches.setdefault(src, {})[str(int(r.neuron))] = pat
        ex = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in r._asdict().items()
              if k not in ("Index", "top_tok", "top_act", "mid_tok", "mid_act", "real_top_tok", "real_top_act", "real_mid_tok", "real_mid_act")
              and not k.startswith(("pos_mean", "pos_active"))}
        extras.setdefault(src, {})[str(int(r.neuron))] = ex
        print("pole", src, r.neuron, r.label, flush=True)
    write_json(os.path.join(a.out, f"model_{a.size}.json"), {"id": mid, "layers": n_blocks, "displayName": f"Timestral {a.size}",
                                                         "url": "https://huggingface.co/mistralai/Timestral-" + a.size})
    write_json(os.path.join(a.out, f"source_sets_{a.size}.json"), [
        {"name": f"ts{a.size.lower()}-mlp-{s}", "modelId": mid, "description": f"Timestral {a.size} MLP neurons, {n} pole (SwiGLU gate*silu(up), input of w3)"}
        for s, n in (("pos", "positive"), ("neg", "negative"))])
    for src, fs in feats.items():
        validate({"modelId": mid, "source": src, "features": fs})
        write_json(os.path.join(a.out, f"upload_{a.size}_{src}.json"), {"modelId": mid, "source": src, "features": fs})
        write_json(os.path.join(a.out, "patches", f"{a.size}_{src}.json"), patches[src])
        write_json(os.path.join(a.out, "extra", f"{a.size}_{src}.json"), extras[src])
    write_json(os.path.join(a.out, f"provenance_{a.size}.json"), provenance("scripts/neurons_export_neuronpedia.py", size=a.size, n_poles=len(sel),
                                                                               poles=os.path.basename(a.poles), corpus=a.corpus, device=dev))
    print("wrote", a.out, len(sel), "poles")


if __name__ == "__main__":
    main()
