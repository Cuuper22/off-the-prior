"""Activation correlations between poles on a fixed token subsample: across model sizes and across layers.

    python scripts/neurons_match.py --acts-dir <dir with acts_sub_<size>.npz> --runs results/colab_out --out results/colab_out/nn_match

For every size, `acts_sub_<size>.npz` (from `scripts/neurons_run.py --acts-sub N`) holds the signed MLP activations
`z` of every neuron on the same N label-set tokens (the same token ids for all sizes). A pole's activation is
`max(z, 0)` (positive pole) or `max(-z, 0)` (negative pole).

Writes in --out:
  match_sizes.csv   for each pair of sizes (small -> large) and each pole of the small size with at least 30 active
                    subsample tokens: the Pearson correlation with the best-correlated pole of the larger size, that
                    pole's label, and how many poles of the larger size have correlation >= 0.5 and >= 0.7
  upstream.csv      for each pole (layer >= 1) the three earlier-layer poles of the same size with the highest
                    correlation, with their labels and groups
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.model import default_device  # noqa: E402
from otp.neurons import labels as Lb  # noqa: E402

SIZES = ("5M", "28M", "123M", "209M")
PAIRS = (("5M", "28M"), ("28M", "123M"), ("123M", "209M"), ("5M", "209M"))
MIN_ACTIVE_SUB = 30


def load_poles(acts_dir: str, size: str, dev: str):
    """Standardized pole activations (n_tokens, n_poles) fp16 and a table (layer, sign, neuron) of the poles."""
    with np.load(os.path.join(acts_dir, f"acts_sub_{size}.npz")) as z:
        layers = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
        cols, meta = [], []
        for sign in ("pos", "neg"):
            for li in layers:
                a = torch.as_tensor(z[f"L{li}"]).to(dev).float()
                a = torch.relu(a) if sign == "pos" else torch.relu(-a)
                cols.append(a)
                h = a.shape[1]
                meta.append(pd.DataFrame({"layer": li, "sign": sign, "neuron": np.arange(h)}))
    A = torch.cat(cols, 1)
    meta = pd.concat(meta, ignore_index=True)
    n_act = (A > 0).sum(0).cpu().numpy()
    A = A - A.mean(0, keepdim=True)
    sd = A.pow(2).sum(0).sqrt()
    A = A / sd.clamp(min=1e-9)
    A[:, sd < 1e-9] = 0
    meta["n_active_sub"] = n_act
    return A.half(), meta


def topk_corr(Aq: torch.Tensor, Ab: torch.Tensor, k: int, mask_fn=None, chunk: int = 2048):
    """For each column of Aq the top-k correlations against columns of Ab, plus counts above 0.5 and 0.7."""
    nq = Aq.shape[1]
    vals = torch.zeros(nq, k)
    idx = torch.zeros(nq, k, dtype=torch.long)
    c5 = torch.zeros(nq, dtype=torch.long)
    c7 = torch.zeros(nq, dtype=torch.long)
    for s in range(0, nq, chunk):
        C = (Aq[:, s:s + chunk].T.float() @ Ab.float())
        if mask_fn is not None:
            C = mask_fn(C, s)
        v, i = C.topk(k, dim=1)
        vals[s:s + chunk], idx[s:s + chunk] = v.cpu(), i.cpu()
        c5[s:s + chunk] = (C >= 0.5).sum(1).cpu()
        c7[s:s + chunk] = (C >= 0.7).sum(1).cpu()
    return vals.numpy(), idx.numpy(), c5.numpy(), c7.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--runs", default="results/colab_out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="*", default=list(SIZES))
    a = ap.parse_args()
    dev = default_device()
    os.makedirs(a.out, exist_ok=True)
    poles = {}
    tabs = {}
    for s in a.sizes:
        if not os.path.exists(os.path.join(a.acts_dir, f"acts_sub_{s}.npz")):
            continue
        A, meta = load_poles(a.acts_dir, s, dev)
        t = pd.read_csv(os.path.join(a.runs, f"nn_{s}_main", "scores_partial.csv"),
                        usecols=["layer", "neuron", "sign", "label", "r2_cv", "dead"])
        t["group"] = t["label"].map(lambda x: Lb.GROUP_OF.get(x, "dead"))
        meta = meta.merge(t, on=["layer", "sign", "neuron"], how="left")
        poles[s], tabs[s] = A, meta
        print(s, A.shape, flush=True)
    rows = []
    for small, big in PAIRS:
        if small not in poles or big not in poles:
            continue
        ms, mb = tabs[small], tabs[big]
        live = (ms["n_active_sub"] >= MIN_ACTIVE_SUB).to_numpy()
        mask_big = (mb["n_active_sub"] >= MIN_ACTIVE_SUB).to_numpy()
        Ab = poles[big][:, torch.as_tensor(mask_big)]
        mb_live = mb[mask_big].reset_index(drop=True)
        sel = np.where(live)[0]
        v, i, c5, c7 = topk_corr(poles[small][:, torch.as_tensor(live)], Ab, 2)
        d = ms.iloc[sel][["layer", "sign", "neuron", "label", "group", "r2_cv"]].reset_index(drop=True)
        d.columns = ["layer", "sign", "neuron", "label", "group", "r2_cv"]
        d.insert(0, "large", big)
        d.insert(0, "small", small)
        d["best_corr"], d["second_corr"] = v[:, 0], v[:, 1]
        d["n_ge50"], d["n_ge70"] = c5, c7
        bl = mb_live.iloc[i[:, 0]].reset_index(drop=True)
        d["match_layer"], d["match_sign"], d["match_neuron"] = bl["layer"], bl["sign"], bl["neuron"]
        d["match_label"], d["match_group"], d["match_r2_cv"] = bl["label"], bl["group"], bl["r2_cv"]
        rows.append(d)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(os.path.join(a.out, "match_sizes.csv"), index=False, float_format="%.4g")
    up = []
    for s, A in poles.items():
        m = tabs[s]
        lay = torch.as_tensor(m["layer"].to_numpy(), device=A.device)
        live = (m["n_active_sub"] >= MIN_ACTIVE_SUB).to_numpy() & (m["layer"] >= 1).to_numpy()
        sel = np.where(live)[0]
        selt = torch.as_tensor(sel, device=A.device)

        def mask_fn(C, s0, sel=sel, lay=lay):
            q = torch.as_tensor(sel[s0:s0 + C.shape[0]], device=C.device)
            C[lay[q][:, None] <= lay[None, :]] = -2.0                    # earlier layers only
            return C

        v, i, _, _ = topk_corr(A[:, selt], A, 3, mask_fn)
        d = m.iloc[sel][["layer", "sign", "neuron", "label", "group", "r2_cv"]].reset_index(drop=True)
        d.insert(0, "size", s)
        for r in range(3):
            src = m.iloc[i[:, r]].reset_index(drop=True)
            d[f"up{r}_corr"], d[f"up{r}_layer"], d[f"up{r}_sign"], d[f"up{r}_neuron"] = v[:, r], src["layer"], src["sign"], src["neuron"]
            d[f"up{r}_label"], d[f"up{r}_group"] = src["label"], src["group"]
        up.append(d)
    if up:
        pd.concat(up, ignore_index=True).to_csv(os.path.join(a.out, "upstream.csv"), index=False, float_format="%.4g")
    write_json(os.path.join(a.out, "meta.json"), provenance("scripts/neurons_match.py", sizes=list(poles), min_active_sub=MIN_ACTIVE_SUB))
    print("done", flush=True)


if __name__ == "__main__":
    main()
