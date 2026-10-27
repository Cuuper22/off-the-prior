"""Steering check for a stratified sample of poles (pre-registered in results/neurons/steering_preregistration.md).

    python scripts/neurons_steer.py --size 209M --runs results/colab_out --corpus results/neurons/corpus/main.npz \
        --out results/colab_out/nn_steer

Reads the pole table of the main-context run (`<runs>/nn_<size>_main/scores_partial.csv`), draws the sample described
in the pre-registration, scales each pole by 0, 0.5, 2 and 4 on the tokens where it is active, and writes
`steering_results_<size>.csv` (one row per steered pole and judged quantity) and `steering_effects_<size>.csv`
(mean effect of every scale on every quantity, for the downstream check).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from otp.io import provenance, write_json  # noqa: E402
from otp.model import default_device, set_threads  # noqa: E402
from otp.neurons import capture as Cap  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import labels as Lb  # noqa: E402
from otp.neurons import score as Sc  # noqa: E402
from otp.neurons import steer as St  # noqa: E402
from otp.neurons import variables as V  # noqa: E402

CTX = 1024
SCALES = St.SCALES


def test_sets(c: dict):
    """Held-out mixed set and held-out seasonal set (period 8..64): contexts and periods."""
    held = np.where(c["split"] == 1)[0]
    sc = {n: i for i, n in enumerate(V.SERIES_COLS)}
    rng = np.random.default_rng(St.SAMPLE_SEED)
    mixed = np.sort(rng.choice(held, min(St.N_MIXED, len(held)), replace=False))
    per = 2.0 ** c["sv"][:, sc["s_log2_period"]]
    seas = held[(c["sv"][held, sc["s_has_season"]] == 1) & (per[held] >= 8) & (per[held] <= 64)]
    seas = np.sort(rng.choice(seas, min(St.N_SEASONAL, len(seas)), replace=False))
    return ({"ctx": c["values"][mixed, :CTX], "period": None, "ids": mixed},
            {"ctx": c["values"][seas, :CTX], "period": per[seas].astype(float), "ids": seas})


def draw_sample(t: pd.DataFrame, size: str) -> pd.DataFrame:
    """Labeled sample (cells), complements and controls. t: add_derived pole table of one size."""
    nl = int(t["layer"].max()) + 1
    t = t.copy()
    t["cls"] = Sc.klass(t, St.THETA_SAMPLE, St.FLOOR_SAMPLE)
    t["third"] = t["layer"] * 3 // nl
    t["qgroup"] = t["label"].map(lambda x: St.group_of_label(x) if x in St.PREREG else None)
    rows = []
    rng = np.random.default_rng([St.SAMPLE_SEED, int.from_bytes(size.encode(), 'little') % 1000])
    cand = t[(t["cls"] == "one_job") & t["qgroup"].notna()]
    for (third, qg), d in cand.groupby(["third", "qgroup"]):
        ix = rng.permutation(len(d))[:St.PER_CELL]
        for k, j in enumerate(ix):
            r = d.iloc[j]
            rows.append({"layer": int(r["layer"]), "neuron": int(r["neuron"]), "sign": r["sign"], "role": "labeled",
                         "cell": f"{third}/{qg}", "k": k})
            if k < St.N_COMPLEMENT:
                rows.append({"layer": int(r["layer"]), "neuron": int(r["neuron"]), "sign": "neg" if r["sign"] == "pos" else "pos",
                             "role": "complement", "cell": f"{third}/{qg}", "k": k})
    live = t[t["cls"] != "dead"]
    for j in rng.permutation(len(live))[:St.N_CONTROL]:
        r = live.iloc[j]
        rows.append({"layer": int(r["layer"]), "neuron": int(r["neuron"]), "sign": r["sign"], "role": "control", "cell": "", "k": -1})
    s = pd.DataFrame(rows).drop_duplicates(["layer", "neuron", "sign", "role"])
    return s.merge(t, on=["layer", "neuron", "sign"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--runs", default="results/colab_out")
    ap.add_argument("--corpus", default="results/neurons/corpus/main.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-poles", type=int, default=0, help="steer only the first n sampled poles (tests)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    set_threads()
    dev = a.device or default_device()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    c, _ = C.load_corpus(a.corpus)
    t = Sc.add_derived(pd.read_csv(os.path.join(a.runs, f"nn_{a.size}_main", "scores_partial.csv")))
    samp = draw_sample(t, a.size)
    if a.max_poles:
        samp = samp.iloc[:a.max_poles]
    print(f"{a.size}: {len(samp)} poles ({samp['role'].value_counts().to_dict()})", flush=True)
    sets = dict(zip(("mixed", "seasonal"), test_sets(c)))
    base, basem = {}, {}
    for name, s in sets.items():
        base[name] = St.run_conditions(a.size, s["ctx"], [[]], device=dev)[0]
        basem[name] = St.measures(s["ctx"], base[name], s["period"])
    rng = np.random.default_rng(St.SAMPLE_SEED + 1)
    res_rows, eff_rows = [], []
    for ri, (_, r) in enumerate(samp.iterrows()):
        lab = r["label"]
        pred = St.PREREG.get(lab)
        seasonal_pole = pred is not None and pred[0] in ("amp", "phase") and r["role"] != "control"
        sname = "seasonal" if seasonal_pole else "mixed"
        s = sets[sname]
        sg = 1 if r["sign"] == "pos" else -1
        conds = [[(int(r["layer"]), int(r["neuron"]), sg, float(sc))] for sc in SCALES]
        q = St.run_conditions(a.size, s["ctx"], conds, device=dev)
        eff = [St.effects(basem[sname], St.measures(s["ctx"], q[i], s["period"])) for i in range(len(SCALES))]
        key = {"size": a.size, "layer": int(r["layer"]), "neuron": int(r["neuron"]), "sign": r["sign"], "role": r["role"],
               "cell": r["cell"], "label": lab, "group": r["group"], "r2_cv": r["r2_cv"], "E": r["E"], "ratio": r["ratio"],
               "label_corr": r["label_corr"], "test_set": sname, "n_series": len(s["ctx"])}
        for sc_i, sc in enumerate(SCALES):
            for qn in eff[sc_i]:
                eff_rows.append({**key, "scale": sc, "quantity": qn, "mean_effect": float(
                    St._circ_mean(eff[sc_i][qn]) if qn == "phase" else eff[sc_i][qn].mean())})
        sign_corr = float(r["label_corr"]) if np.isfinite(r["label_corr"]) else 1.0
        if r["role"] == "control":
            for qn in ("level", "slope", "spread"):
                sc_ = float(rng.choice([-1.0, 1.0]))
                j = St.judge(eff, qn, "+", sc_, seed=ri)
                res_rows.append({**key, "quantity": qn, "direction": "+", "random_sign": sc_, **{k: v for k, v in j.items() if k != "quantity" and k != "e"},
                                 **{f"e{i}": v for i, v in enumerate(j["e"])}})
        elif pred is not None:
            j = St.judge(eff, pred[0], pred[1], sign_corr, seed=ri)
            res_rows.append({**key, "quantity": j["quantity"], "predicted": pred[0], "direction": pred[1], **{k: v for k, v in j.items() if k not in ("quantity", "e")},
                             **{f"e{i}": v for i, v in enumerate(j["e"])}})
        else:
            res_rows.append({**key, "quantity": "", "predicted": "", "direction": "", "match": np.nan})
        if ri % 20 == 0:
            print(f"[{time.time() - t0:.0f}s] {ri + 1}/{len(samp)}", flush=True)
            pd.DataFrame(res_rows).to_csv(os.path.join(a.out, f"steering_results_{a.size}.csv"), index=False, float_format="%.5g")
    pd.DataFrame(res_rows).to_csv(os.path.join(a.out, f"steering_results_{a.size}.csv"), index=False, float_format="%.5g")
    pd.DataFrame(eff_rows).to_csv(os.path.join(a.out, f"steering_effects_{a.size}.csv"), index=False, float_format="%.5g")
    write_json(os.path.join(a.out, f"meta_{a.size}.json"), provenance(
        "scripts/neurons_steer.py", size=a.size, n_poles=len(samp), device=dev, wall_s=round(time.time() - t0, 1),
        scales=list(SCALES), n_mixed=len(sets["mixed"]["ctx"]), n_seasonal=len(sets["seasonal"]["ctx"])))
    print("done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
