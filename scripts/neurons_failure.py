"""Tie the pole labels to the failure map: which poles are active where Timestral fails, and what does damping them do?

    python scripts/neurons_failure.py --size 209M --runs results/colab_out --out results/colab_out/nn_failure

Contexts: samples of the chaos family (same generator code, configs and seeds as `results/map/chaos`: the last 2048 steps
of an 8192-step context, horizon 128, 8 seeds per config; the oracle uses 200 paths and the realized future is the first
128 steps of a path simulated for 128 steps, so futures can differ from the map's). A config is a *failing* config for a size when Timestral's MASE is at least 1.5 times the
oracle's and the method of analogues beats Timestral (geometric means over its seeds); a *reference* config is
any other config of the same three generators whose Timestral MASE is within 1.5 times the oracle's, or one of 64
held-out in-prior seasonal series (period 8..64, first 2048 steps of the long audit corpus). Generators: Mackey-Glass, Rossler, Kuramoto.

Analyses per size:
  activation   mean activation of period- and phase-labeled one-job poles on failing and on reference contexts
  elevated     poles whose mean activation (last 16 context patches and the forecast tokens) is higher on failing than on
               reference contexts by Cohen's d >= 0.8; their label groups against the base rate of the groups
  ablate       scale the elevated period/phase poles (all at once) by 0 and 0.5 on failing contexts and compare, before
               and after: MASE to the realized future, mean absolute distance of the median to the oracle's median and to
               the analogue forecast (in naive-scale units), correlation with the seasonal template; the same for the same
               number of random poles of the same layers (control)
  single       each of the 10 elevated period/phase poles with the largest d, ablated alone
"""
import argparse
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analog_test import analog_forecast  # noqa: E402
from otp.generators.base import make_sample  # noqa: E402
from otp.grids import chaos as G  # noqa: E402
from otp.io import provenance, write_json  # noqa: E402
from otp.mechanism import dominant_period, template_forecast  # noqa: E402
from otp.metrics import naive_scale  # noqa: E402
from otp.model import MEDIAN_INDEX, default_device, set_threads  # noqa: E402
from otp.neurons import capture as Cap  # noqa: E402
from otp.neurons import corpus as C  # noqa: E402
from otp.neurons import variables as V  # noqa: E402
from otp.neurons import score as Sc  # noqa: E402
from otp.neurons import steer as St  # noqa: E402

T, H, SEEDS, N_PATHS = 2048, 128, 8, 200
T_BUILD = 8192      # the map builds samples with an 8192-step context and cuts the last T; the context is the same here
GENS = ("mackey_glass", "rossler", "kuramoto")
PERIODIC_GROUPS = ("period", "phase")
D_MIN = 0.8
TAIL = 16


def build_samples():
    out = []
    for cid, factory in G.configs():
        gen = cid.split("/")[0]
        if gen not in GENS:
            continue
        for seed in range(SEEDS):
            s = make_sample(factory(seed), T_BUILD, H, seed=zlib.crc32(f"{cid}/{seed}".encode()) % (2 ** 31), n_paths=N_PATHS,
                            sample_id=f"{cid}/s{seed}")
            s.meta["config_id"] = cid
            out.append(s)
    return out


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def forecasts(size, ctx, conds, dev):
    """Median forecasts (n_cond, N, H) of the plain forward pass for each condition."""
    q = St.run_conditions(size, ctx, conds, device=dev, max_rows=512)
    return q[..., MEDIAN_INDEX]


def per_sample_metrics(med, fut, orc, ana, tmpl, scale):
    mae = lambda f: np.abs(f - fut).mean(1) / scale                      # noqa: E731
    return {"mase": mae(med), "dist_oracle": np.abs(med - orc).mean(1) / scale,
            "dist_analog": np.abs(med - ana).mean(1) / scale,
            "corr_template": np.array([corr(m, t) for m, t in zip(med, tmpl)]),
            "corr_oracle": np.array([corr(m, o) for m, o in zip(med, orc)])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--runs", default="results/colab_out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-corpus", default="results/neurons/corpus/long.npz",
                    help="held-out seasonal series (period 8..64) of this corpus, first 2048 steps, serve as in-prior reference contexts")
    ap.add_argument("--n-ref", type=int, default=64)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    set_threads()
    dev = a.device or default_device()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cache = os.path.join(a.out, "failure_inputs.npz")           # contexts, futures and oracle medians do not depend on the size
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        ctx, fut, orc, cfg, sids = z["ctx"], z["fut"], z["orc"], z["cfg"], z["sid"]
    else:
        S = build_samples()
        ctx = np.stack([s.context[0, -T:] for s in S]).astype(np.float32)
        fut = np.stack([s.future[0, :H] for s in S])
        orc = np.stack([s.oracle_q[0, :H, MEDIAN_INDEX] for s in S])
        cfg = np.array([s.meta["config_id"] for s in S])
        sids = np.array([s.meta["sample_id"] for s in S])
        np.savez_compressed(cache, ctx=ctx, fut=fut, orc=orc, cfg=cfg, sid=sids)
    scale = naive_scale(ctx.astype(float), 1)
    tmpl = np.stack([template_forecast(c.astype(float), H, dominant_period(c)) for c in ctx])
    ana = np.stack([analog_forecast(c.astype(float), H) for c in ctx])
    print(f"{len(ctx)} samples built in {time.time() - t0:.0f}s", flush=True)

    base = forecasts(a.size, ctx, [[]], dev)[0]
    m0 = per_sample_metrics(base, fut, orc, ana, tmpl, scale)
    ana_m = np.abs(ana - fut).mean(1) / scale
    orc_m = np.abs(orc - fut).mean(1) / scale
    gm = lambda x: float(np.exp(np.log(np.clip(x, 1e-12, None)).mean()))         # noqa: E731
    status = {}
    for c_ in sorted(set(cfg)):
        i = cfg == c_
        ratio = gm(m0["mase"][i]) / gm(orc_m[i])
        status[c_] = "failing" if (ratio >= 1.5 and gm(ana_m[i]) < gm(m0["mase"][i])) else ("reference" if ratio < 1.5 else "other")
    st = np.array([status[c_] for c_ in cfg], dtype="U9")
    # in-prior seasonal reference contexts (activations only; no realized future is scored)
    ref_ctx = np.zeros((0, T), np.float32)
    if a.ref_corpus and os.path.exists(a.ref_corpus):
        cc, _ = C.load_corpus(a.ref_corpus)
        sc_ = {n: i for i, n in enumerate(V.SERIES_COLS)}
        per_ = 2.0 ** cc["sv"][:, sc_["s_log2_period"]]
        held = np.where((cc["split"] == 1) & (cc["sv"][:, sc_["s_has_season"]] == 1) & (per_ >= 8) & (per_ <= 64))[0]
        pick = np.sort(np.random.default_rng(St.SAMPLE_SEED).choice(held, min(a.n_ref, len(held)), replace=False))
        ref_ctx = cc["values"][pick, :T].astype(np.float32)
    n_seas = len(ref_ctx)
    ctx_all = np.concatenate([ctx, ref_ctx]) if n_seas else ctx
    st_all = np.concatenate([st, np.array(["reference"] * n_seas, dtype=st.dtype)]) if n_seas else st
    cfg_rows = [{"size": a.size, "config": c_, "status": status[c_], "n": int((cfg == c_).sum()),
                 "mase_timestral": gm(m0["mase"][cfg == c_]), "mase_oracle": gm(orc_m[cfg == c_]),
                 "mase_analog": gm(ana_m[cfg == c_])} for c_ in sorted(set(cfg))]
    pd.DataFrame(cfg_rows).to_csv(os.path.join(a.out, f"failure_configs_{a.size}.csv"), index=False, float_format="%.5g")
    print({k: int((st == k).sum()) for k in ("failing", "reference", "other")}, flush=True)

    # pole activations on these contexts: mean over the last TAIL context patches and the forecast tokens
    n_blocks, hidden = Cap.layer_dims(a.size, dev)
    acc = {"pos": np.zeros((len(ctx_all), n_blocks, hidden), np.float32), "neg": np.zeros((len(ctx_all), n_blocks, hidden), np.float32)}
    import torch
    for i in range(0, len(ctx_all), 32):
        xb = torch.as_tensor(ctx_all[i:i + 32])
        _, zs, _ = Cap.forward_with_mlp(a.size, xb, device=dev)
        for li in range(n_blocks):
            z = zs[li][:, -(TAIL + Cap.N_FC):, :].float()
            acc["pos"][i:i + 32, li] = torch.relu(z).mean(1).cpu().numpy()
            acc["neg"][i:i + 32, li] = torch.relu(-z).mean(1).cpu().numpy()
    t = Sc.add_derived(pd.read_csv(os.path.join(a.runs, f"nn_{a.size}_main", "scores_partial.csv")))
    t["cls"] = Sc.klass(t, 0.7, 0.1)
    fail, ref = st_all == "failing", st_all == "reference"
    rows = []

    def d_stat(x, y):
        sd = np.sqrt((x.var(0, ddof=1) * (len(x) - 1) + y.var(0, ddof=1) * (len(y) - 1)) / max(len(x) + len(y) - 2, 1))
        return (x.mean(0) - y.mean(0)) / np.maximum(sd, 1e-6)

    dmat = {s: d_stat(acc[s][fail], acc[s][ref]) for s in ("pos", "neg")}         # (n_blocks, hidden)
    t["d"] = [dmat[r.sign][r.layer, r.neuron] for r in t.itertuples()]
    t["act_fail"] = [acc[r.sign][fail][:, r.layer, r.neuron].mean() for r in t.itertuples()]
    t["act_ref"] = [acc[r.sign][ref][:, r.layer, r.neuron].mean() for r in t.itertuples()]
    t["periodic"] = t["group"].isin(PERIODIC_GROUPS) & (t["cls"] == "one_job")
    live = t[t["cls"] != "dead"]
    # activation of periodic poles on failing vs reference contexts
    for name, m in (("periodic_one_job", t["periodic"]), ("all_live", t["cls"] != "dead")):
        x = t[m]
        rows.append({"size": a.size, "analysis": "activation", "set": name, "n_poles": len(x),
                     "mean_act_failing": float(x["act_fail"].mean()), "mean_act_reference": float(x["act_ref"].mean()),
                     "share_higher_on_failing": float((x["d"] > 0).mean()), "share_d_ge_thresh": float((x["d"] >= D_MIN).mean()),
                     "share_d_le_neg_thresh": float((x["d"] <= -D_MIN).mean())})
    el = t[(t["d"] >= D_MIN) & (t["cls"] != "dead")]
    base_rate = live["group"].value_counts(normalize=True)
    for g, n in el["group"].value_counts().items():
        rows.append({"size": a.size, "analysis": "elevated", "set": g, "n_poles": int(n), "share_of_elevated": n / len(el),
                     "base_rate": float(base_rate.get(g, 0.0)), "enrichment": float(n / len(el) / max(base_rate.get(g, 0.0), 1e-9))})
    rows.append({"size": a.size, "analysis": "elevated", "set": "all", "n_poles": len(el), "share_of_elevated": 1.0,
                 "base_rate": 1.0, "share_unexplained_or_dead": float((el["cls"] != "one_job").mean())})
    # ablations on failing contexts
    sel_per = t[t["periodic"] & (t["d"] >= D_MIN)].sort_values("d", ascending=False)
    rng = np.random.default_rng(0)
    fi = np.where(fail)[0]                      # indices below len(ctx): reference-only seasonal contexts are never failing
    n_set = len(sel_per)
    sets = {"periodic_elevated": [(int(r.layer), int(r.neuron), 1 if r.sign == "pos" else -1) for r in sel_per.itertuples()]}
    if n_set:
        pool = live[~live["periodic"]]
        # random poles with the same layer counts as the periodic set
        rnd = []
        for lay, k in sel_per["layer"].value_counts().items():
            c_ = pool[pool["layer"] == lay]
            rnd += [(int(r.layer), int(r.neuron), 1 if r.sign == "pos" else -1) for r in c_.iloc[rng.permutation(len(c_))[:k]].itertuples()]
        sets["random_same_layers"] = rnd
    if len(fi) and n_set:
        for name, members in sets.items():
            for sc in (0.0, 0.5):
                cond = [[(l, n, s, sc) for (l, n, s) in members]]
                med = forecasts(a.size, ctx[fi], cond, dev)[0]
                m1 = per_sample_metrics(med, fut[fi], orc[fi], ana[fi], tmpl[fi], scale[fi])
                m_b = {k: v[fi] for k, v in m0.items()}
                rows.append({"size": a.size, "analysis": "ablate", "set": name, "scale": sc, "n_poles": len(members), "n_samples": len(fi),
                             "mase_before": float(m_b["mase"].mean()), "mase_after": float(m1["mase"].mean()),
                             "dist_oracle_before": float(m_b["dist_oracle"].mean()), "dist_oracle_after": float(m1["dist_oracle"].mean()),
                             "dist_analog_before": float(m_b["dist_analog"].mean()), "dist_analog_after": float(m1["dist_analog"].mean()),
                             "corr_template_before": float(np.nanmean(m_b["corr_template"])), "corr_template_after": float(np.nanmean(m1["corr_template"])),
                             "share_closer_to_oracle": float((m1["dist_oracle"] < m_b["dist_oracle"]).mean()),
                             "share_closer_to_analog": float((m1["dist_analog"] < m_b["dist_analog"]).mean()),
                             "share_lower_mase": float((m1["mase"] < m_b["mase"]).mean())})
        for r in sel_per.head(10).itertuples():
            cond = [[(int(r.layer), int(r.neuron), 1 if r.sign == "pos" else -1, 0.0)]]
            med = forecasts(a.size, ctx[fi], cond, dev)[0]
            m1 = per_sample_metrics(med, fut[fi], orc[fi], ana[fi], tmpl[fi], scale[fi])
            m_b = {k: v[fi] for k, v in m0.items()}
            rows.append({"size": a.size, "analysis": "single", "set": f"L{r.layer}.{r.neuron}.{r.sign}:{r.label}", "scale": 0.0, "n_poles": 1,
                         "n_samples": len(fi), "d": float(r.d), "mase_before": float(m_b["mase"].mean()), "mase_after": float(m1["mase"].mean()),
                         "dist_oracle_before": float(m_b["dist_oracle"].mean()), "dist_oracle_after": float(m1["dist_oracle"].mean()),
                         "dist_analog_before": float(m_b["dist_analog"].mean()), "dist_analog_after": float(m1["dist_analog"].mean())})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"failure_link_{a.size}.csv"), index=False, float_format="%.5g")
    per = pd.DataFrame({"size": a.size, "sample_id": sids, "config": cfg, "status": st,
                        **{f"{k}_timestral": v for k, v in m0.items()}, "mase_oracle": orc_m, "mase_analog": ana_m})
    per.to_csv(os.path.join(a.out, f"failure_samples_{a.size}.csv"), index=False, float_format="%.5g")
    write_json(os.path.join(a.out, f"meta_{a.size}.json"), provenance(
        "scripts/neurons_failure.py", size=a.size, device=dev, wall_s=round(time.time() - t0, 1), n_samples=len(ctx), n_seasonal_reference=n_seas, ref_corpus=a.ref_corpus, T=T, H=H,
        n_paths=N_PATHS, seeds=SEEDS, d_min=D_MIN, tail_patches=TAIL, status_counts={k: int((st_all == k).sum()) for k in ("failing", "reference", "other")}))
    print("done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
