"""Build the writeup from committed result files.

paper/src.md holds the text with placeholders:
  {{num:KEY}}    a number computed below from a result file
  {{table:KEY}}  a markdown table computed below from a result file
This script fills them, writes paper/off_the_prior.md, and writes paper/numbers.json: for every
key, the value, the source file(s) and a one-line description of how it was computed. Any
placeholder without a definition stops the build, so no number can be typed in by hand.

python paper/build.py
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from otp.analysis import ratios  # noqa: E402

NUMS: dict[str, dict] = {}
TABLES: dict[str, dict] = {}
SIZES = ["timestral-5M", "timestral-28M", "timestral-123M", "timestral-209M"]
SIZE_LABEL = {s: s.replace("timestral-", "") for s in SIZES}


def p(*parts):
    return os.path.join(ROOT, *parts)


def gm(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else float("nan")


def num(key, value, src, how, fmt="{:.2f}"):
    NUMS[key] = {"value": value, "text": fmt.format(value), "src": src, "how": how}


def table(key, df: pd.DataFrame, src, how, floatfmt="{:.2f}"):
    def f(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("–" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    int_cols = {c for c in df.columns if pd.api.types.is_integer_dtype(df[c])}
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(int(v)) if c in int_cols else f(v) for c, v in zip(df.columns, row)) + " |")
    TABLES[key] = {"text": "\n".join(lines), "src": src, "how": how}


def load_summary(path):
    s = pd.read_csv(p(path))
    s["knob_value"] = s["knob_value"].astype(str)
    return s


# --------------------------------------------------------------------------------------------
def in_prior():
    src = "results/map/in_prior/summary.csv"
    s = load_summary(src)
    g = s.groupby(["forecaster", "T", "H"]).mase_ratio_oracle_gmean.apply(gm).reset_index()
    short = g[g.H <= 128]
    num("inprior_short_min", short.mase_ratio_oracle_gmean.min(), src, "min over sizes, T, H<=128 of the geometric mean over configs of the per-config geometric-mean MASE ratio to oracle")
    num("inprior_short_max", short.mase_ratio_oracle_gmean.max(), src, "max of the same")
    rows = []
    for fc in SIZES:
        r = {"size": SIZE_LABEL[fc]}
        for h in (16, 128, 1024, 2048):
            v = g[(g.forecaster == fc) & (g["T"] == 2048) & (g.H == h)].mase_ratio_oracle_gmean
            r[f"H={h}"] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    table("inprior", pd.DataFrame(rows), src, "geometric mean over the 4 in-prior configs of the per-config geometric-mean MASE ratio to oracle, context 2048")


def family_table(src, key, configs, T=2048, H=128, label=None):
    s = load_summary(src)
    q = s[(s["T"] == T) & (s.H == H) & s.config_id.isin(configs)]
    rows = []
    for cid in configs:
        r = {"series": label(cid) if label else cid}
        for fc in SIZES:
            v = q[(q.config_id == cid) & (q.forecaster == fc)].mase_ratio_oracle_gmean
            r[SIZE_LABEL[fc]] = float(v.iloc[0]) if len(v) else float("nan")
        v = q[(q.config_id == cid) & (q.forecaster == "timestral-209M")].mase_ratio_classical_gmean
        r["209M vs best classical"] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    table(key, pd.DataFrame(rows), src, f"per-config geometric mean over seeds of MASE ratio to oracle (and to the best classical model), context {T}, horizon {H}")
    return q


def chaos():
    src = "results/map/chaos/summary_all_sizes.csv"
    cfg = ["lorenz63/rho20", "lorenz63/rho28", "lorenz63/rho28_oe2", "rossler/c3", "rossler/c5.7", "rossler/c9",
           "mackey_glass/tau10", "mackey_glass/tau17", "mackey_glass/tau30", "logistic_map/r3.5", "logistic_map/r3.9",
           "kuramoto/K1", "kuramoto/K8"]
    q = family_table(src, "chaos_main", cfg)
    v = q[q.forecaster == "timestral-209M"].mase_ratio_classical_gmean
    num("chaos_209m_vs_classical_median", float(np.median(v)), src, "median over chaos configs of the 209M geometric-mean MASE ratio to the best classical model, T=2048, H=128")
    for cid, k in [("mackey_glass/tau30", "mg30"), ("kuramoto/K1", "kur1"), ("kuramoto/K8", "kur8"), ("rossler/c9", "ros9"),
                   ("lorenz63/rho28_oe2", "lor_oe2")]:
        for fc in ("timestral-5M", "timestral-209M"):
            r = q[(q.config_id == cid) & (q.forecaster == fc)]
            num(f"chaos_{k}_{SIZE_LABEL[fc]}", float(r.mase_ratio_oracle_gmean.iloc[0]), src, f"{cid}, {fc}, T=2048, H=128, geometric-mean MASE ratio to oracle")
    # Lyapunov-time view for Lorenz rho=28 at four sampling intervals
    s = load_summary(src)
    dt = {"lorenz63/rho28_oe2": 0.02, "lorenz63/rho28_oe5": 0.05, "lorenz63/rho28": 0.1, "lorenz63/rho28_oe20": 0.2}
    lyap = pd.read_csv(p("results/map/chaos/lyapunov.csv"))
    lam = float(lyap[lyap.config_id == "lorenz63/rho28"].lyap_per_time.mean())
    num("lorenz_lyap", lam, "results/map/chaos/lyapunov.csv", "largest Lyapunov exponent per time unit, Lorenz rho=28, mean over seeds", "{:.3f}")
    r = s[(s["T"] == 2048) & (s.forecaster == "timestral-209M") & s.config_id.isin(dt)].copy()
    r["lyapunov_times"] = r.config_id.map(dt) * r.H.astype(float) * lam
    r = r.sort_values("lyapunov_times")
    out = pd.DataFrame({"sampling step (time units)": r.config_id.map(dt).values, "horizon (steps)": r.H.astype(int).values,
                        "horizon (Lyapunov times)": r.lyapunov_times.values, "209M MASE / oracle": r.mase_ratio_oracle_gmean.values})
    table("lorenz_lyap", out, src + " + results/map/chaos/lyapunov.csv", "209M, context 2048; Lyapunov times = sampling step x horizon x exponent")


def tails():
    src = "results/map/tails/summary_all_sizes.csv"
    s = load_summary(src)
    q = s[(s["T"] == 2048) & (s.H == 128)]
    num("tails_min", q.mase_ratio_oracle_gmean.min(), src, "min over all tails configs and sizes, T=2048, H=128")
    num("tails_max", q.mase_ratio_oracle_gmean.max(), src, "max of the same")
    cfg = ["tails/stable_ar/alpha2_rho0.9", "tails/stable_ar/alpha1.3_rho0.9", "tails/stable_ar/alpha1.3_rho1.0",
           "tails/fbm/H0.2", "tails/fbm/H0.8", "tails/hawkes/eta0.3", "tails/hawkes/eta0.95", "tails/msm/k8_ret", "tails/msm/k8_cum"]
    family_table(src, "tails_main", cfg)


def long_ctx():
    src = "results/map/long_ctx/scores.csv"
    df = ratios(pd.read_csv(p(src)))
    ts = df[df.forecaster.isin(SIZES) & (df.H == 128)]
    g = ts.groupby(["config_id", "forecaster", "T"]).mase_ratio_oracle.apply(gm).reset_index()
    rows = []
    for cid in ["mackey_glass/tau17", "kuramoto/K1", "rossler/c5.7", "lorenz63/rho28", "regime/mean_jump/context/delta4/d64",
                "tails/fbm/H0.8", "in_prior/p24"]:
        for fc in ("timestral-5M", "timestral-209M"):
            r = {"series": cid, "size": SIZE_LABEL[fc]}
            for t in (2048, 8192, 32768):
                v = g[(g.config_id == cid) & (g.forecaster == fc) & (g["T"] == t)].mase_ratio_oracle
                r[f"T={t}"] = float(v.iloc[0]) if len(v) else float("nan")
            rows.append(r)
    table("long_ctx", pd.DataFrame(rows), src, "geometric mean over 8 seeds of MASE ratio to oracle at H=128 (oracle rows matched per context length)")
    for cid, k in [("mackey_glass/tau17", "mg17")]:
        for fc in ("timestral-5M", "timestral-209M"):
            for t in (2048, 32768):
                v = float(g[(g.config_id == cid) & (g.forecaster == fc) & (g["T"] == t)].mase_ratio_oracle.iloc[0])
                num(f"long_{k}_{SIZE_LABEL[fc]}_T{t}", v, src, f"{cid}, {fc}, T={t}, H=128")
    bad = df[(df.forecaster == "oracle") & ~np.isfinite(df.mase)].sample_id.nunique()
    num("long_nonfinite_oracle_samples", bad, src, "samples with a non-finite oracle (diverged Monte Carlo paths)", "{:d}")


def symavg():
    src = "results/mechanism/symavg/off_over_on.csv"
    w = pd.read_csv(p(src))
    q = w[w.H == 128]
    num("symavg_min", q.scrps_off_over_on.min(), src, "min over configs and sizes of sCRPS ratio (averaging off / on), H=128", "{:.3f}")
    num("symavg_max", q.scrps_off_over_on.max(), src, "max of the same", "{:.3f}")


def normalization():
    src = "results/mechanism/normalization/summary.csv"
    s = pd.read_csv(p(src))
    full = s[s.condition == "full"].set_index(["case", "T", "size"])
    post = s[s.condition == "post_only"].set_index(["case", "T", "size"])
    j = full.join(post, lsuffix="_full", rsuffix="_post")
    num("norm_cells", len(j), src, "case x context x size cells with both conditions", "{:d}")
    num("norm_post_better", int((j.scrps_ratio_gmean_post < j.scrps_ratio_gmean_full).sum()), src, "cells where cutting to the post-switch part lowered the sCRPS ratio", "{:d}")
    v = s[(s.case == "variance_jump/f0.25/d64") & (s.condition == "full")]
    num("norm_vdrop_cov80_min", v.cov80.min(), src, "variance drop x0.25, 64 steps back, full context: min cov80 over sizes and contexts", "{:.3f}")
    num("norm_vdrop_scrps_min", v.scrps_ratio_gmean.min(), src, "same cells: min sCRPS ratio to oracle")
    num("norm_vdrop_scrps_max", v.scrps_ratio_gmean.max(), src, "same cells: max sCRPS ratio to oracle")


def analog():
    src = "results/mechanism/analog/summary.csv"
    s = pd.read_csv(p(src))
    cfg = ["mackey_glass/tau17", "mackey_glass/tau30", "rossler/c5.7", "rossler/c9", "kuramoto/K1", "kuramoto/K8",
           "lorenz63/rho28", "logistic_map/r3.9"]
    cols = [c for c in ["mase_oracle", "mase_analog", "mase_template", "mase_5M", "mase_28M", "mase_123M", "mase_209M"] if c in s]
    out = s[s.config_id.isin(cfg)].set_index("config_id").loc[cfg, cols].reset_index()
    out.columns = ["series"] + [c.replace("mase_", "") for c in cols]
    table("analog", out, src, "geometric mean over 8 seeds of MASE (context 2048, horizon 128) for the oracle, the method of analogues, the seasonal template, and Timestral")


def chaos_noise():
    src = "results/mechanism/chaos_noise/summary.csv"
    s = load_summary(src)
    s["system"] = s.config_id.str.split("/").str[1]
    s["noise"] = s.config_id.str.extract(r"/n([\d.]+)$")[0].astype(float)
    q = s[s.H == 128]
    rows = []
    for (sysname, nz), g in q.groupby(["system", "noise"]):
        r = {"system": sysname, "noise (% of scale)": nz * 100}
        for fc in SIZES:
            v = g[g.forecaster == fc].mase_ratio_oracle_gmean
            r[SIZE_LABEL[fc]] = float(v.iloc[0]) if len(v) else float("nan")
        rows.append(r)
    table("chaos_noise", pd.DataFrame(rows), src, "geometric mean over 16 seeds of MASE ratio to oracle, context 2048, horizon 128")
    mg = q[q.system == "mackey_glass_tau17"]
    for nz, k in [(0.001, "lo"), (0.1, "hi")]:
        v = mg[mg.noise == nz].mase_ratio_oracle_gmean
        num(f"noise_mg_{k}_min", float(v.min()), src, f"Mackey-Glass tau=17, noise {nz}, min over sizes")
        num(f"noise_mg_{k}_max", float(v.max()), src, f"Mackey-Glass tau=17, noise {nz}, max over sizes")


def readouts():
    src = "results/mechanism/readouts/summary.csv"
    s = pd.read_csv(p(src))
    s = s[s.valid]
    for study, kv, k in [("mean_jump_d", 16.0, "mj16"), ("mean_jump_d", 256.0, "mj256"), ("freq_jump_d", 64.0, "fj64"),
                         ("mackey_tau", 17.0, "mg17")]:
        g = s[(s.study == study) & (s.knob_value == kv)]
        for col, name in [("output_r2", "out"), ("best_readout_r2", "ro"), ("mase_ratio_oracle_gmean", "ratio"),
                          ("step1_output_r2", "s1out"), ("step1_readout_r2", "s1ro"), ("attn_post_change_mean", "post"),
                          ("template_corr_model", "tm"), ("template_corr_oracle", "to")]:
            if col in g and g[col].notna().any():
                num(f"ro_{k}_{name}_min", float(g[col].min()), src, f"{study} {kv}: min over the four sizes of {col}", "{:.3f}" if "r2" in col or "corr" in col else "{:.2f}")
                num(f"ro_{k}_{name}_max", float(g[col].max()), src, f"{study} {kv}: max over the four sizes of {col}", "{:.3f}" if "r2" in col or "corr" in col else "{:.2f}")
    g = s[(s.study == "in_prior") & (s.knob_value == 24.0)]
    for col, name in [("output_r2", "out"), ("best_readout_r2", "ro")]:
        num(f"ro_ip_{name}_min", float(g[col].min()), src, f"in_prior period 24: min over sizes of {col}", "{:.3f}")
        num(f"ro_ip_{name}_max", float(g[col].max()), src, f"in_prior period 24: max over sizes of {col}", "{:.3f}")
    for size in ("5M", "209M"):
        g = s[(s.study == "mackey_tau") & (s.knob_value == 17.0) & (s["size"] == size)]
        num(f"ro_mg17_tm_{size}", float(g.template_corr_model.iloc[0]), src, f"Mackey-Glass tau=17, {size}: correlation of Timestral's median with the periodic template", "{:.2f}")
    rows = []
    for study, kv in [("mean_jump_d", 16.0), ("mean_jump_d", 64.0), ("mean_jump_d", 256.0), ("freq_jump_d", 64.0),
                      ("freq_jump_d", 1024.0), ("mackey_tau", 17.0), ("lorenz_rho", 28.0), ("in_prior", 24.0)]:
        for size in ("5M", "209M"):
            g = s[(s.study == study) & (s.knob_value == kv) & (s["size"] == size)]
            if not len(g):
                continue
            r = g.iloc[0]
            rows.append({"setting": f"{study} {kv:g}", "size": size, "MASE / oracle": r.mase_ratio_oracle_gmean,
                         "output R2": r.output_r2, "best readout R2": r.best_readout_r2,
                         "step-1 output R2": r.get("step1_output_r2", np.nan), "step-1 readout R2": r.get("step1_readout_r2", np.nan)})
    table("readouts", pd.DataFrame(rows), src, "300 samples per setting; cross-validated R2 of predicting the oracle's expected change (first patch, or step 1) from Timestral's output vs. from the best per-layer linear readout of the first forecast token", "{:.3f}")


def selfval():
    src = "results/fix/selfval_confirm/frozen_rule_confirmation.csv"
    r = pd.read_csv(p(src))
    rows = []
    for fam in ("chaos", "regime", "tails", "in_prior"):
        row = {"family": fam}
        for fc in SIZES:
            v = r[(r.family == fam) & (r["size"] == SIZE_LABEL[fc])].mase_fix_over_timestral
            row[SIZE_LABEL[fc]] = float(v.iloc[0]) if len(v) else float("nan")
        v = r[r.family == fam].switched_share
        row["share switched (range over sizes)"] = f"{v.min():.3f}-{v.max():.3f}"
        rows.append(row)
    table("selfval", pd.DataFrame(rows), src, "geometric mean over fresh samples of MASE(fix) / MASE(Timestral); frozen rule K=2, m=0.7", "{:.3f}")
    c = r[r.family == "chaos"].set_index("size").mase_fix_over_timestral
    for size in ("5M", "209M"):
        num(f"sv_chaos_{size}", float(c[size]), src, f"chaos, {size}: MASE(fix)/MASE(Timestral) on fresh seeds", "{:.3f}")
    t = r[r.family == "tails"].mase_fix_over_timestral
    num("sv_tails_max", float(t.max()), src, "tails: max over sizes of MASE(fix)/MASE(Timestral)", "{:.3f}")
    s2 = pd.read_csv(p("results/fix/selfval/chosen_rule_test_half.csv"))
    tt = s2[s2.family == "tails"].mase_fix_over_timestral
    num("sv_firstrule_tails_min", float(tt.min()), "results/fix/selfval/chosen_rule_test_half.csv", "first pre-set rule, odd seeds: min over sizes, tails", "{:.3f}")
    num("sv_firstrule_tails_max", float(tt.max()), "results/fix/selfval/chosen_rule_test_half.csv", "first pre-set rule, odd seeds: max over sizes, tails", "{:.3f}")


def regime():
    src = "results/map/regime/summary_all_sizes.csv"
    s = load_summary(src)
    q = s[(s["T"] == 2048) & (s.H == 128)].copy()
    q["d"] = pd.to_numeric(q.config_id.str.extract(r"/d(\d+)$")[0], errors="coerce")
    rows = []
    for prefix, label in [("regime/mean_jump/context/delta4/", "level +4 sd"), ("regime/frequency_jump/context/ratio2/", "period x2"),
                          ("regime/variance_jump/context/factor0.25/", "noise sd x0.25")]:
        for fc in ("timestral-5M", "timestral-209M"):
            r = {"switch": label, "size": SIZE_LABEL[fc]}
            for d in (16, 64, 256, 1024):
                v = q[(q.config_id == f"{prefix}d{d}") & (q.forecaster == fc)].mase_ratio_oracle_gmean
                r[f"{d} steps ago"] = float(v.iloc[0]) if len(v) else float("nan")
            rows.append(r)
    table("regime", pd.DataFrame(rows), src, "geometric mean over 8 seeds of MASE ratio to oracle, context 2048, horizon 128")
    for gen, kv, d, k in [("mean_jump", "delta4", 16, "mj4_16"), ("mean_jump", "delta4", 1024, "mj4_1024"), ("frequency_jump", "ratio3", 64, "fj3_64")]:
        for fc in ("timestral-5M", "timestral-209M"):
            v = float(q[(q.config_id == f"regime/{gen}/context/{kv}/d{d}") & (q.forecaster == fc)].mase_ratio_oracle_gmean.iloc[0])
            num(f"reg_{k}_{SIZE_LABEL[fc]}", v, src, f"{gen} {kv} switch {d} steps back, {fc}, T=2048, H=128")
    hz = q[q.config_id.str.contains("/horizon/")].mase_ratio_oracle_gmean
    num("reg_horizon_min", hz.min(), src, "switches during the horizon (all generators, knob values, sizes): min")
    num("reg_horizon_max", hz.max(), src, "same: max")
    mk = q[q.generator == "markov_ar"].mase_ratio_oracle_gmean
    num("reg_markov_min", mk.min(), src, "Markov-switching AR (all switching rates, sizes): min")
    num("reg_markov_max", mk.max(), src, "same: max")
    asrc = "results/map/regime/adoption.csv"
    a = pd.read_csv(p(asrc))
    a = a[(a["T"] == 2048) & (a.delta.astype(str) == "all")]
    for size in ("5M", "209M"):
        for d in (16, 64, 256):
            v = float(a[(a["size"] == size) & (a.d == d)].adoption_mean.iloc[0])
            num(f"adopt_{size}_{d}", v, asrc, f"level adoption at H<=16, all deltas pooled, {size}, switch {d} steps back, T=2048")


def bench():
    import json as _json

    order = ["oracle", "timestral-209M", "timestral-123M", "timestral-28M", "timestral-5M", "ets", "arima", "seasonal_naive"]
    for set_name in ("diag", "lb"):
        rows, srcs = [], []
        for m in order:
            f = f"bench/leaderboard/{m}_{set_name}_all/summary.json"
            if not os.path.exists(p(f)):
                continue
            o = _json.load(open(p(f)))["overall"]
            srcs.append(f)
            rows.append({"model": m, "series": int(o["n"]), "MASE / seasonal naive": o["mase_rel_snaive"],
                         "sCRPS / seasonal naive": o["scrps_rel_snaive"], "MASE / oracle": o["mase_rel_oracle"],
                         "80% coverage": o["cov80"]})
        if rows:
            table(f"bench_{set_name}", pd.DataFrame(rows), ", ".join(srcs), "bench/eval.py summaries (geometric means over series)", "{:.3f}")
    f = "bench/leaderboard/timestral-209M_diag_all/summary.json"
    if os.path.exists(p(f)):
        ft = _json.load(open(p(f)))["by_failure_type"]
        rows = [{"failure type": k, "series": int(v["n"]), "209M MASE / oracle": v["mase_rel_oracle"],
                 "209M MASE / seasonal naive": v["mase_rel_snaive"]} for k, v in sorted(ft.items())]
        table("bench_diag_ft", pd.DataFrame(rows), f, "Timestral-209M on OffPrior-Diag by failure type", "{:.3f}")
    for set_name in ("diag", "lb"):
        mf = f"bench/data/{set_name}_manifest.csv"
        if os.path.exists(p(mf)):
            num(f"bench_n_{set_name}", len(pd.read_csv(p(mf))), mf, "number of series", "{:d}")


def main():
    for fn in (in_prior, chaos, tails, long_ctx, symavg, normalization, analog, chaos_noise, readouts, selfval, regime, bench):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
    src = open(p("paper", "src.md")).read()
    missing = []

    def sub(m):
        kind, key = m.group(1), m.group(2)
        d = (NUMS if kind == "num" else TABLES).get(key)
        if d is None:
            missing.append(m.group(0))
            return m.group(0)
        return d["text"]

    out = re.sub(r"\{\{(num|table):([A-Za-z0-9_]+)\}\}", sub, src)
    if missing:
        raise SystemExit(f"undefined placeholders: {sorted(set(missing))}")
    open(p("paper", "off_the_prior.md"), "w").write(out)
    audit = {k: {kk: (vv if kk != "value" else float(vv)) for kk, vv in v.items()} for k, v in NUMS.items()}
    audit.update({f"table:{k}": {"src": v["src"], "how": v["how"]} for k, v in TABLES.items()})
    with open(p("paper", "numbers.json"), "w") as f:
        f.write(json.JSONEncoder(indent=1).encode(audit) + "\n")
    print(f"wrote paper/off_the_prior.md ({len(NUMS)} numbers, {len(TABLES)} tables)")


if __name__ == "__main__":
    main()
